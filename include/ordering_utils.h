#ifndef AMIGO_ORDERING_UTILS_H
#define AMIGO_ORDERING_UTILS_H

#include <algorithm>

#include "block_amd.h"

#ifdef AMIGO_USE_METIS
extern "C" {
#include "metis.h"
}
#endif

namespace amigo {

enum class OrderingType { NESTED_DISSECTION, AMD, NATURAL };

class OrderingUtils {
 public:
  static void reorder(OrderingType order, int nrows, int *rowp, int *cols,
                      int **perm_, int **iperm_) {
    if (order == OrderingType ::NESTED_DISSECTION) {
      nested_dissection(nrows, rowp, cols, perm_, iperm_);
    } else if (order == OrderingType ::AMD) {
      amd(nrows, rowp, cols, 0, nullptr, perm_, iperm_);
    } else {  // order == OrderingType::NATURAL
      // Natural ordering
      int *perm = new int[nrows];
      int *iperm = new int[nrows];
      for (int i = 0; i < nrows; i++) {
        perm[i] = iperm[i] = i;
      }
      *perm_ = perm;
      *iperm_ = iperm;
    }
  }

  static void reorder_block(OrderingType order, int nrows, int *rowp, int *cols,
                            int nmult, int *mult, int **perm_, int **iperm_) {
    if (order == OrderingType ::NESTED_DISSECTION ||
        order == OrderingType::NATURAL) {
      reorder(order, nrows, rowp, cols, perm_, iperm_);
      int *perm = *perm_;
      int *iperm = *iperm_;

      int *is_mult = new int[nrows];
      std::fill(is_mult, is_mult + nrows, 0);
      for (int i = 0; i < nrows; i++) {
        if (mult[i] >= 0 && mult[i] < nrows) {
          is_mult[mult[i]] = 1;
        }
      }

      // Perform a stable partition on perm to create the new output
      std::stable_partition(perm, perm + nrows,
                            [&](int index) { return !is_mult[index]; });

      for (int i = 0; i < nrows; i++) {
        iperm[perm[i]] = i;
      }

      delete[] is_mult;
    } else {  // order == OrderingType::AMD
      amd(nrows, rowp, cols, nmult, mult, perm_, iperm_);
    }
  }

  /**
   * @brief Compute a nested dissection ordering.
   *
   * Note that the CSR structure must not include the diagonal element (compute
   * with include_diagonal = false).
   *
   * new_var = iperm[old_var]
   * perm[new_var] = old_var
   *
   * Note that for a variable k, these arrays satisfy iperm[perm[k]] = k
   */
  static void nested_dissection(int nrows, int *rowp, int *cols, int **perm_,
                                int **iperm_) {
    int *perm = new int[nrows];
    int *iperm = new int[nrows];
#ifdef AMIGO_USE_METIS
    // Set the default options in METIS
    int options[METIS_NOPTIONS];
    METIS_SetDefaultOptions(options);

    // Use 0-based numbering
    options[METIS_OPTION_NUMBERING] = 0;
    METIS_NodeND(&nrows, rowp, cols, NULL, options, perm, iperm);
#else
    for (int i = 0; i < nrows; i++) {
      perm[i] = i;
      iperm[i] = i;
    }
#endif
    *perm_ = perm;
    *iperm_ = iperm;
  }

  /**
   * @brief Perform an AMD reordering
   */
  static void amd(int nrows, int *rowp, int *cols, int nmult, int *mult,
                  int **perm_, int **iperm_) {
    int *perm = new int[nrows];
    int *iperm = new int[nrows];

    int use_exact_degree = 0;
    BlockAMD::amd(nrows, rowp, cols, nmult, mult, perm, use_exact_degree);

    for (int i = 0; i < nrows; i++) {
      iperm[perm[i]] = i;
    }

    *perm_ = perm;
    *iperm_ = iperm;
  }

  /**
   * @brief Compute the CSR data based on the element -> node information
   *
   * @tparam Functor Class type for the functor
   * @param nnodes Number of nodes in the mesh
   * @param nelems Number of elements
   * @param element_nodes Functor returning the number of nodes and node numbers
   * @param include_diagonal Include the diagonal element in the entries
   * @param sort_columns Sort the columns of the matrix in each row
   * @param rowp_ Output pointer into the columns array
   * @param cols_ Array of column indices
   */
  template <class Functor>
  static void create_csr_from_elements(int nrows, int ncols, int nelems,
                                       const Functor &element_nodes,
                                       bool include_diagonal, bool sort_columns,
                                       int **rowp_, int **cols_) {
    int *node_to_elem_ptr = nullptr;
    int *node_to_elem = nullptr;
    compute_node_to_element_ptr(ncols, nelems, element_nodes, &node_to_elem_ptr,
                                &node_to_elem);

    // Set up the CSR data structure
    int *rowp = new int[nrows + 1];
    int *counter = new int[ncols];

    // Initialize the counter
    for (int i = 0; i < ncols; i++) {
      counter[i] = -1;
    }

    // Count up the number of non-zero entries
    rowp[0] = 0;
    int nnz_count = 0;
    for (int i = 0; i < nrows; i++) {
      // Count the diagonal first
      counter[i] = i;
      if (include_diagonal) {
        nnz_count++;
      }

      // Count the off-diagonals
      for (int j = node_to_elem_ptr[i]; j < node_to_elem_ptr[i + 1]; j++) {
        int elem = node_to_elem[j];

        const int *ptr;
        int nodes_per_element = element_nodes(elem, &ptr);
        for (int k = 0; k < nodes_per_element; k++, ptr++) {
          int node = ptr[0];
          if (counter[node] < i) {
            counter[node] = i;
            nnz_count++;
          }
        }
      }
      rowp[i + 1] = nnz_count;
    }

    // Allocate the column indices
    int *cols = new int[nnz_count];

    // Reset the counter
    for (int i = 0; i < ncols; i++) {
      counter[i] = -1;
    }

    // Count up the number of non-zero entries
    nnz_count = 0;
    for (int i = 0; i < nrows; i++) {
      // Add the diagonal
      counter[i] = i;
      if (include_diagonal) {
        cols[nnz_count] = i;
        nnz_count++;
      }

      // Add the off-diagonals
      for (int j = node_to_elem_ptr[i]; j < node_to_elem_ptr[i + 1]; j++) {
        int elem = node_to_elem[j];

        const int *ptr;
        int nodes_per_element = element_nodes(elem, &ptr);
        for (int k = 0; k < nodes_per_element; k++, ptr++) {
          int node = ptr[0];
          if (counter[node] < i) {
            counter[node] = i;
            cols[nnz_count] = node;
            nnz_count++;
          }
        }
      }

      // Sort the column indices for later use
      if (sort_columns) {
        std::sort(&cols[rowp[i]], &cols[rowp[i + 1]]);
      }
    }

    // Free unused data
    delete[] node_to_elem_ptr;
    delete[] node_to_elem;
    delete[] counter;

    *rowp_ = rowp;
    *cols_ = cols;
  }

  /**
   * @brief Given the element to nodal information, compute a way to quickly
   * find the elements that touch a given node
   *
   * @tparam Functor Class type for the functor
   * @param nnodes Number of nodes in the mesh
   * @param nelems Number of elements
   * @param element_nodes Functor returning the number of nodes and node numbers
   * @param node_to_elem_ptr_ The node to element pointer
   * @param node_to_elem_ The node to element data
   */
  template <class Functor>
  static void compute_node_to_element_ptr(int nnodes, int nelems,
                                          const Functor &element_nodes,
                                          int **node_to_elem_ptr_,
                                          int **node_to_elem_) {
    // Create data to store node -> element connectivity
    int *node_to_elem_ptr = new int[nnodes + 1];
    for (int i = 0; i < nnodes + 1; i++) {
      node_to_elem_ptr[i] = 0;
    }

    for (int i = 0; i < nelems; i++) {
      const int *ptr;
      int nodes_per_element = element_nodes(i, &ptr);
      for (int j = 0; j < nodes_per_element; j++, ptr++) {
        node_to_elem_ptr[ptr[0] + 1]++;
      }
    }

    for (int i = 0; i < nnodes; i++) {
      node_to_elem_ptr[i + 1] += node_to_elem_ptr[i];
    }

    // Set up the node to element data
    int *node_to_elem = new int[node_to_elem_ptr[nnodes]];
    for (int i = 0; i < nelems; i++) {
      const int *ptr;
      int nodes_per_element = element_nodes(i, &ptr);
      for (int j = 0; j < nodes_per_element; j++, ptr++) {
        int node = ptr[0];
        node_to_elem[node_to_elem_ptr[node]] = i;
        node_to_elem_ptr[node]++;
      }
    }

    for (int i = nnodes; i > 0; i--) {
      node_to_elem_ptr[i] = node_to_elem_ptr[i - 1];
    }
    node_to_elem_ptr[0] = 0;

    // Set the outputs
    *node_to_elem_ptr_ = node_to_elem_ptr;
    *node_to_elem_ = node_to_elem;
  }

  /**
   * @brief Find a coloring of the elements that is good for parallel
   * computation
   *
   * @param nelems Number of elements
   * @param nnodes_per_elem Number of components (nodes) for each element
   * @param elem_nodes List of nodes for each element
   * @param num_colors_ Output number of colors
   * @param elem_by_color_ptr_ Pointer into the elem_by_color array
   * @param elem_by_color_ Elements listed by color
   */
  static void color_elements(const int nelems, const int nnodes_per_elem,
                             const int *elem_nodes, int *num_colors_,
                             int **elem_by_color_ptr_, int **elem_by_color_) {
    int *elem_to_elem_ptr, *elem_to_elem;
    build_element_to_element(nelems, nnodes_per_elem, elem_nodes,
                             &elem_to_elem_ptr, &elem_to_elem);

    // Greedy coloring
    int *elem_colors = new int[nelems];
    std::fill(elem_colors, elem_colors + nelems, -1);

    // Keep track of the number of colors
    int num_colors = 0;

    int *flags = new int[nelems];
    std::fill(flags, flags + nelems, -1);

    for (int e = 0; e < nelems; e++) {
      int start = elem_to_elem_ptr[e];
      int end = elem_to_elem_ptr[e + 1];
      for (int k = start; k < end; k++) {
        int neigh = elem_to_elem[k];
        int c = elem_colors[neigh];

        if (c >= 0) {
          flags[c] = e;
        }
      }

      // Find smallest non-conflicting color if any
      int found = false;
      for (int k = 0; k < num_colors; k++) {
        if (flags[k] != e) {
          elem_colors[e] = k;
          found = true;
          break;
        }
      }

      // No color was found, create a new color
      if (!found) {
        elem_colors[e] = num_colors;
        num_colors++;
      }
    }

    // Set up the elements by color
    int *elem_by_color = new int[nelems];
    int *elem_by_color_ptr = new int[num_colors + 1];
    std::fill(elem_by_color_ptr, elem_by_color_ptr + num_colors + 1, 0);
    for (int e = 0; e < nelems; e++) {
      elem_by_color_ptr[elem_colors[e] + 1]++;
    }

    for (int i = 0; i < num_colors; i++) {
      elem_by_color_ptr[i + 1] += elem_by_color_ptr[i];
    }

    for (int e = 0; e < nelems; e++) {
      elem_by_color[elem_by_color_ptr[elem_colors[e]]] = e;
      elem_by_color_ptr[elem_colors[e]]++;
    }

    // Reset the pointer
    for (int i = num_colors - 1; i >= 0; i--) {
      elem_by_color_ptr[i + 1] = elem_by_color_ptr[i];
    }
    elem_by_color_ptr[0] = 0;

    delete[] elem_colors;
    delete[] flags;

    *num_colors_ = num_colors;
    *elem_by_color_ptr_ = elem_by_color_ptr;
    *elem_by_color_ = elem_by_color;
  }

  /**
   * @brief Build an element to neighboring element connectivity
   *
   * @param nelems Number of elements
   * @param nnodes_per_elem Number of nodes for each element
   * @param elem_nodes Nodes for each element
   * @param elem_to_elem_ptr_ Output pointer into the elem_to_elem array
   * @param elem_to_elem_ Output element to element data
   */
  static void build_element_to_element(const int nelems,
                                       const int nnodes_per_elem,
                                       const int *elem_nodes,
                                       int **elem_to_elem_ptr_,
                                       int **elem_to_elem_) {
    int max_node = 0;
    for (int i = 0; i < nelems * nnodes_per_elem; i++) {
      if (elem_nodes[i] > max_node) {
        max_node = elem_nodes[i];
      }
    }
    max_node++;

    // Create a pointer from the nodes back to the elements
    int *node_to_elem_ptr = new int[max_node + 1];
    std::fill(node_to_elem_ptr, node_to_elem_ptr + max_node + 1, 0);

    for (int i = 0; i < nelems * nnodes_per_elem; i++) {
      node_to_elem_ptr[elem_nodes[i] + 1]++;
    }
    for (int i = 0; i < max_node; i++) {
      node_to_elem_ptr[i + 1] += node_to_elem_ptr[i];
    }

    int *node_to_elem = new int[nelems * nnodes_per_elem];

    // Fill in the element numbers
    for (int i = 0; i < nelems; i++) {
      for (int j = 0; j < nnodes_per_elem; j++) {
        int node = elem_nodes[nnodes_per_elem * i + j];
        node_to_elem[node_to_elem_ptr[node]] = i;
        node_to_elem_ptr[node]++;
      }
    }

    // Fix the now broken node_to_elem_ptr array
    for (int i = max_node - 1; i >= 0; i--) {
      node_to_elem_ptr[i + 1] = node_to_elem_ptr[i];
    }
    node_to_elem_ptr[0] = 0;

    // Compute the element -> element data structure
    int *elem_flags = new int[nelems];
    std::fill(elem_flags, elem_flags + nelems, -1);

    int *elem_to_elem_ptr = new int[nelems + 1];
    elem_to_elem_ptr[0] = 0;
    for (int i = 0; i < nelems; i++) {
      int count = 0;

      for (int j = 0; j < nnodes_per_elem; j++) {
        int node = elem_nodes[nnodes_per_elem * i + j];

        // Find the adjacent elements
        int start = node_to_elem_ptr[node];
        int end = node_to_elem_ptr[node + 1];
        for (int k = start; k < end; k++) {
          int e = node_to_elem[k];

          if (e != i && elem_flags[e] != i) {
            count++;
            elem_flags[e] = i;
          }
        }
      }
      elem_to_elem_ptr[i + 1] = count;
    }

    for (int i = 0; i < nelems; i++) {
      elem_to_elem_ptr[i + 1] += elem_to_elem_ptr[i];
    }

    std::fill(elem_flags, elem_flags + nelems, -1);
    int *elem_to_elem = new int[elem_to_elem_ptr[nelems]];
    for (int i = 0; i < nelems; i++) {
      int count = 0;

      for (int j = 0; j < nnodes_per_elem; j++) {
        int node = elem_nodes[nnodes_per_elem * i + j];

        // Find the adjacent elements
        int start = node_to_elem_ptr[node];
        int end = node_to_elem_ptr[node + 1];
        for (int k = start; k < end; k++) {
          int e = node_to_elem[k];

          if (e != i && elem_flags[e] != i) {
            elem_to_elem[elem_to_elem_ptr[i] + count] = e;
            count++;
            elem_flags[e] = i;
          }
        }
      }
    }

    delete[] node_to_elem;
    delete[] node_to_elem_ptr;
    delete[] elem_flags;

    *elem_to_elem_ptr_ = elem_to_elem_ptr;
    *elem_to_elem_ = elem_to_elem;
  }
};

}  // namespace amigo

#endif  // AMIGO_ORDERING_UTILS_H