#ifndef AMIGO_ORDERING_UTILS_H
#define AMIGO_ORDERING_UTILS_H

#include <algorithm>

#include "block_amd.h"

// #ifdef AMIGO_USE_MPI
#include <mpi.h>
// #endif

#ifdef AMIGO_USE_METIS
extern "C" {
#include "metis.h"
}
#endif

namespace amigo {

enum class OrderingType { NESTED_DISSECTION, AMD, MULTI_COLOR, NATURAL };

class OrderingUtils {
 public:
  static void reorder(OrderingType order, int nrows, int* rowp, int* cols,
                      int** perm_, int** iperm_) {
    if (order == OrderingType::NESTED_DISSECTION) {
      nested_dissection(nrows, rowp, cols, perm_, iperm_);
    } else if (order == OrderingType::AMD) {
      amd(nrows, rowp, cols, 0, nullptr, perm_, iperm_);
    } else if (order == OrderingType::MULTI_COLOR) {
      multi_color(nrows, rowp, cols, perm_, iperm_);
    } else {  // order == OrderingType::NATURAL
      // Natural ordering
      int* perm = new int[nrows];
      int* iperm = new int[nrows];
      for (int i = 0; i < nrows; i++) {
        perm[i] = iperm[i] = i;
      }
      *perm_ = perm;
      *iperm_ = iperm;
    }
  }

  static void reorder_block(OrderingType order, int nrows, int* rowp, int* cols,
                            int nmult, int* mult, int** perm_, int** iperm_) {
    if (order == OrderingType::NESTED_DISSECTION ||
        order == OrderingType::NATURAL || order == OrderingType::MULTI_COLOR) {
      reorder(order, nrows, rowp, cols, perm_, iperm_);
      int* perm = *perm_;
      int* iperm = *iperm_;

      int* is_mult = new int[nrows];
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
   * Compute an element partition given the connectivity
   *
   * @tparam Functor Class type for the functor
   * @param nnodes Number of nodes
   * @param nelems Number of elements
   * @param element_nodes Functor returning the number of nodes and node numbers
   * @param part_size Partition size
   * @param partition_ New array of length = nelems with paritition assignments
   */
  template <class Functor>
  static void compute_partition(int nnodes, int nelems,
                                const Functor& element_nodes, int part_size,
                                int* partition_[]) {
    // Create a pointer from the nodes back to the elements
    int *elem_to_elem_ptr, *elem_to_elem;
    build_element_to_element(nnodes, nelems, element_nodes, &elem_to_elem_ptr,
                             &elem_to_elem);

    int* partition = new int[nelems];

#ifdef AMIGO_USE_METIS
    if (part_size > 1) {
      // Partition via METIS
      int ncon = 1;  // "It should be at least 1"??

      // Set the default options
      int options[METIS_NOPTIONS];
      METIS_SetDefaultOptions(options);

      // Use 0-based numbering
      options[METIS_OPTION_NUMBERING] = 0;

      // The objective value in METIS
      int objval = 0;

      if (part_size < 8) {
        METIS_PartGraphRecursive(&nelems, &ncon, elem_to_elem_ptr, elem_to_elem,
                                 NULL, NULL, NULL, &part_size, NULL, NULL,
                                 options, &objval, partition);
      } else {
        METIS_PartGraphKway(&nelems, &ncon, elem_to_elem_ptr, elem_to_elem,
                            NULL, NULL, NULL, &part_size, NULL, NULL, options,
                            &objval, partition);
      }
    } else {
      for (int i = 0; i < nelems; i++) {
        partition[i] = 0;
      }
    }
#else
    for (int i = 0; i < nelems; i++) {
      partition[i] = 0;
    }
#endif  // AMIGO_USE_METIS

    delete[] elem_to_elem_ptr;
    delete[] elem_to_elem;
    *partition_ = partition;
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
  static void nested_dissection(int nrows, int* rowp, int* cols, int** perm_,
                                int** iperm_) {
    int* perm = new int[nrows];
    int* iperm = new int[nrows];
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
  static void amd(int nrows, int* rowp, int* cols, int nmult, int* mult,
                  int** perm_, int** iperm_) {
    int* perm = new int[nrows];
    int* iperm = new int[nrows];

    int use_exact_degree = 0;
    BlockAMD::amd(nrows, rowp, cols, nmult, mult, perm, use_exact_degree);

    for (int i = 0; i < nrows; i++) {
      iperm[perm[i]] = i;
    }

    *perm_ = perm;
    *iperm_ = iperm;
  }

  /**
   * @brief Perform a multicolor ordering of CSR data
   *
   * @param nrows Number of rows in the matrix
   * @param rowp Pointer into the column index
   * @param cols Column array
   * @param perm_ Permutation of the array
   * @param iperm_ Inverse permutation
   */
  static void multi_color(int nrows, const int* rowp, const int* cols,
                          int** perm_, int** iperm_) {
    int* colors = new int[nrows];
    int* tmp = new int[nrows + 1];
    std::fill(colors, colors + nrows, -1);
    std::fill(tmp, tmp + nrows, -1);

    int num_colors = 0;
    for (int i = 0; i < nrows; i++) {
      // Find the minimum color that is not referred to by any adjacent
      // node.
      for (int jp = rowp[i]; jp < rowp[i + 1]; jp++) {
        int j = cols[jp];
        if (colors[j] >= 0) {
          tmp[colors[j]] = i;
        }
      }

      // Set the color for this variable if it already exists
      int flag = 1;
      for (int k = 0; k < num_colors; k++) {
        if (tmp[k] != i) {
          colors[i] = k;
          flag = 0;
          break;
        }
      }

      // Create a new color
      if (flag) {
        colors[i] = num_colors;
        num_colors++;
      }
    }

    // Now that all the nodes have been colored, order them
    std::fill(tmp, tmp + (num_colors + 1), 0);

    // Count up the number of nodes for each color
    for (int i = 0; i < nrows; i++) {
      tmp[colors[i] + 1]++;
    }

    // Set tmp as an offset for each color
    for (int i = 1; i < num_colors + 1; i++) {
      tmp[i] += tmp[i - 1];
    }

    int* iperm = new int[nrows];
    int* perm = new int[nrows];

    // Create the new color variables
    for (int i = 0; i < nrows; i++) {
      iperm[i] = tmp[colors[i]];
      tmp[colors[i]]++;
    }

    for (int i = 0; i < nrows; i++) {
      iperm[perm[i]] = i;
    }

    delete[] tmp;
    delete[] colors;

    *iperm_ = iperm;
    *perm_ = perm;
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
  static void create_csr_from_element_conn(int nrows, int ncols, int nelems,
                                           const Functor& element_nodes,
                                           bool include_diagonal,
                                           bool sort_columns, int** rowp_,
                                           int** cols_) {
    int* node_to_elem_ptr = nullptr;
    int* node_to_elem = nullptr;
    compute_node_to_element_ptr(ncols, nelems, element_nodes, &node_to_elem_ptr,
                                &node_to_elem);

    // Set up the CSR data structure
    int* rowp = new int[nrows + 1];
    int* counter = new int[ncols];

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

        const int* ptr;
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
    int* cols = new int[nnz_count];

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

        const int* ptr;
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
                                          const Functor& element_nodes,
                                          int** node_to_elem_ptr_,
                                          int** node_to_elem_) {
    // Create data to store node -> element connectivity
    int* node_to_elem_ptr = new int[nnodes + 1];
    for (int i = 0; i < nnodes + 1; i++) {
      node_to_elem_ptr[i] = 0;
    }

    for (int i = 0; i < nelems; i++) {
      const int* ptr;
      int nodes_per_element = element_nodes(i, &ptr);
      for (int j = 0; j < nodes_per_element; j++, ptr++) {
        int node = ptr[0];
        if (node >= 0 && node < nnodes) {
          node_to_elem_ptr[ptr[0] + 1]++;
        }
      }
    }

    for (int i = 0; i < nnodes; i++) {
      node_to_elem_ptr[i + 1] += node_to_elem_ptr[i];
    }

    // Set up the node to element data
    int* node_to_elem = new int[node_to_elem_ptr[nnodes]];
    for (int i = 0; i < nelems; i++) {
      const int* ptr;
      int nodes_per_element = element_nodes(i, &ptr);
      for (int j = 0; j < nodes_per_element; j++, ptr++) {
        int node = ptr[0];
        if (node >= 0 && node < nnodes) {
          node_to_elem[node_to_elem_ptr[node]] = i;
          node_to_elem_ptr[node]++;
        }
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
   * @brief Find a coloring of the elements using a direct element -> node
   * information
   *
   * @param nelems Number of elements
   * @param num_colors_ Output number of colors
   * @param nnodes_per_elem Number of nodes for each element
   * @param elem_nodes Nodes for each element
   * @param elem_by_color_ptr_ Pointer into the elem_by_color array
   * @param elem_by_color_ Elements listed by color
   */
  static void color_elements(const int nelems, const int nnodes_per_elem,
                             const int* elem_nodes, int* num_colors_,
                             int** elem_by_color_ptr_, int** elem_by_color_) {
    int max_node = 0;
    for (int i = 0; i < nelems * nnodes_per_elem; i++) {
      if (elem_nodes[i] > max_node) {
        max_node = elem_nodes[i];
      }
    }
    max_node++;

    auto element_nodes = [&](int elem, const int* ptr[]) {
      *ptr = &elem_nodes[elem * nnodes_per_elem];
      return nnodes_per_elem;
    };

    color_elements(max_node, nelems, element_nodes, num_colors_,
                   elem_by_color_ptr_, elem_by_color_);
  }

  /**
   * @brief Find a coloring of the elements that is good for parallel
   * computation
   *
   * @tparam Functor Class type for the functor
   * @param nnodes Number of nodes
   * @param nelems Number of elements
   * @param element_nodes Functor returning the number of nodes and node numbers
   * @param num_colors_ Output number of colors
   * @param elem_by_color_ptr_ Pointer into the elem_by_color array
   * @param elem_by_color_ Elements listed by color
   */
  template <class Functor>
  static void color_elements(const int nnodes, const int nelems,
                             const Functor& element_nodes, int* num_colors_,
                             int** elem_by_color_ptr_, int** elem_by_color_) {
    int *elem_to_elem_ptr, *elem_to_elem;
    build_element_to_element(nnodes, nelems, element_nodes, &elem_to_elem_ptr,
                             &elem_to_elem);

    // Greedy coloring
    int* elem_colors = new int[nelems];
    std::fill(elem_colors, elem_colors + nelems, -1);

    // Keep track of the number of colors
    int num_colors = 0;

    int* flags = new int[nelems];
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
    int* elem_by_color = new int[nelems];
    int* elem_by_color_ptr = new int[num_colors + 1];
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
   * @tparam Functor Class type for the functor
   * @param nnodes Number of nodes
   * @param nelems Number of elements
   * @param element_nodes Functor returning the number of nodes and node numbers
   * @param elem_to_elem_ptr_ Output pointer into the elem_to_elem array
   * @param elem_to_elem_ Output element to element data
   */
  template <class Functor>
  static void build_element_to_element(const int nnodes, const int nelems,
                                       const Functor& element_nodes,
                                       int** elem_to_elem_ptr_,
                                       int** elem_to_elem_) {
    // Create a pointer from the nodes back to the elements
    int *node_to_elem_ptr, *node_to_elem;
    compute_node_to_element_ptr(nnodes, nelems, element_nodes,
                                &node_to_elem_ptr, &node_to_elem);

    // Compute the element -> element data structure
    int* elem_flags = new int[nelems];
    std::fill(elem_flags, elem_flags + nelems, -1);

    int* elem_to_elem_ptr = new int[nelems + 1];
    elem_to_elem_ptr[0] = 0;
    for (int i = 0; i < nelems; i++) {
      int count = 0;

      const int* ptr;
      int nnodes_per_elem = element_nodes(i, &ptr);
      for (int j = 0; j < nnodes_per_elem; j++, ptr++) {
        int node = ptr[0];

        if (node >= 0 && node < nnodes) {
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
      }
      elem_to_elem_ptr[i + 1] = count;
    }

    for (int i = 0; i < nelems; i++) {
      elem_to_elem_ptr[i + 1] += elem_to_elem_ptr[i];
    }

    std::fill(elem_flags, elem_flags + nelems, -1);
    int* elem_to_elem = new int[elem_to_elem_ptr[nelems]];
    for (int i = 0; i < nelems; i++) {
      int count = 0;

      const int* ptr;
      int nnodes_per_elem = element_nodes(i, &ptr);
      for (int j = 0; j < nnodes_per_elem; j++, ptr++) {
        int node = ptr[0];

        if (node >= 0 && node < nnodes) {
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
    }

    delete[] node_to_elem;
    delete[] node_to_elem_ptr;
    delete[] elem_flags;

    *elem_to_elem_ptr_ = elem_to_elem_ptr;
    *elem_to_elem_ = elem_to_elem;
  }

  /**
   * Sort the rows of the CSR matrix and make sure that the rows are unique
   *
   * The operation is performed in place
   *
   * @param nrows Number of rows in the matrix
   * @param rowp Pointer into the rows (modified during the call)
   * @param cols Column indices for each row (sorted on exit)
   */
  static void sort_and_uniquify_csr(int nrows, int rowp[], int cols[]) {
    // Sort and uniquify the csr structure
    int start = rowp[0];
    for (int i = 0; i < nrows; i++) {
      int size = rowp[i + 1] - start;
      int* array = &cols[start];
      std::sort(array, array + size);

      // Uniquify an array with duplicates
      int new_size = 0;
      if (size > 0) {
        // Copy the first entry
        cols[rowp[i]] = array[0];
        new_size++;

        for (int read_idx = 1; read_idx < size; read_idx++) {
          if (array[read_idx] != cols[rowp[i] + new_size - 1]) {
            cols[rowp[i] + new_size] = array[read_idx];
            new_size++;
          }
        }
      }

      start = rowp[i + 1];
      rowp[i + 1] = rowp[i] + new_size;
    }
  }

  /**
   * Create a CSR matrix structure from the input/output relationships
   *
   * @param nrows Number of outputs
   * @param ncols Number of inputs
   * @param nelems Number of elements
   * @param elements Function returning inputs/outputs for each element
   * @param rowp_ Output pointer into the column indices
   * @param cols_ Output columns for each row
   */
  template <class Functor>
  static void create_csr_from_output_data(int nrows, int ncols, int nelems,
                                          const Functor& elements, int** rowp_,
                                          int** cols_) {
    int* rowp = new int[nrows + 1];
    std::fill(rowp, rowp + (nrows + 1), 0);
    for (int elem = 0; elem < nelems; elem++) {
      int nout, nin;
      const int *outputs, *inputs;
      elements(elem, &nout, &nin, &outputs, &inputs);

      for (int i = 0; i < nout; i++) {
        rowp[outputs[i] + 1] += nin;
      }
    }

    // Figure out the size of the structure
    for (int i = 0; i < nrows; i++) {
      rowp[i + 1] += rowp[i];
    }

    int* cols = new int[rowp[nrows]];
    for (int elem = 0; elem < nelems; elem++) {
      int nout, nin;
      const int *outputs, *inputs;
      elements(elem, &nout, &nin, &outputs, &inputs);

      for (int i = 0; i < nout; i++) {
        int pos = rowp[outputs[i]];
        for (int j = 0; j < nin; j++) {
          cols[pos + j] = inputs[j];
        }
        rowp[outputs[i]] += nin;
      }
    }

    for (int i = nrows; i > 0; i--) {
      rowp[i] = rowp[i - 1];
    }
    rowp[0] = 0;

    sort_and_uniquify_csr(nrows, rowp, cols);

    *rowp_ = rowp;
    *cols_ = cols;
  }

  /**
   * Add the constraint CSR pattern to the existing non-zero pattern for the
   * matrix
   *
   * @tparam Functor Class type for the functor
   * @param nrows Number of rows in the non-zero pattern
   * @param ncols Number of columns in the non-zero pattern
   * @param rowp Input pointer into the rows for this matrix
   * @param cols Input column indices
   * @param num_comps Number of components that contribute constraints
   * @param get_component_csr Get the component compressed sparse matrix
   * @param _new_rowp Output pointer into the columns array
   * @param _new_cols Array of column indices
   */
  template <class Functor>
  static void add_constraint_csr_pattern(int nrows, int ncols, const int rowp[],
                                         const int cols[], int num_comps,
                                         const Functor& get_component_csr,
                                         int** _new_rowp, int** _new_cols) {
    int* new_rowp = new int[nrows + 1];

    new_rowp[nrows] = 0;
    for (int i = 0; i < nrows; i++) {
      new_rowp[i] = rowp[i + 1] - rowp[i];
    }

    for (int comp = 0; comp < num_comps; comp++) {
      int local_nrows, local_ncols;
      const int *local_rows, *local_columns;
      const int *local_rowp, *local_cols;
      get_component_csr(comp, &local_nrows, &local_ncols, &local_rows,
                        &local_columns, &local_rowp, &local_cols);

      // Add the row
      for (int i = 0; i < local_nrows; i++) {
        int row = local_rows[i];
        new_rowp[row] += local_rowp[i + 1] - local_rowp[i];
      }

      // Add the result from the transpose
      for (int i = 0; i < local_nrows; i++) {
        for (int jp = local_rowp[i]; jp < local_rowp[i + 1]; jp++) {
          int j = local_cols[jp];
          int col = local_columns[j];
          new_rowp[col]++;
        }
      }
    }

    // Now, count up enough space for everything
    int count = 0;
    for (int i = 0; i < nrows; i++) {
      int temp = new_rowp[i];
      new_rowp[i] = count;
      count += temp;
    }
    new_rowp[nrows] = count;

    // Allocate enough space to store everything
    int* new_cols = new int[count];

    // Copy over the original non-zero pattern
    for (int i = 0; i < nrows; i++) {
      for (int jp = rowp[i]; jp < rowp[i + 1]; jp++) {
        new_cols[new_rowp[i]] = cols[jp];
        new_rowp[i]++;
      }
    }

    // Add the non-zeros from the new pattern
    for (int comp = 0; comp < num_comps; comp++) {
      int local_nrows, local_ncols;
      const int *local_rows, *local_columns;
      const int *local_rowp, *local_cols;
      get_component_csr(comp, &local_nrows, &local_ncols, &local_rows,
                        &local_columns, &local_rowp, &local_cols);

      // Add the new rows
      for (int i = 0; i < local_nrows; i++) {
        int row = local_rows[i];
        for (int jp = local_rowp[i]; jp < local_rowp[i + 1]; jp++) {
          int j = local_cols[jp];
          int col = local_columns[j];

          // Add the contribution from the Jacobian
          new_cols[new_rowp[row]] = col;
          new_rowp[row]++;

          // Add the contribution from the transpose Jacobian
          new_cols[new_rowp[col]] = row;
          new_rowp[col]++;
        }
      }
    }

    // Reset the new_rowp pointer
    for (int i = nrows; i > 0; i--) {
      new_rowp[i] = new_rowp[i - 1];
    }
    new_rowp[0] = 0;

    sort_and_uniquify_csr(nrows, new_rowp, new_cols);

    *_new_rowp = new_rowp;
    *_new_cols = new_cols;
  }

  /**
   * @brief Compute a reordering of the variables that is consistent with the
   * partitioning of the mesh
   *
   * @tparam Functor Class type for the functor
   * @param nnodes Number of nodes
   * @param nelems Number of elements
   * @param element_nodes Functor returning the number of nodes and node numbers
   * @param part_size Partition size
   * @param partition Array of paritition assignments
   * @param node_ranges Array of node ranges associated with the assignment
   * @param new_node_numbers New numbering scheme established by greedy ordering
   */
  template <class Functor>
  static void reorder_for_partition(const int nnodes, const int nelems,
                                    const Functor& element_nodes, int part_size,
                                    const int partition[],
                                    int new_node_numbers[], int node_ranges[]) {
    std::fill(new_node_numbers, new_node_numbers + nnodes, -1);
    std::fill(node_ranges, node_ranges + (part_size + 1), 0);

    // Count up the number of times each node will be assigned
    for (int i = 0; i < nelems; i++) {
      const int* ptr;
      int nnodes_per_elem = element_nodes(i, &ptr);
      for (int j = 0; j < nnodes_per_elem; j++, ptr++) {
        int node = ptr[0];
        if (new_node_numbers[node] == -1) {
          int part = partition[i];
          new_node_numbers[node] = 1;
          node_ranges[part + 1]++;
        }
      }
    }

    // Adjust the node ranges and reset the node numbers
    for (int i = 0; i < part_size; i++) {
      node_ranges[i + 1] += node_ranges[i];
    }
    std::fill(new_node_numbers, new_node_numbers + nnodes, -1);

    // Count up the number of times each node will be assigned
    for (int i = 0; i < nelems; i++) {
      const int* ptr;
      int nnodes_per_elem = element_nodes(i, &ptr);
      for (int j = 0; j < nnodes_per_elem; j++, ptr++) {
        int node = ptr[0];
        if (new_node_numbers[node] == -1) {
          int part = partition[i];
          new_node_numbers[node] = node_ranges[part];
          node_ranges[part]++;
        }
      }
    }

    // Reset the node range values
    for (int i = part_size - 1; i >= 0; i--) {
      node_ranges[i + 1] = node_ranges[i];
    }
    node_ranges[0] = 0;
  }

  /**
   * @brief Distribute the elements to the processors from the root
   *
   * @param comm MPI communicator
   * @param nelems Number of elements
   * @param nnodes_per_elem Number of nodes per element
   * @param elem_nodes Nodes in the original ordering
   * @param partition The element assignment to each processor
   * @param new_node_numbers Global assignment of the new nodes
   * @param nelems_local Number of local elements
   * @param elem_nodes_local Element nodes on the local processor
   * @param root Root processor rank (typically = 0)
   */
  static void distribute_elements(MPI_Comm comm, const int nelems,
                                  const int nnodes_per_elem,
                                  const int elem_nodes[], const int partition[],
                                  const int new_node_numbers[],
                                  int* nelems_local_, int* elem_nodes_local_[],
                                  const int root = 0) {
    int size, rank;
    MPI_Comm_size(comm, &size);
    MPI_Comm_rank(comm, &rank);

    int *count = nullptr, *disp = nullptr;
    if (rank == root) {
      count = new int[size];
      disp = new int[size + 1];

      std::fill(count, count + size, 0);

      // Count up which elements are going where...
      for (int i = 0; i < nelems; i++) {
        int part = partition[i];
        count[part]++;
      }

      // Count up the displacements for later use
      disp[0] = 0;
      for (int i = 0; i < size; i++) {
        disp[i + 1] = disp[i] + count[i];
      }
    }

    // The number of local elements
    int nelems_local = 0;

    // Send the element counts
    MPI_Scatter(count, 1, MPI_INT, &nelems_local, 1, MPI_INT, root, comm);

    int* elem_nodes_root = nullptr;
    int* elem_nodes_local = new int[nelems_local * nnodes_per_elem];

    if (rank == root) {
      elem_nodes_root = new int[nelems * nnodes_per_elem];

      for (int i = 0; i < nelems; i++) {
        int part = partition[i];
        int elem = disp[part];

        for (int j = 0; j < nnodes_per_elem; j++) {
          elem_nodes_root[nnodes_per_elem * elem + j] =
              new_node_numbers[elem_nodes[nnodes_per_elem * i + j]];
        }
        disp[part]++;
      }

      // Reset the displacements and adjust the counts
      disp[0] = 0;
      for (int i = 0; i < size; i++) {
        count[i] = nnodes_per_elem * count[i];
        disp[i + 1] = disp[i] + count[i];
      }
    }

    MPI_Scatterv(elem_nodes_root, count, disp, MPI_INT, elem_nodes_local,
                 nelems_local * nnodes_per_elem, MPI_INT, root, comm);

    if (rank == root) {
      delete[] count;
      delete[] disp;
      delete[] elem_nodes_root;
    }

    *nelems_local_ = nelems_local;
    *elem_nodes_local_ = elem_nodes_local;
  }

  /**
   * @brief Match the intervals for passing the sorted halo nodes to external
   * values
   *
   * @param size Size of the partition
   * @param range Ownership ranges for each processor
   * @param nnodes Number of nodes
   * @param nodes Sorted node numbers
   * @param ptr Output pointer length (size + 1)
   */
  static void match_intervals(int size, const int range[], int nnodes,
                              const int nodes[], int ptr[]) {
    if (nnodes == 0) {
      for (int i = 0; i < size + 1; i++) {
        ptr[i] = 0;
      }
      return;
    }

    for (int i = 0; i < size + 1; i++) {
      if (range[i] <= nodes[0]) {
        ptr[i] = 0;
      } else if (range[i] > nodes[nnodes - 1]) {
        ptr[i] = nnodes;
      } else {
        // Binary search for the interval
        int low = 0;
        int high = nnodes - 1;
        int mid = low + ((high - low) / 2);

        // Maintain that the variable is in the interval
        // (vars[low],vars[high]) note that if high-low=1, then mid = high
        while (high != mid) {
          if (nodes[mid] == range[i]) {
            break;
          }

          if (range[i] < nodes[mid]) {
            high = mid;
          } else {
            low = mid;
          }

          mid = high - ((high - low) / 2);
        }

        ptr[i] = mid;
      }
    }
  }
};

}  // namespace amigo

#endif  // AMIGO_ORDERING_UTILS_H