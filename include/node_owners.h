#ifndef AMIGO_NODE_OWNERS_H
#define AMIGO_NODE_OWNERS_H

#include <mpi.h>

namespace amigo {

/**
 * @brief Store the node to processor assignments and the implicit mapping from
 * the local indices to the global indices
 *
 * The global node numbers on each processor are mapped to local indices through
 * the ext_nodes array. This is an array of sorted global node numbers. The
 * local nodes on a processor are 0 <= node <= num_owned_nodes + num_ext_nodes,
 * where
 *
 * num_owned_nodes = range[mpi_rank + 1] - range[mpi_rank],
 *
 * and num_ext_nodes is passed as an argument to NodeOwners.
 *
 * The mapping from the local node numbers corresponding global node numbers is:
 *
 * if 0 <= local_node <= num_owned_nodes
 *    global_node = range[mpi_rank] + local_node
 * else:
 *    global_node = ext_nodes[local_node - num_owned_nodes]
 *
 */
class NodeOwners {
 public:
  NodeOwners(MPI_Comm comm, const int range_[], int num_ext_nodes = 0,
             const int ext_nodes_[] = nullptr)
      : comm(comm), num_ext_nodes(num_ext_nodes) {
    int size, rank;
    MPI_Comm_size(comm, &size);
    MPI_Comm_rank(comm, &rank);

    range = new int[size + 1];
    for (int i = 0; i < size + 1; i++) {
      range[i] = range_[i];
    }
    ext_nodes = new int[num_ext_nodes];
    for (int i = 0; i < num_ext_nodes; i++) {
      ext_nodes[i] = ext_nodes_[i];
    }
    num_local_nodes = range[rank + 1] - range[rank];
  }
  ~NodeOwners() {
    delete[] range;
    if (ext_nodes) {
      delete[] ext_nodes;
    }
  }

  /**
   * @brief Get the MPI communicator object
   */
  MPI_Comm get_mpi_comm() const { return comm; }

  /**
   * Get the range of nodes owned by each processor
   */
  const int* get_range() const { return range; }

  /**
   * @brief Get the number of local variables
   */
  int get_local_size() const { return num_local_nodes; }

  /**
   * @brief Get the number of local
   */
  int get_ext_size() const { return num_ext_nodes; }

  /**
   * @brief Get the external nodes
   */
  int get_ext_nodes(const int* nodes[]) const {
    if (nodes) {
      *nodes = ext_nodes;
    }
    return num_ext_nodes;
  }

  /**
   * @brief Compute the global node numbers based on the local number
   *
   * @param nnodes Number of input nodes
   * @param nodes Local node numbers
   * @param global Outupt global node numbers
   */
  void local_to_global(int nnodes, const int nodes[], int global[]) const {
    int rank;
    MPI_Comm_rank(comm, &rank);
    for (int i = 0; i < nnodes; i++) {
      if (nodes[i] < num_local_nodes) {
        global[i] = range[rank] + nodes[i];
      } else {
        global[i] = ext_nodes[nodes[i] - num_local_nodes];
      }
    }
  }

 private:
  MPI_Comm comm;
  int* range;
  int num_local_nodes;

  // Store mapping of the local node numbers
  int num_ext_nodes;
  int* ext_nodes;
};

}  // namespace amigo

#endif  // AMIGO_NODE_OWNERS_H