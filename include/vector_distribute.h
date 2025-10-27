#ifndef AMIGO_VECTOR_DISTRIBUTE
#define AMIGO_VECTOR_DISTRIBUTE

#include <mpi.h>

#include "node_owners.h"
#include "ordering_utils.h"
#include "vector.h"

namespace amigo {

template <typename T>
constexpr MPI_Datatype get_mpi_type() {
  if constexpr (std::is_same<T, int>::value) {
    return MPI_INT;
  } else if constexpr (std::is_same<T, float>::value) {
    return MPI_FLOAT;
  } else if constexpr (std::is_same<T, double>::value) {
    return MPI_DOUBLE;
  } else if constexpr (std::is_same<T, std::complex<float>>::value) {
    return MPI_C_COMPLEX;
  } else if constexpr (std::is_same<T, std::complex<double>>::value) {
    return MPI_C_DOUBLE_COMPLEX;
  } else {
    return MPI_DATATYPE_NULL;
  }
}

class VectorDistribute {
 public:
  /**
   * @brief DistributeContext that stores the data
   *
   * @tparam T Data type
   */
  template <typename T>
  class VecDistributeContext {
   public:
    VecDistributeContext(int nnodes, int nsends, int nrecvs, int tag = 0)
        : tag(tag) {
      buffer = new T[nnodes];
      send_requests = new MPI_Request[nsends];
      recv_requests = new MPI_Request[nrecvs];
    }
    ~VecDistributeContext() {
      delete[] buffer;
      delete[] send_requests;
      delete[] recv_requests;
    }

    void set_buffer_values(int nnodes, const int* nodes, const T* array) {
      T* b = buffer;
      const int* n = nodes;
      for (int i = 0; i < nnodes; i++, b++, n++) {
        b[0] = array[*n];
      }
    }

    void add_buffer_values(int nnodes, const int* nodes, T* array) const {
      const T* b = buffer;
      const int* n = nodes;
      for (int i = 0; i < nnodes; i++, b++, n++) {
        array[*n] += b[0];
      }
    }

    int tag;
    T* buffer;
    MPI_Request* send_requests;
    MPI_Request* recv_requests;
  };

  /**
   * @brief Construct a Distribute object that transmits node variables
   * from their owners to the halo values on other processors.
   *
   * This object works in conjunction with a context object that stores the
   * buffered information and MPI data needed to transmit the data. The same
   * communication can therefore be used by multiple vectors at the same time.
   *
   * @param comm MPI communicator
   * @param owners Ownership range for each processor
   */
  VectorDistribute(std::shared_ptr<NodeOwners> owners)
      : comm(owners->get_mpi_comm()), owners(owners) {
    // Get the range of variables that are owned by each processor
    const int* range = owners->get_range();

    // Get the external node numbers
    const int* ext_nodes;
    int num_ext_nodes = owners->get_ext_nodes(&ext_nodes);

    // Get the processor rank and size
    int mpi_size, mpi_rank;
    MPI_Comm_rank(comm, &mpi_rank);
    MPI_Comm_size(comm, &mpi_size);
    int* full_recv_ptr = new int[mpi_size + 1];

    // Set the number of owned nodes
    num_owned_nodes = range[mpi_rank + 1] - range[mpi_rank];

    // Figure out where stuff should go by matching the intervals
    OrderingUtils::match_intervals(mpi_size, range, num_ext_nodes, ext_nodes,
                                   full_recv_ptr);

    int* full_recv_count = new int[mpi_size];
    for (int i = 0; i < mpi_size; i++) {
      full_recv_count[i] = full_recv_ptr[i + 1] - full_recv_ptr[i];
    }
    delete[] full_recv_ptr;

    int* full_send_count = new int[mpi_size];

    // Do one Alltoall
    MPI_Alltoall(full_recv_count, 1, MPI_INT, full_send_count, 1, MPI_INT,
                 comm);

    // Many processors will likely have no variables associated with
    // them. Create a data structure so that we can skip these
    // processors and avoid wasting CPU time.
    num_sends = 0;
    num_recvs = 0;
    for (int i = 0; i < mpi_size; i++) {
      if (i != mpi_rank && full_send_count[i] > 0) {
        num_sends++;
      }
      if (i != mpi_rank && full_recv_count[i] > 0) {
        num_recvs++;
      }
    }

    // Number of external processors from which we are getting external
    // variable values
    send_procs = new int[num_sends];
    send_ptr = new int[num_sends + 1];
    send_count = new int[num_sends];

    // Number of processors to which we are sending data
    recv_procs = new int[num_recvs];
    recv_ptr = new int[num_recvs + 1];
    recv_count = new int[num_recvs];

    // Set pointers to the external/requests processor ranks
    send_ptr[0] = 0;
    recv_ptr[0] = 0;
    for (int i = 0, sends = 0, recvs = 0; i < mpi_size; i++) {
      if (i != mpi_rank && full_send_count[i] > 0) {
        send_procs[sends] = i;
        send_count[sends] = full_send_count[i];
        send_ptr[sends + 1] = send_ptr[sends] + send_count[sends];
        sends++;
      }
      if (i != mpi_rank && full_recv_count[i] > 0) {
        recv_procs[recvs] = i;
        recv_count[recvs] = full_recv_count[i];
        recv_ptr[recvs + 1] = recv_ptr[recvs] + recv_count[recvs];
        recvs++;
      }
    }

    delete[] full_send_count;
    delete[] full_recv_count;

    // Indices of the data that we will send to the processors
    send_indices = new int[send_ptr[num_sends]];

    MPI_Request* send_requests = new MPI_Request[num_sends];
    MPI_Request* recv_requests = new MPI_Request[num_recvs];

    // Reverse communication from the destination to the source
    int tag = 0;
    for (int i = 0; i < num_recvs; i++) {
      MPI_Isend(&ext_nodes[recv_ptr[i]], recv_count[i], MPI_INT, recv_procs[i],
                tag, comm, &recv_requests[i]);
    }
    for (int i = 0; i < num_sends; i++) {
      MPI_Irecv(&send_indices[send_ptr[i]], send_count[i], MPI_INT,
                send_procs[i], tag, comm, &send_requests[i]);
    }

    MPI_Waitall(num_sends, send_requests, MPI_STATUSES_IGNORE);
    MPI_Waitall(num_recvs, recv_requests, MPI_STATUSES_IGNORE);

    // Convert to the local ordering
    for (int i = 0; i < send_ptr[num_sends]; i++) {
      send_indices[i] -= range[mpi_rank];
    }

    delete[] send_requests;
    delete[] recv_requests;
  }
  ~VectorDistribute() {
    delete[] send_count;
    delete[] send_ptr;
    delete[] send_procs;
    delete[] send_indices;
    delete[] recv_count;
    delete[] recv_ptr;
    delete[] recv_procs;
  }

  /**
   * @brief Get the range of variables that belong to this processor
   *
   * @return const NodeOwners*
   */
  const std::shared_ptr<NodeOwners> get_node_owners() const { return owners; }

  /**
   * @brief Create a context object that stores the data for transferring a
   * vector
   *
   * @tparam T Data type
   * @return DistributeContext<T>* New context object
   */
  template <typename T>
  VecDistributeContext<T>* create_context() {
    return new VecDistributeContext<T>(send_ptr[num_sends], num_sends,
                                       num_recvs);
  }

  /**
   * @brief Begin the transfer of data from the owners to the halo nodes
   *
   * @tparam T Data type
   * @param vars Variables to transfer
   * @param ctx Context/buffer where the data is stored temporarily
   */
  template <typename T>
  void begin_forward(std::shared_ptr<Vector<T>> vars,
                     VecDistributeContext<T>* ctx) {
    T* array = vars->get_array();

    // Copy the data to the buffer
    ctx->set_buffer_values(send_ptr[num_sends], send_indices, array);

    // Post the sends
    for (int i = 0; i < num_sends; i++) {
      T* ptr = ctx->buffer + send_ptr[i];
      MPI_Isend(ptr, send_count[i], get_mpi_type<T>(), send_procs[i], ctx->tag,
                comm, &ctx->send_requests[i]);
    }

    // Post the recvs
    for (int i = 0; i < num_recvs; i++) {
      T* ptr = array + (recv_ptr[i] + num_owned_nodes);
      MPI_Irecv(ptr, recv_count[i], get_mpi_type<T>(), recv_procs[i], ctx->tag,
                comm, &ctx->recv_requests[i]);
    }
  }

  /**
   * @brief Finalize the transfer of data from the owners to the halo nodes
   *
   * @tparam T Data type
   * @param vars Variables to transfer
   * @param ctx Context/buffer where the data is stored temporarily
   */
  template <typename T>
  void end_forward(std::shared_ptr<Vector<T>> vars,
                   VecDistributeContext<T>* ctx) {
    MPI_Waitall(num_sends, ctx->send_requests, MPI_STATUSES_IGNORE);
    MPI_Waitall(num_recvs, ctx->recv_requests, MPI_STATUSES_IGNORE);
  }

  /**
   * @brief Send data from the halo nodes back to the owners, adding the halo
   * values to the owner values
   *
   * @tparam T Data type
   * @param vars Variables to transfer
   * @param ctx Context/buffer where the data is stored temporarily
   */
  template <typename T>
  void begin_reverse_add(std::shared_ptr<Vector<T>> vars,
                         VecDistributeContext<T>* ctx) {
    T* array = vars->get_array();

    // Post the sends
    for (int i = 0; i < num_sends; i++) {
      T* ptr = ctx->buffer + send_ptr[i];
      MPI_Irecv(ptr, send_count[i], get_mpi_type<T>(), send_procs[i], ctx->tag,
                comm, &ctx->send_requests[i]);
    }

    // Post the recvs
    for (int i = 0; i < num_recvs; i++) {
      T* ptr = array + (recv_ptr[i] + num_owned_nodes);
      MPI_Isend(ptr, recv_count[i], get_mpi_type<T>(), recv_procs[i], ctx->tag,
                comm, &ctx->recv_requests[i]);
    }
  }

  /**
   * @brief Finalize sending the data from the halo nodes back to the owners,
   * adding the halo values to the owner values
   *
   * @tparam T Data type
   * @param vars Variables to transfer
   * @param ctx Context/buffer where the data is stored temporarily
   */
  template <typename T>
  void end_reverse_add(std::shared_ptr<Vector<T>> vars,
                       VecDistributeContext<T>* ctx) {
    MPI_Waitall(num_sends, ctx->send_requests, MPI_STATUSES_IGNORE);
    MPI_Waitall(num_recvs, ctx->recv_requests, MPI_STATUSES_IGNORE);

    // Add the data to the buffer
    T* array = vars->get_array();
    ctx->add_buffer_values(send_ptr[num_sends], send_indices, array);
  }

 private:
  MPI_Comm comm;
  const std::shared_ptr<NodeOwners> owners;
  int num_owned_nodes;

  // Data that will be sent to other processors
  int num_sends;
  int* send_count;
  int* send_ptr;
  int* send_procs;
  int* send_indices;  // Indices of the data from this proc

  // Data that will be recieved by this processor
  int num_recvs;
  int* recv_count;
  int* recv_ptr;
  int* recv_procs;
};

}  // namespace amigo

#endif  // AMIGO_VECTOR_DISTRIBUTE