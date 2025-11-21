#ifndef AMIGO_VECTOR_DISTRIBUTE
#define AMIGO_VECTOR_DISTRIBUTE

#include <mpi.h>

#include <memory>

#include "amigo.h"
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

namespace detail {

template <typename T>
void set_buffer_values_kernel_cuda(int nnodes, const int* nodes, const T* array,
                                   T* buffer);

template <typename T>
void add_buffer_values_kernel_cuda(int nnodes, const int* nodes,
                                   const T* buffer, T* array);

}  // namespace detail

template <ExecPolicy policy>
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
    VecDistributeContext(int nsends, int nrecvs,
                         std::shared_ptr<Vector<int>> send_indices,
                         int num_recv_nodes, int num_owned_nodes, int tag = 0)
        : send_indices(send_indices),
          num_recv_nodes(num_recv_nodes),
          num_owned_nodes(num_owned_nodes),
          tag(tag),
          send_buffer(nullptr),
          recv_buffer(nullptr),
          d_send_buffer(nullptr) {
      int nnodes = send_indices->get_size();
      send_buffer = new T[nnodes];
      recv_buffer = nullptr;
      d_send_buffer = nullptr;

      if (policy == ExecPolicy::CUDA) {
#ifdef AMIGO_USE_CUDA
        recv_buffer = new T[num_recv_nodes];
        AMIGO_CHECK_CUDA(cudaMalloc(&d_send_buffer, nnodes * sizeof(T)));
#endif  // AMIGO_USE_CUDA
      }

      send_requests = new MPI_Request[nsends];
      recv_requests = new MPI_Request[nrecvs];
    }
    ~VecDistributeContext() {
      delete[] send_buffer;
      delete[] send_requests;
      delete[] recv_requests;
      if (recv_buffer) {
        delete[] recv_buffer;
      }
      if (d_send_buffer) {
#ifdef AMIGO_USE_CUDA
        cudaFree(d_send_buffer);
#endif  // AMIGO_USE_CUDA
      }
    }

    void set_buffer_values_kernel(int nnodes, const int* node, const T* array,
                                  T* buffer) {
      const int* node_end = node + nnodes;
      for (; node < node_end; node++, buffer++) {
        buffer[0] = array[*node];
      }
    }

    void add_buffer_values_kernel(int nnodes, const int* node, const T* buffer,
                                  T* array) const {
      const int* node_end = node + nnodes;
      for (; node < node_end; node++, buffer++) {
        array[*node] += buffer[0];
      }
    }

    T* forward_set_send_buffer(T* array) {
      int nnodes = send_indices->get_size();
      const int* nodes = send_indices->template get_array<policy>();

      if constexpr (policy == ExecPolicy::SERIAL ||
                    policy == ExecPolicy::OPENMP) {
        set_buffer_values_kernel(nnodes, nodes, array, send_buffer);
      } else {
#ifdef AMIGO_USE_CUDA
        detail::set_buffer_values_kernel_cuda(nnodes, nodes, array,
                                              d_send_buffer);
        AMIGO_CHECK_CUDA(cudaMemcpy(send_buffer, d_send_buffer,
                                    nnodes * sizeof(T),
                                    cudaMemcpyDeviceToHost));
#endif  // AMIGO_USE_CUDA
      }
      return send_buffer;
    }

    T* forward_get_recv_buffer(T* array) {
      if constexpr (policy == ExecPolicy::SERIAL ||
                    policy == ExecPolicy::OPENMP) {
        return &array[num_owned_nodes];
      }
      return recv_buffer;
    }

    void forward_set_recv_values(T* array) {
      if constexpr (policy == ExecPolicy::CUDA) {
#ifdef AMIGO_USE_CUDA
        AMIGO_CHECK_CUDA(cudaMemcpy(&array + num_owned_nodes, recv_buffer,
                                    num_recv_nodes * sizeof(T),
                                    cudaMemcpyHostToDevice));
#endif  // AMIGO_USE_CUDA
      }
    }

    T* reverse_get_send_buffer() { return send_buffer; }

    T* reverse_get_recv_buffer(T* array) {
      if constexpr (policy == ExecPolicy::SERIAL ||
                    policy == ExecPolicy::OPENMP) {
        return &array[num_owned_nodes];
      }
      return recv_buffer;
    }

    void reverse_add_buffer_values(T* array) {
      int nnodes = send_indices->get_size();
      const int* nodes = send_indices->template get_array<policy>();

      if constexpr (policy == ExecPolicy::SERIAL ||
                    policy == ExecPolicy::OPENMP) {
        add_buffer_values_kernel(nnodes, nodes, send_buffer, array);
      } else {
#ifdef AMIGO_USE_CUDA
        AMIGO_CHECK_CUDA(cudaMemcpy(d_send_buffer, send_buffer,
                                    nnodes * sizeof(T),
                                    cudaMemcpyHostToDevice));
        detail::add_buffer_values_kernel_cuda(nnodes, nodes, d_send_buffer,
                                              array);
#endif  // AMIGO_USE_CUDA
      }
    }

    std::shared_ptr<Vector<int>> send_indices;
    int num_recv_nodes;
    int num_owned_nodes;

    T* send_buffer;
    T* recv_buffer;
    T* d_send_buffer;
    int tag;

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
    send_indices = std::make_shared<Vector<int>>(send_ptr[num_sends]);

    // Get the host array for the indices
    int* send_indices_ptr = send_indices->get_array();

    MPI_Request* send_requests = new MPI_Request[num_sends];
    MPI_Request* recv_requests = new MPI_Request[num_recvs];

    // Reverse communication from the destination to the source
    int tag = 0;
    for (int i = 0; i < num_recvs; i++) {
      MPI_Isend(&ext_nodes[recv_ptr[i]], recv_count[i], MPI_INT, recv_procs[i],
                tag, comm, &recv_requests[i]);
    }
    for (int i = 0; i < num_sends; i++) {
      MPI_Irecv(&send_indices_ptr[send_ptr[i]], send_count[i], MPI_INT,
                send_procs[i], tag, comm, &send_requests[i]);
    }

    MPI_Waitall(num_sends, send_requests, MPI_STATUSES_IGNORE);
    MPI_Waitall(num_recvs, recv_requests, MPI_STATUSES_IGNORE);

    // Convert to the local ordering
    for (int i = 0; i < send_ptr[num_sends]; i++) {
      send_indices_ptr[i] -= range[mpi_rank];
    }

    // Copy the send indices to the device
    if constexpr (policy == ExecPolicy::CUDA) {
      send_indices->copy_host_to_device();
    }

    delete[] send_requests;
    delete[] recv_requests;
  }
  ~VectorDistribute() {
    delete[] send_count;
    delete[] send_ptr;
    delete[] send_procs;
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
    return new VecDistributeContext<T>(num_sends, num_recvs, send_indices,
                                       recv_ptr[num_recvs], num_owned_nodes);
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
    // Get the array pointer based on the policy - this is either a host or
    // device pointer, depending on the policy value
    T* array = vars->template get_array<policy>();

    // Set values into the buffer. If this is on the CPU, this is a straight
    // copy, if this is on the device, copy the values to a device buffer first,
    // then copy those values to the host and return the host buffer.
    T* send_buffer = ctx->forward_set_send_buffer(array);

    // Post the sends
    for (int i = 0; i < num_sends; i++) {
      T* ptr = send_buffer + send_ptr[i];
      MPI_Isend(ptr, send_count[i], get_mpi_type<T>(), send_procs[i], ctx->tag,
                comm, &ctx->send_requests[i]);
    }

    // Get the receive buffer
    T* recv_buffer = ctx->forward_get_recv_buffer(array);

    // Post the receives
    for (int i = 0; i < num_recvs; i++) {
      T* ptr = recv_buffer + recv_ptr[i];
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
    // Wait for everything to complete
    MPI_Waitall(num_sends, ctx->send_requests, MPI_STATUSES_IGNORE);
    MPI_Waitall(num_recvs, ctx->recv_requests, MPI_STATUSES_IGNORE);

    // Get the array pointer based on the policy - this is either a host or
    // device pointer, depending on the policy value
    T* array = vars->template get_array<policy>();

    // Set the values from the recv buffer into the array. This is only used
    // when policy == CUDA
    ctx->forward_set_recv_values(array);
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
    // Get the array that depends on the policy
    T* array = vars->template get_array<policy>();

    // Post the sends
    T* send_buffer = ctx->reverse_get_send_buffer();
    for (int i = 0; i < num_sends; i++) {
      T* ptr = send_buffer + send_ptr[i];
      MPI_Irecv(ptr, send_count[i], get_mpi_type<T>(), send_procs[i], ctx->tag,
                comm, &ctx->send_requests[i]);
    }

    // Post the recvs
    T* recv_buffer = ctx->reverse_get_recv_buffer(array);
    for (int i = 0; i < num_recvs; i++) {
      T* ptr = recv_buffer + recv_ptr[i];
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
    T* array = vars->template get_array<policy>();
    ctx->reverse_add_buffer_values(array);
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

  // Data that will be recieved by this processor
  int num_recvs;
  int* recv_count;
  int* recv_ptr;
  int* recv_procs;

  // Store the send_indices as a vector
  std::shared_ptr<Vector<int>> send_indices;
};

}  // namespace amigo

#endif  // AMIGO_VECTOR_DISTRIBUTE