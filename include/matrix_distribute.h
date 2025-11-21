#ifndef AMIGO_MATRIX_DISTRIBUTE_H
#define AMIGO_MATRIX_DISTRIBUTE_H

#include <mpi.h>

#include "amigo.h"
#include "csr_matrix.h"
#include "node_owners.h"
#include "ordering_utils.h"
#include "vector_distribute.h"

namespace amigo {

template <ExecPolicy policy>
class MatrixDistribute {
 public:
  template <typename T>
  class MatDistributeContext {
   public:
    MatDistributeContext(int num_send_procs, int num_recv_procs,
                         int num_send_entries,
                         std::shared_ptr<Vector<int>> recv_locs,
                         int num_owned_rows, int tag = 0)
        : num_send_entries(num_send_entries),
          num_recv_entries(recv_locs->get_size()),
          recv_locs(recv_locs),
          num_owned_rows(num_owned_rows),
          tag(tag) {
      send_A = nullptr;
      recv_A = new T[num_recv_entries];
      d_recv_A = nullptr;

      if (policy == ExecPolicy::CUDA) {
#ifdef AMIGO_USE_CUDA
        send_A = new T[num_send_entries];
        AMIGO_CHECK_CUDA(cudaMalloc(&d_recv_A, num_recv_entries * sizeof(T)));
#endif  // AMIGO_USE_CUDA
      }

      recv_requests = new MPI_Request[num_recv_procs];
      send_requests = new MPI_Request[num_send_procs];
    }
    ~MatDistributeContext() {
      delete[] recv_A;
      delete[] recv_requests;
      delete[] send_requests;

      if (send_A) {
        delete[] send_A;
      }
      if (d_recv_A) {
#ifdef AMIGO_USE_CUDA
        cudaFree(d_recv_A);
#endif  // AMIGO_USE_CUDA
      }
    }

    T* get_recv_buffer() { return recv_A; }

    T* get_send_entries(std::shared_ptr<CSRMat<T>> mat) {
      const int* rowp;
      T* data;
      mat->get_data(nullptr, nullptr, nullptr, &rowp, nullptr, &data);
      int offset = rowp[num_owned_rows];

      if constexpr (policy == ExecPolicy::SERIAL ||
                    policy == ExecPolicy::OPENMP) {
        return &data[offset];
      } else {
#ifdef AMIGO_USE_CUDA
        T* d_data;
        mat->get_device_data(nullptr, nullptr, &d_data);
        AMIGO_CHECK_CUDA(cudaMemcpy(send_A, d_data + offset,
                                    num_send_entries * sizeof(T),
                                    cudaMemcpyDeviceToHost));
#endif  // AMIGO_USE_CUDA
        return send_A;
      }
    }

    void add_recv_entries(std::shared_ptr<CSRMat<T>> mat) {
      if constexpr (policy == ExecPolicy::SERIAL ||
                    policy == ExecPolicy::OPENMP) {
        T* data;
        mat->get_data(nullptr, nullptr, nullptr, nullptr, nullptr, &data);
        const int* recv_loc_array = recv_locs->get_array();

        for (int i = 0; i < num_recv_entries; i++) {
          data[recv_loc_array[i]] += recv_A[i];
        }
      } else {
#ifdef AMIGO_USE_CUDA
        T* d_data;
        mat->get_device_data(nullptr, nullptr, &d_data);

        const int* recv_loc_array = recv_locs->get_device_array();
        AMIGO_CHECK_CUDA(cudaMemcpy(d_recv_A, recv_A,
                                    num_recv_entries * sizeof(T),
                                    cudaMemcpyHostToDevice));
        detail::add_buffer_values_kernel_cuda(num_recv_entries, recv_loc_array,
                                              d_recv_A, d_data);

#endif  // AMIGO_USE_CUDA
      }
    }

    int num_send_entries;
    int num_recv_entries;
    std::shared_ptr<Vector<int>> recv_locs;
    int num_owned_rows;

    T* send_A;    // Host buffer for sending entries
    T* recv_A;    // Storage for the incoming entries
    T* d_recv_A;  // Device buffer for receiving entries

    int tag;                     // MPI tag value
    MPI_Request* recv_requests;  // Requests for recving data
    MPI_Request* send_requests;  // Requests for sending info
  };

  /**
   * @brief Construct a new Mat Distribute object
   *
   * @param comm MPI communicator
   * @param mem_loc Memory location for the data entries
   * @param row_owners Row owners for the matrix
   * @param col_owners Column owners for the matrix
   * @param nrows Number of local rows
   * @param ncols Number of local columns
   * @param rowp Pointer into the column indices for the local contributions
   * @param cols Column indices for the local contributions
   * @param csr The output CSRMat matrix
   */
  template <typename T>
  MatrixDistribute(MPI_Comm comm, MemoryLocation mem_loc,
                   std::shared_ptr<NodeOwners> row_owners,
                   std::shared_ptr<NodeOwners> col_owners, int nrows, int ncols,
                   const int* rowp, const int* cols,
                   std::shared_ptr<CSRMat<T>>& csr)
      : comm(comm), row_owners(row_owners), col_owners(col_owners) {
    int mpi_size, mpi_rank;
    MPI_Comm_rank(comm, &mpi_rank);
    MPI_Comm_size(comm, &mpi_size);

    // Get how the rows are distributed between processors
    const int* row_ranges = row_owners->get_range();
    const int* ext_rows;
    int num_ext_rows = row_owners->get_ext_nodes(&ext_rows);

    // Set the number of local rows
    int num_local_rows = row_ranges[mpi_rank + 1] - row_ranges[mpi_rank];

    // Get the processor rank and size
    int* full_send_ptr = new int[mpi_size + 1];

    // Figure out where stuff should go by matching the intervals in the
    // sorted ext_rows array
    OrderingUtils::match_intervals(mpi_size, row_ranges, num_ext_rows, ext_rows,
                                   full_send_ptr);

    // Count up the number of entries
    int* full_send_count = new int[2 * mpi_size];
    for (int i = 0; i < mpi_size; i++) {
      // Set the number of rows that will be sent
      full_send_count[2 * i] = full_send_ptr[i + 1] - full_send_ptr[i];

      // Set the number of entries that will be sent
      int start = full_send_ptr[i] + num_local_rows;
      int end = full_send_ptr[i + 1] + num_local_rows;
      full_send_count[2 * i + 1] = rowp[end] - rowp[start];
    }
    delete[] full_send_ptr;

    // Allocate space to store the number of in or out rows
    int* full_recv_count = new int[2 * mpi_size];

    // Do one Alltoall
    MPI_Alltoall(full_send_count, 2, MPI_INT, full_recv_count, 2, MPI_INT,
                 comm);

    // Count up the number of sends/recvs
    num_send_procs = 0;
    num_recv_procs = 0;
    for (int i = 0; i < mpi_size; i++) {
      if (i != mpi_rank && full_recv_count[2 * i] > 0) {
        num_recv_procs++;
      }
      if (i != mpi_rank && full_send_count[2 * i] > 0) {
        num_send_procs++;
      }
    }

    // Number of external processors from which we are getting external
    // variable values
    recv_procs = new int[num_recv_procs];
    recv_entry_count = new int[num_recv_procs];
    int* recv_row_count = new int[num_recv_procs];

    // Number of processors to which we are sending data
    send_procs = new int[num_send_procs];
    send_entry_count = new int[num_send_procs];
    int* send_row_count = new int[num_send_procs];

    // Count up the rows to send first
    int num_recv_rows = 0;
    num_recv_entries = 0;
    num_send_entries = 0;
    for (int i = 0, send = 0, recv = 0; i < mpi_size; i++) {
      if (i != mpi_rank && full_send_count[2 * i] > 0) {
        send_procs[send] = i;
        send_row_count[send] = full_send_count[2 * i];
        send_entry_count[send] = full_send_count[2 * i + 1];
        num_send_entries += send_entry_count[send];
        send++;
      }
      if (i != mpi_rank && full_recv_count[2 * i] > 0) {
        recv_procs[recv] = i;
        recv_row_count[recv] = full_recv_count[2 * i];
        recv_entry_count[recv] = full_recv_count[2 * i + 1];
        num_recv_rows += recv_row_count[recv];
        num_recv_entries += recv_entry_count[recv];
        recv++;
      }
    }

    delete[] full_send_count;
    delete[] full_recv_count;

    // Indices of the data that we will send to the processors
    int* recv_rows = new int[num_recv_rows];

    MPI_Request* recv_requests = new MPI_Request[num_recv_procs];
    MPI_Request* send_requests = new MPI_Request[num_send_procs];

    // Transmit the rows from the source to the destination processors
    int tag = 0;
    for (int i = 0, offset = 0; i < num_send_procs; i++) {
      MPI_Isend(&ext_rows[offset], send_row_count[i], MPI_INT, send_procs[i],
                tag, comm, &send_requests[i]);
      offset += send_row_count[i];
    }
    for (int i = 0, offset = 0; i < num_recv_procs; i++) {
      MPI_Irecv(&recv_rows[offset], recv_row_count[i], MPI_INT, recv_procs[i],
                tag, comm, &recv_requests[i]);
      offset += recv_row_count[i];
    }

    MPI_Waitall(num_recv_procs, recv_requests, MPI_STATUSES_IGNORE);
    MPI_Waitall(num_send_procs, send_requests, MPI_STATUSES_IGNORE);

    // Convert to the local ordering
    for (int i = 0; i < num_recv_rows; i++) {
      recv_rows[i] -= row_ranges[mpi_rank];
    }

    // Now send the row sizes
    int* send_row_sizes = new int[num_ext_rows];
    for (int i = 0; i < num_ext_rows; i++) {
      send_row_sizes[i] =
          rowp[i + 1 + num_local_rows] - rowp[i + num_local_rows];
    }

    // Allocate space for the row sizes
    int* recv_row_ptr = new int[num_recv_rows + 1];

    // Transmit the row sizes from the source to the destination processors
    for (int i = 0, offset = 0; i < num_send_procs; i++) {
      MPI_Isend(&send_row_sizes[offset], send_row_count[i], MPI_INT,
                send_procs[i], tag, comm, &send_requests[i]);
      offset += send_row_count[i];
    }
    for (int i = 0, offset = 0; i < num_recv_procs; i++) {
      MPI_Irecv(&recv_row_ptr[offset + 1], recv_row_count[i], MPI_INT,
                recv_procs[i], tag, comm, &recv_requests[i]);
      offset += recv_row_count[i];
    }

    MPI_Waitall(num_recv_procs, recv_requests, MPI_STATUSES_IGNORE);
    MPI_Waitall(num_send_procs, send_requests, MPI_STATUSES_IGNORE);

    // Sum up the sizes to create a pointer into each row
    recv_row_ptr[0] = 0;
    for (int i = 0; i < num_recv_rows; i++) {
      recv_row_ptr[i + 1] += recv_row_ptr[i];
    }

    // Convert the column indices from local to global and sort them in each
    // row
    const int* col_ranges = col_owners->get_range();
    const int* ext_cols;
    col_owners->get_ext_nodes(&ext_cols);

    int num_local_cols = col_ranges[mpi_rank + 1] - col_ranges[mpi_rank];

    // Set up the column indices that will be sent
    int send_size = rowp[nrows] - rowp[num_local_rows];
    int* send_global_cols = new int[send_size];

    const int* c = &cols[rowp[num_local_rows]];
    for (int i = 0; i < send_size; i++, c++) {
      if (c[0] < num_local_cols) {
        send_global_cols[i] = c[0] + col_ranges[mpi_rank];
      } else {
        send_global_cols[i] = ext_cols[c[0] - num_local_cols];
      }
    }

    // Sort the global columns before sending them to the other processors
    for (int i = 0, *ptr = send_global_cols; i < num_ext_rows; i++) {
      int size = send_row_sizes[i];
      std::sort(ptr, ptr + size);
      ptr += size;
    }

    int* recv_cols = new int[num_recv_entries];

    // Distribute the columns to the matrices
    for (int i = 0, offset = 0; i < num_send_procs; i++) {
      MPI_Isend(&send_global_cols[offset], send_entry_count[i], MPI_INT,
                send_procs[i], tag, comm, &send_requests[i]);
      offset += send_entry_count[i];
    }
    for (int i = 0, offset = 0; i < num_recv_procs; i++) {
      MPI_Irecv(&recv_cols[offset], recv_entry_count[i], MPI_INT, recv_procs[i],
                tag, comm, &recv_requests[i]);
      offset += recv_entry_count[i];
    }

    MPI_Waitall(num_recv_procs, recv_requests, MPI_STATUSES_IGNORE);
    MPI_Waitall(num_send_procs, send_requests, MPI_STATUSES_IGNORE);

    delete[] send_global_cols;
    delete[] recv_requests;
    delete[] send_requests;

    // Set the pointer into the off-proc parts of the matrix
    int* ptr_to_rows = new int[num_recv_procs + 1];
    int* index_to_rows = new int[num_recv_procs];
    ptr_to_rows[0] = 0;
    for (int i = 0; i < num_recv_procs; i++) {
      index_to_rows[i] = ptr_to_rows[i];
      ptr_to_rows[i + 1] = ptr_to_rows[i] + recv_row_count[i];
    }

    delete[] send_row_sizes;
    delete[] recv_row_count;
    delete[] send_row_count;

    // Now that everything is on the root processor, assemble it all together
    // into a CSR matrix data structure. Include the local and external rows
    int* assembled_rowp = new int[nrows + 1];

    // Allocate the maximum size of the rows
    int max_cols_size = rowp[nrows] + recv_row_ptr[num_recv_rows];
    int* assembled_cols = new int[max_cols_size];

    // Merge the data to form a larger CSR matrix structure
    int nnz = 0;
    assembled_rowp[0] = 0;
    for (int i = 0; i < nrows; i++) {
      // Convert the local contribution into a global index
      for (int jp = rowp[i]; jp < rowp[i + 1]; jp++) {
        int col = 0;
        if (cols[jp] < num_local_cols) {
          col = cols[jp] + col_ranges[mpi_rank];
        } else {
          col = ext_cols[cols[jp] - num_local_cols];
        }

        // Add the entry to the column
        assembled_cols[nnz] = col;
        nnz++;
      }

      // Add the external values into the matrix into the matrix
      for (int j = 0; j < num_recv_procs; j++) {
        int index = index_to_rows[j];

        if (index < ptr_to_rows[j + 1] && recv_rows[index] == i) {
          for (int jp = recv_row_ptr[index]; jp < recv_row_ptr[index + 1];
               jp++) {
            int col = recv_cols[jp];

            // Add the entry to the column
            assembled_cols[nnz] = col;
            nnz++;
          }

          index_to_rows[j]++;
        }
      }

      // Sort and remove duplicates from the row
      int start = assembled_rowp[i];
      int size = nnz - assembled_rowp[i];
      std::sort(&assembled_cols[start], &assembled_cols[start] + size);
      auto last =
          std::unique(&assembled_cols[start], &assembled_cols[start] + size);

      // Update the size of the row
      nnz = last - assembled_cols;
      assembled_rowp[i + 1] = nnz;
    }

    // Create the CSR data structure
    csr = CSRMat<T>::create_from_csr_data(nrows, col_ranges[mpi_size], nnz,
                                          assembled_rowp, assembled_cols,
                                          mem_loc, row_owners, col_owners);

    delete[] ptr_to_rows;
    delete[] index_to_rows;
    delete[] assembled_rowp;
    delete[] assembled_cols;

    // Allocate a vector to store the locations of where to place the entries
    // of the CSR matrix from other processors
    recv_locs = std::make_shared<Vector<int>>(num_recv_entries, 0,
                                              MemoryLocation::HOST_AND_DEVICE);

    int* recv_loc_array = recv_locs->get_array();

    // Now, assemble all the components into the matrix
    for (int i = 0, ptr = 0; i < num_recv_rows; i++) {
      int row = recv_rows[i];
      int size = recv_row_ptr[i + 1] - recv_row_ptr[i];
      csr->get_sorted_locations(row, size, &recv_cols[ptr],
                                &recv_loc_array[ptr]);
      ptr += size;
    }

    // Copy the locations to the device if required
    if constexpr (policy == ExecPolicy::CUDA) {
      recv_locs->copy_host_to_device();
    }

    delete[] recv_rows;
    delete[] recv_row_ptr;
    delete[] recv_cols;
  }

  ~MatrixDistribute() {
    delete[] send_procs;
    delete[] send_entry_count;

    delete[] recv_procs;
    delete[] recv_entry_count;
  }

  /**
   * @brief Create a context object for distributing the components of the
   * matrix across different processors
   *
   * @tparam T type
   * @return MatDistributeContext<T>* The matrix distribution context
   */
  template <typename T>
  MatDistributeContext<T>* create_context() {
    int mpi_rank;
    MPI_Comm_rank(row_owners->get_mpi_comm(), &mpi_rank);
    const int* range = row_owners->get_range();
    int num_owned_rows = range[mpi_rank + 1] - range[mpi_rank];

    // Create the context
    return new MatDistributeContext<T>(num_send_procs, num_recv_procs,
                                       num_send_entries, recv_locs,
                                       num_owned_rows);
  }

  /**
   * @brief Begin assembly of the matrix - this initiates the communication of
   * external contributions to other processors
   *
   * @tparam T type
   * @param mat The matrix (or a duplicate) created at initialization
   * @param ctx The matrix distribution context
   */
  template <typename T>
  void begin_assembly(std::shared_ptr<CSRMat<T>> mat,
                      MatDistributeContext<T>* ctx) {
    T* recv_A = ctx->get_recv_buffer();
    T* send_A = ctx->get_send_entries(mat);

    // Send the data to the receiving processors
    for (int i = 0, offset = 0; i < num_send_procs; i++) {
      MPI_Isend(&send_A[offset], send_entry_count[i], get_mpi_type<T>(),
                send_procs[i], ctx->tag, comm, &ctx->send_requests[i]);
      offset += send_entry_count[i];
    }
    for (int i = 0, offset = 0; i < num_recv_procs; i++) {
      MPI_Irecv(&recv_A[offset], recv_entry_count[i], get_mpi_type<T>(),
                recv_procs[i], ctx->tag, comm, &ctx->recv_requests[i]);
      offset += recv_entry_count[i];
    }
  }

  /**
   * @brief Finalize assembly of the matrix, adding all local components to
   * the matrix
   *
   * @tparam T type
   * @param mat The matrix (or a duplicate) created at initialization
   * @param ctx The matrix distribution context
   */
  template <typename T>
  void end_assembly(std::shared_ptr<CSRMat<T>> mat,
                    MatDistributeContext<T>* ctx) {
    MPI_Waitall(num_recv_procs, ctx->recv_requests, MPI_STATUSES_IGNORE);
    MPI_Waitall(num_send_procs, ctx->send_requests, MPI_STATUSES_IGNORE);

    ctx->add_recv_entries(mat);
  }

 private:
  MPI_Comm comm;  // MPI Communicator

  // Row and column indices
  std::shared_ptr<NodeOwners> row_owners;
  std::shared_ptr<NodeOwners> col_owners;

  // Data destined for other processes
  // ---------------------------------
  int num_send_procs;     // Number of processors that will be sent data
  int num_send_entries;   // Number of entries that will be sent
  int* send_procs;        // External proc numbers
  int* send_entry_count;  // Number of entries sent to each proc

  // Data received from other processors
  // -----------------------------------
  int num_recv_procs;     // Number of processors that give contributions
  int num_recv_entries;   // Total number of entries from other procs
  int* recv_procs;        // Ranks of processors that send to this proc
  int* recv_entry_count;  // Number of entries sent to this proc

  // Store a pointer into the data array of the matrix
  std::shared_ptr<Vector<int>> recv_locs;
};

}  // namespace amigo

#endif  // AMIGO_MATRIX_DISTRIBUTE_H
