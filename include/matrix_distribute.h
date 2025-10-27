#ifndef AMIGO_MATRIX_DISTRIBUTE_H
#define AMIGO_MATRIX_DISTRIBUTE_H

#include <mpi.h>

#include "node_owners.h"
#include "ordering_utils.h"
#include "vector_distribute.h"

namespace amigo {

class MatrixDistribute {
 public:
  template <typename T>
  class MatDistributeContext {
   public:
    MatDistributeContext(int tag, int num_ext_procs, int num_in_procs,
                         int num_in_entries)
        : tag(tag) {
      in_A = new T[num_in_entries];
      in_requests = new MPI_Request[num_in_procs];
      ext_requests = new MPI_Request[num_ext_procs];
    }
    ~MatDistributeContext() {
      delete[] in_A;
      delete[] in_requests;
      delete[] ext_requests;
    }

    int tag;                    // MPI tag value
    T* in_A;                    // Storage for the incoming entries
    MPI_Request* in_requests;   // Requests for recving data
    MPI_Request* ext_requests;  // Requests for sending info
  };

  /**
   * @brief Construct a new Mat Distribute object
   *
   * @param comm
   * @param row_owners
   * @param col_owners
   * @param rowp
   * @param cols
   */
  template <typename T>
  MatrixDistribute(MPI_Comm comm, std::shared_ptr<NodeOwners> row_owners,
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
    int* full_ext_ptr = new int[mpi_size + 1];

    // Figure out where stuff should go by matching the intervals in the sorted
    // ext_rows array
    OrderingUtils::match_intervals(mpi_size, row_ranges, num_ext_rows, ext_rows,
                                   full_ext_ptr);

    // Count up the number of entries
    int* full_ext_count = new int[2 * mpi_size];
    for (int i = 0; i < mpi_size; i++) {
      // Set the number of rows that will be sent
      full_ext_count[2 * i] = full_ext_ptr[i + 1] - full_ext_ptr[i];

      // Set the number of entries that will be sent
      int start = full_ext_ptr[i] + num_local_rows;
      int end = full_ext_ptr[i + 1] + num_local_rows;
      full_ext_count[2 * i + 1] = rowp[end] - rowp[start];
    }
    delete[] full_ext_ptr;

    // Allocate space to store the number of in or out rows
    int* full_in_count = new int[2 * mpi_size];

    // Do one Alltoall
    MPI_Alltoall(full_ext_count, 2, MPI_INT, full_in_count, 2, MPI_INT, comm);

    // Count up the number of sends/recvs
    num_ext_procs = 0;
    num_in_procs = 0;
    for (int i = 0; i < mpi_size; i++) {
      if (i != mpi_rank && full_in_count[2 * i] > 0) {
        num_in_procs++;
      }
      if (i != mpi_rank && full_ext_count[2 * i] > 0) {
        num_ext_procs++;
      }
    }

    // Number of external processors from which we are getting external
    // variable values
    in_procs = new int[num_in_procs];
    in_entry_count = new int[num_in_procs];
    int* in_row_count = new int[num_in_procs];

    // Number of processors to which we are sending data
    ext_procs = new int[num_ext_procs];
    ext_entry_count = new int[num_ext_procs];
    int* ext_row_count = new int[num_ext_procs];

    // Count up the rows to send first
    num_in_rows = 0;
    num_in_entries = 0;
    for (int i = 0, ext = 0, in = 0; i < mpi_size; i++) {
      if (i != mpi_rank && full_ext_count[2 * i] > 0) {
        ext_procs[ext] = i;
        ext_row_count[ext] = full_ext_count[2 * i];
        ext_entry_count[ext] = full_ext_count[2 * i + 1];
        ext++;
      }
      if (i != mpi_rank && full_in_count[2 * i] > 0) {
        in_procs[in] = i;
        in_row_count[in] = full_in_count[2 * i];
        in_entry_count[in] = full_in_count[2 * i + 1];
        num_in_rows += in_row_count[in];
        num_in_entries += in_entry_count[in];
        in++;
      }
    }

    delete[] full_ext_count;
    delete[] full_in_count;

    // Indices of the data that we will send to the processors
    in_rows = new int[num_in_rows];

    MPI_Request* in_requests = new MPI_Request[num_in_procs];
    MPI_Request* ext_requests = new MPI_Request[num_ext_procs];

    // Transmit the rows from the source to the destination processors
    int tag = 0;
    for (int i = 0, offset = 0; i < num_ext_procs; i++) {
      MPI_Isend(&ext_rows[offset], ext_row_count[i], MPI_INT, ext_procs[i], tag,
                comm, &ext_requests[i]);
      offset += ext_row_count[i];
    }
    for (int i = 0, offset = 0; i < num_in_procs; i++) {
      MPI_Irecv(&in_rows[offset], in_row_count[i], MPI_INT, in_procs[i], tag,
                comm, &in_requests[i]);
      offset += in_row_count[i];
    }

    MPI_Waitall(num_in_procs, in_requests, MPI_STATUSES_IGNORE);
    MPI_Waitall(num_ext_procs, ext_requests, MPI_STATUSES_IGNORE);

    // Convert to the local ordering
    for (int i = 0; i < num_in_rows; i++) {
      in_rows[i] -= row_ranges[mpi_rank];
    }

    // Now send the row sizes
    int* ext_row_sizes = new int[num_ext_rows];
    for (int i = 0; i < num_ext_rows; i++) {
      ext_row_sizes[i] =
          rowp[i + 1 + num_local_rows] - rowp[i + num_local_rows];
    }

    // Allocate space for the row sizes
    in_row_ptr = new int[num_in_rows + 1];

    // Transmit the row sizes from the source to the destination processors
    for (int i = 0, offset = 0; i < num_ext_procs; i++) {
      MPI_Isend(&ext_row_sizes[offset], ext_row_count[i], MPI_INT, ext_procs[i],
                tag, comm, &ext_requests[i]);
      offset += ext_row_count[i];
    }
    for (int i = 0, offset = 0; i < num_in_procs; i++) {
      MPI_Irecv(&in_row_ptr[offset + 1], in_row_count[i], MPI_INT, in_procs[i],
                tag, comm, &in_requests[i]);
      offset += in_row_count[i];
    }

    MPI_Waitall(num_in_procs, in_requests, MPI_STATUSES_IGNORE);
    MPI_Waitall(num_ext_procs, ext_requests, MPI_STATUSES_IGNORE);

    // Sum up the sizes to create a pointer into each row
    in_row_ptr[0] = 0;
    for (int i = 0; i < num_in_rows; i++) {
      in_row_ptr[i + 1] += in_row_ptr[i];
    }

    // Convert the column indices from local to global and sort them in each row
    const int* col_ranges = col_owners->get_range();
    const int* ext_cols;
    col_owners->get_ext_nodes(&ext_cols);

    int num_local_cols = col_ranges[mpi_rank + 1] - col_ranges[mpi_rank];

    // Set up the column indices that will be sent
    int ext_size = rowp[nrows] - rowp[num_local_rows];
    int* ext_global_cols = new int[ext_size];

    const int* c = &cols[rowp[num_local_rows]];
    for (int i = 0; i < ext_size; i++, c++) {
      if (c[0] < num_local_cols) {
        ext_global_cols[i] = c[0] + col_ranges[mpi_rank];
      } else {
        ext_global_cols[i] = ext_cols[c[0] - num_local_cols];
      }
    }

    // Sort the global columns before sending them to the other processors
    for (int i = 0, *ptr = ext_global_cols; i < num_ext_rows; i++) {
      int size = ext_row_sizes[i];
      std::sort(ptr, ptr + size);
      ptr += size;
    }

    in_cols = new int[num_in_entries];

    // Distribute the columns to the matrices
    for (int i = 0, offset = 0; i < num_ext_procs; i++) {
      MPI_Isend(&ext_global_cols[offset], ext_entry_count[i], MPI_INT,
                ext_procs[i], tag, comm, &ext_requests[i]);
      offset += ext_entry_count[i];
    }
    for (int i = 0, offset = 0; i < num_in_procs; i++) {
      MPI_Irecv(&in_cols[offset], in_entry_count[i], MPI_INT, in_procs[i], tag,
                comm, &in_requests[i]);
      offset += in_entry_count[i];
    }

    MPI_Waitall(num_in_procs, in_requests, MPI_STATUSES_IGNORE);
    MPI_Waitall(num_ext_procs, ext_requests, MPI_STATUSES_IGNORE);

    delete[] ext_global_cols;
    delete[] in_requests;
    delete[] ext_requests;

    // Set the pointer into the off-proc parts of the matrix
    int* ptr_to_rows = new int[num_in_procs + 1];
    int* index_to_rows = new int[num_in_procs];
    ptr_to_rows[0] = 0;
    for (int i = 0; i < num_in_procs; i++) {
      index_to_rows[i] = ptr_to_rows[i];
      ptr_to_rows[i + 1] = ptr_to_rows[i] + in_row_count[i];
    }

    delete[] ext_row_sizes;
    delete[] in_row_count;
    delete[] ext_row_count;

    // Now that everything is on the root processor, assemble it all together
    // into a CSR matrix data structure. Include the local and external rows
    int* assembled_rowp = new int[nrows + 1];

    // Allocate the maximum size of the rows
    int max_cols_size = rowp[nrows] + in_row_ptr[num_in_rows];
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
      for (int j = 0; j < num_in_procs; j++) {
        int index = index_to_rows[j];

        if (index < ptr_to_rows[j + 1] && in_rows[index] == i) {
          for (int jp = in_row_ptr[index]; jp < in_row_ptr[index + 1]; jp++) {
            int col = in_cols[jp];

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
                                          row_owners, col_owners);

    delete[] ptr_to_rows;
    delete[] index_to_rows;
    delete[] assembled_rowp;
    delete[] assembled_cols;
  }

  ~MatrixDistribute() {
    delete[] ext_procs;
    delete[] ext_entry_count;

    delete[] in_procs;
    delete[] in_entry_count;
    delete[] in_rows;
    delete[] in_row_ptr;
    delete[] in_cols;
  }

  /**
   * @brief Create a context object for distributing the components of the
   * matrix across different processors
   *
   * @tparam T type
   * @return MatDistributeContext<T>* The matrix distribution context
   */
  template <typename T>
  MatrixDistribute::MatDistributeContext<T>* create_context() {
    return new MatrixDistribute::MatDistributeContext<T>(
        0, num_ext_procs, num_in_procs, num_in_entries);
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
    // Get the pointer to the external part of A
    int num_local_rows = row_owners->get_local_size();

    const int* rowp;
    T* A;
    mat->get_data(nullptr, nullptr, nullptr, &rowp, nullptr, &A);

    // Send the data to the receiving processors
    for (int i = 0, offset = rowp[num_local_rows]; i < num_ext_procs; i++) {
      MPI_Isend(&A[offset], ext_entry_count[i], get_mpi_type<T>(), ext_procs[i],
                ctx->tag, comm, &ctx->ext_requests[i]);
      offset += ext_entry_count[i];
    }
    for (int i = 0, offset = 0; i < num_in_procs; i++) {
      MPI_Irecv(&ctx->in_A[offset], in_entry_count[i], get_mpi_type<T>(),
                in_procs[i], ctx->tag, comm, &ctx->in_requests[i]);
      offset += in_entry_count[i];
    }
  }

  /**
   * @brief Finalize assembly of the matrix, adding all local components to the
   * matrix
   *
   * @tparam T type
   * @param mat The matrix (or a duplicate) created at initialization
   * @param ctx The
   *  matrix distribution context
   */
  template <typename T>
  void end_assembly(std::shared_ptr<CSRMat<T>> mat,
                    MatDistributeContext<T>* ctx) {
    MPI_Waitall(num_in_procs, ctx->in_requests, MPI_STATUSES_IGNORE);
    MPI_Waitall(num_ext_procs, ctx->ext_requests, MPI_STATUSES_IGNORE);

    // Now, assemble all the components into the matrix
    for (int i = 0, ptr = 0; i < num_in_rows; i++) {
      int row = in_rows[i];
      int size = in_row_ptr[i + 1] - in_row_ptr[i];
      mat->add_row_sorted(row, size, &in_cols[ptr], &ctx->in_A[ptr]);
      ptr += size;
    }
  }

 private:
  MPI_Comm comm;

  // Row and column indices
  std::shared_ptr<NodeOwners> row_owners;
  std::shared_ptr<NodeOwners> col_owners;

  // Data destined for other processes
  // ---------------------------------
  int num_ext_procs;     // Number of processors that will be sent data
  int* ext_procs;        // External proc numbers
  int* ext_entry_count;  // Number of entries sent to each proc

  // Data received from other processors
  // -----------------------------------
  int num_in_procs;     // Number of processors that give contributions
  int num_in_rows;      // Total number of rows from other procs
  int num_in_entries;   // Total number of entries from other procs
  int* in_procs;        // Ranks of processors that send to this proc
  int* in_entry_count;  // Number of entries sent to this proc
  int* in_rows;         // Row indices received (converted to local row index)
  int* in_row_ptr;      // Pointer into the columns object of each row received
  int* in_cols;         // Global column indices
};

}  // namespace amigo

#endif  // AMIGO_MATRIX_DISTRIBUTE_H
