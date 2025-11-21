#ifndef AMIGO_OPTIMIZATION_PROBLEM_H
#define AMIGO_OPTIMIZATION_PROBLEM_H

#include <mpi.h>

#include "component_group_base.h"
#include "matrix_distribute.h"
#include "node_owners.h"
#include "vector_distribute.h"

namespace amigo {

template <typename T, ExecPolicy policy>
class OptimizationProblem {
 public:
  /**
   * @brief Construct a new Optimization Problem object
   *
   * @param comm The MPI communicator
   * @param data_owners The owners of the data
   * @param var_owners The owners for the variables (inputs/multipliers)
   * @param output_owners The owners for the outputs
   * @param is_multiplier Integer array indicating if this is a multiper or not
   * @param components The component groups for the model
   */
  OptimizationProblem(
      MPI_Comm comm, std::shared_ptr<NodeOwners> data_owners,
      std::shared_ptr<NodeOwners> var_owners,
      std::shared_ptr<NodeOwners> output_owners,
      std::shared_ptr<Vector<int>> is_multiplier_,
      const std::vector<std::shared_ptr<ComponentGroupBase<T, policy>>>&
          components)
      : comm(comm),
        data_owners(data_owners),
        var_owners(var_owners),
        output_owners(output_owners),
        is_multiplier(is_multiplier_),
        components(components),
        data_dist(data_owners),
        var_dist(var_owners),
        output_dist(output_owners) {
    data_ctx = data_dist.template create_context<T>();
    var_ctx = var_dist.template create_context<T>();
    output_ctx = output_dist.template create_context<T>();

    data_vec = create_data_vector();

    // Set the default array (no multipliers) if none is provided
    if (!is_multiplier) {
      is_multiplier = std::make_shared<Vector<int>>(
          var_owners->get_local_size(), var_owners->get_ext_size());
    }

    dist_node_numbers = nullptr;
    dist_data_numbers = nullptr;
    dist_output_numbers = nullptr;

    // Hessian matrix
    mat = nullptr;
    mat_dist = nullptr;
    mat_dist_ctx = nullptr;

    // Derivative of output wrt inputs
    input_jac = nullptr;
    input_jac_dist = nullptr;
    input_jac_dist_ctx = nullptr;

    // Derivative of the output wrt data
    data_jac = nullptr;
    data_jac_dist = nullptr;
    data_jac_dist_ctx = nullptr;

    // Derivative of the gradient wrt data
    grad_jac = nullptr;
    grad_jac_dist = nullptr;
    grad_jac_dist_ctx = nullptr;
  }
  ~OptimizationProblem() {
    delete data_ctx;
    delete var_ctx;
    delete output_ctx;

    if (mat_dist) {
      delete mat_dist;
    }
    if (mat_dist_ctx) {
      delete mat_dist_ctx;
    }
    if (input_jac_dist) {
      delete input_jac_dist;
    }
    if (input_jac_dist_ctx) {
      delete input_jac_dist_ctx;
    }
    if (data_jac_dist) {
      delete data_jac_dist;
    }
    if (data_jac_dist_ctx) {
      delete data_jac_dist_ctx;
    }
    if (grad_jac_dist) {
      delete grad_jac_dist;
    }
    if (grad_jac_dist_ctx) {
      delete grad_jac_dist_ctx;
    }
  }

  /**
   * @brief Get the MPI communicator
   *
   * @return MPI_Comm
   */
  MPI_Comm get_mpi_comm() { return comm; }

  /**
   * @brief Get the num variables that are owned by this processor
   *
   * @return int Number of variables
   */
  int get_num_variables() const { return var_owners->get_local_size(); }

  /**
   * @brief Create a vector object for the input and multipliers
   *
   * @return std::shared_ptr<Vector<T>> The vector object
   */
  std::shared_ptr<Vector<T>> create_vector(
      MemoryLocation loc = MemoryLocation::HOST_AND_DEVICE) const {
    return std::make_shared<Vector<T>>(var_owners->get_local_size(),
                                       var_owners->get_ext_size(), loc);
  }

  /**
   * @brief Create a output vector object for storing output values
   *
   * @return std::shared_ptr<Vector<T>> The output vector object
   */
  std::shared_ptr<Vector<T>> create_output_vector(
      MemoryLocation loc = MemoryLocation::HOST_AND_DEVICE) const {
    return std::make_shared<Vector<T>>(output_owners->get_local_size(),
                                       output_owners->get_ext_size(), loc);
  }

  /**
   * @brief Get the data vector object that is stored internally
   *
   * @return std::shared_ptr<Vector<T>> The data object stored locally
   */
  std::shared_ptr<Vector<T>> get_data_vector() { return data_vec; }

  /**
   * @brief Set the data vector object replacing the existing one
   *
   * @param vec The data vector to use
   */
  void set_data_vector(std::shared_ptr<Vector<T>> vec) { data_vec = vec; }

  /**
   * @brief Create a data vector object
   *
   * @return std::shared_ptr<Vector<T>> The data vector object
   */
  std::shared_ptr<Vector<T>> create_data_vector(
      MemoryLocation loc = MemoryLocation::HOST_AND_DEVICE) const {
    return std::make_shared<Vector<T>>(data_owners->get_local_size(),
                                       data_owners->get_ext_size(), loc);
  }

  /**
   * @brief Get the multiplier indicator vector
   *
   * This indicates which components of the vector are multipliers and which are
   * inputs (variables). This should not be changed.
   *
   * @return const std::shared_ptr<Vector<int>>
   */
  const std::shared_ptr<Vector<int>> get_multiplier_indicator() const {
    return is_multiplier;
  }

  /**
   * @brief Get the node indices for distributing from local to distributed
   * vectors
   *
   * @return std::shared_ptr<Vector<int>>
   */
  std::shared_ptr<Vector<int>> get_local_to_global_node_numbers() {
    return dist_node_numbers;
  }

  /**
   * @brief Get the data indices for distributing from local to distributed
   * vectors
   *
   * @return std::shared_ptr<Vector<int>>
   */
  std::shared_ptr<Vector<int>> get_local_to_global_data_numbers() {
    return dist_data_numbers;
  }

  /**
   * @brief Partition the optimization problem across the available processors
   * from the root processor.
   *
   * In this case, the entire problem must be set up on the root processors,
   * with empty component groups on all remaining processors.
   *
   * @param root Rank of the root processor
   * @return OptimizationProblem<T> The new OptimizationProblem on all procs
   */
  std::shared_ptr<OptimizationProblem<T, policy>> partition_from_root(
      int root = 0) {
    int mpi_size, mpi_rank;
    MPI_Comm_size(comm, &mpi_size);
    MPI_Comm_rank(comm, &mpi_rank);

    int* partition = nullptr;
    int* var_ranges = new int[mpi_size + 1];
    int* data_ranges = new int[mpi_size + 1];
    int* output_ranges = new int[mpi_size + 1];

    if (mpi_rank == root) {
      // The intervals must be the same for both
      std::vector<int> intervals;
      auto element_nodes = get_element_nodes(intervals);
      int nelems = intervals[intervals.size() - 1];

      std::vector<int> data_intervals;
      auto element_data = get_element_data(data_intervals);

      std::vector<int> output_intervals;
      auto element_output = get_element_output(output_intervals);

      // Number of data entries
      const int* data_range = data_owners->get_range();
      int ndata = data_range[root + 1] - data_range[root];

      // Number of nodes
      const int* node_range = var_owners->get_range();
      int nnodes = node_range[root + 1] - node_range[root];

      // Number of outputs
      const int* output_range = output_owners->get_range();
      int noutputs = output_range[root + 1] - output_range[root];

      // Compute the element partition
      OrderingUtils::compute_partition(nnodes, nelems, element_nodes, mpi_size,
                                       &partition);

      dist_node_numbers = std::make_shared<Vector<int>>(nnodes);
      dist_data_numbers = std::make_shared<Vector<int>>(ndata);
      dist_output_numbers = std::make_shared<Vector<int>>(noutputs);

      // Set the new node numbers for the element nodes and element dataa
      OrderingUtils::reorder_for_partition(
          ndata, nelems, element_data, mpi_size, partition,
          dist_data_numbers->get_array(), data_ranges);

      OrderingUtils::reorder_for_partition(
          nnodes, nelems, element_nodes, mpi_size, partition,
          dist_node_numbers->get_array(), var_ranges);

      OrderingUtils::reorder_for_partition(
          noutputs, nelems, element_output, mpi_size, partition,
          dist_output_numbers->get_array(), output_ranges);
    }

    MPI_Bcast(data_ranges, mpi_size + 1, MPI_INT, root, comm);
    MPI_Bcast(var_ranges, mpi_size + 1, MPI_INT, root, comm);
    MPI_Bcast(output_ranges, mpi_size + 1, MPI_INT, root, comm);

    std::vector<int> new_nelems(components.size());
    std::vector<std::shared_ptr<Vector<int>>> new_nodes(components.size());
    std::vector<std::shared_ptr<Vector<int>>> new_data(components.size());
    std::vector<std::shared_ptr<Vector<int>>> new_outputs(components.size());

    // Partition the element and data
    int* part = partition;
    for (size_t i = 0; i < components.size(); i++) {
      // Create a new vector for the data
      int nelems;
      int ndata_per_elem;
      const int* elem_data;
      components[i]->get_data_layout_data(&nelems, &ndata_per_elem, &elem_data);

      int nelems_local = 0;
      int* elem_data_local = nullptr;
      int* new_data_numbers = nullptr;
      if (dist_data_numbers) {
        new_data_numbers = dist_data_numbers->get_array();
      }

      if (ndata_per_elem > 0) {
        OrderingUtils::distribute_elements(
            comm, nelems, ndata_per_elem, elem_data, part, new_data_numbers,
            &nelems_local, &elem_data_local, root);
      }

      new_data[i] = std::make_shared<Vector<int>>(nelems_local * ndata_per_elem,
                                                  0, &elem_data_local);

      // Create the new vector for the variables
      int nnodes_per_elem;
      const int* elem_nodes = nullptr;
      components[i]->get_layout_data(&nelems, &nnodes_per_elem, &elem_nodes);

      int* new_node_numbers = nullptr;
      if (dist_node_numbers) {
        new_node_numbers = dist_node_numbers->get_array();
      }

      int* elem_nodes_local = nullptr;
      OrderingUtils::distribute_elements(
          comm, nelems, nnodes_per_elem, elem_nodes, part, new_node_numbers,
          &nelems_local, &elem_nodes_local, root);

      new_nelems[i] = nelems_local;
      new_nodes[i] = std::make_shared<Vector<int>>(
          nelems_local * nnodes_per_elem, 0, &elem_nodes_local);

      // Create the new vector for the outputs
      int outputs_per_elem;
      const int* elem_outputs = nullptr;
      components[i]->get_output_layout_data(&nelems, &outputs_per_elem,
                                            &elem_outputs);

      int* new_output_numbers = nullptr;
      if (dist_output_numbers) {
        new_output_numbers = dist_output_numbers->get_array();
      }

      int* elem_outputs_local = nullptr;
      OrderingUtils::distribute_elements(
          comm, nelems, outputs_per_elem, elem_outputs, part,
          new_output_numbers, &nelems_local, &elem_outputs_local, root);

      new_outputs[i] = std::make_shared<Vector<int>>(
          nelems_local * outputs_per_elem, 0, &elem_outputs_local);

      // Increment the pointer into the partition
      part += nelems;
    }

    // Reorder the data array with a local ordering
    std::vector<int> ext_data_nodes;
    reorder_from_global_to_local(data_ranges[mpi_rank],
                                 data_ranges[mpi_rank + 1], new_data,
                                 ext_data_nodes);
    std::shared_ptr<NodeOwners> new_data_owners = std::make_shared<NodeOwners>(
        comm, data_ranges, ext_data_nodes.size(), ext_data_nodes.data());

    // Reorder the variables/constraints with a local ordering
    std::vector<int> ext_nodes;
    reorder_from_global_to_local(var_ranges[mpi_rank], var_ranges[mpi_rank + 1],
                                 new_nodes, ext_nodes);
    std::shared_ptr<NodeOwners> new_var_owners = std::make_shared<NodeOwners>(
        comm, var_ranges, ext_nodes.size(), ext_nodes.data());

    // Reorder the outputs with a local ordering
    std::vector<int> ext_output_nodes;
    reorder_from_global_to_local(output_ranges[mpi_rank],
                                 output_ranges[mpi_rank + 1], new_outputs,
                                 ext_output_nodes);
    std::shared_ptr<NodeOwners> new_output_owners =
        std::make_shared<NodeOwners>(comm, output_ranges,
                                     ext_output_nodes.size(),
                                     ext_output_nodes.data());

    // Allocate space to store the new components
    std::vector<std::shared_ptr<ComponentGroupBase<T, policy>>> new_comps(
        components.size());

    // Make the new component with the new ordering
    for (size_t i = 0; i < components.size(); i++) {
      new_comps[i] = components[i]->clone(new_nelems[i], new_data[i],
                                          new_nodes[i], new_outputs[i]);
    }

    // Free memory here
    if (partition) {
      delete[] partition;
    }

    std::shared_ptr<OptimizationProblem<T, policy>> opt =
        std::make_shared<OptimizationProblem<T, policy>>(
            comm, new_data_owners, new_var_owners, new_output_owners, nullptr,
            new_comps);

    bool distribute = true;
    scatter_vector(is_multiplier, opt, opt->is_multiplier, root, distribute);

    return opt;
  }

  /**
   * @brief Scatter vector components from the root processor to a distributed
   * version of the vector
   *
   * @tparam T1 The type of the vector
   * @param root_vec The vector on the root processor
   * @param dist_prob The distributed version of the problem
   * @param dist_vec The distributed vector (output from the code)
   * @param root The rank of the root processor
   * @param distribute Boolean indicating whether to distribute external values
   */
  template <typename T1>
  void scatter_vector(
      const std::shared_ptr<Vector<T1>> root_vec,
      const std::shared_ptr<OptimizationProblem<T, policy>> dist_prob,
      std::shared_ptr<Vector<T1>> dist_vec, int root = 0,
      bool distribute = true) {
    int mpi_rank, mpi_size;
    MPI_Comm_rank(comm, &mpi_rank);
    MPI_Comm_size(comm, &mpi_size);

    T1* reordered = nullptr;
    int* counts = nullptr;
    const int* disp = nullptr;

    if (mpi_rank == root) {
      int size = root_vec->get_size();
      reordered = new T1[size];

      counts = new int[mpi_size];
      disp = dist_prob->var_owners->get_range();
      for (int i = 0; i < mpi_size; i++) {
        counts[i] = disp[i + 1] - disp[i];
      }

      if (dist_node_numbers) {
        const int* new_node_numbers = dist_node_numbers->get_array();
        const T1* array = root_vec->get_array();

        for (int i = 0; i < size; i++) {
          reordered[new_node_numbers[i]] = array[i];
        }
      }
    }

    MPI_Scatterv(reordered, counts, disp, get_mpi_type<T1>(),
                 dist_vec->get_array(), dist_vec->get_size(),
                 get_mpi_type<T1>(), root, comm);

    if (reordered) {
      delete[] reordered;
      delete[] counts;
    }

    if (distribute) {
      auto ctx = dist_prob->var_dist.template create_context<T1>();

      // If policy == CUDA, copy the vector to the device
      if constexpr (policy == ExecPolicy::CUDA) {
        dist_vec->copy_host_to_device();
      }

      dist_prob->var_dist.begin_forward(dist_vec, ctx);
      dist_prob->var_dist.end_forward(dist_vec, ctx);

      delete ctx;
    }
  }

  /**
   * @brief Gather vector components to the root processor
   *
   * @tparam T1 The data type of the vector
   * @param dist_prob The distributed version of the problem
   * @param dist_vec The distributed vector
   * @param root_vec The vector on the root processor
   * @param root The rank of the root processor
   */
  template <typename T1>
  void gather_vector(
      const std::shared_ptr<OptimizationProblem<T, policy>> dist_prob,
      const std::shared_ptr<Vector<T1>> dist_vec,
      std::shared_ptr<Vector<T1>> root_vec, int root = 0) {
    int mpi_rank, mpi_size;
    MPI_Comm_rank(comm, &mpi_rank);
    MPI_Comm_size(comm, &mpi_size);

    T1* reordered = nullptr;
    int* counts = nullptr;
    const int* disp = nullptr;

    if (mpi_rank == root) {
      int size = root_vec->get_size();
      reordered = new T1[size];

      counts = new int[mpi_size];
      disp = dist_prob->var_owners->get_range();
      for (int i = 0; i < mpi_size; i++) {
        counts[i] = disp[i + 1] - disp[i];
      }
    }

    int size = dist_prob->var_owners->get_local_size();
    const T1* array = dist_vec->get_array();
    MPI_Gatherv(array, size, get_mpi_type<T1>(), reordered, counts, disp,
                get_mpi_type<T1>(), root, comm);

    // Reorder the local vector
    if (mpi_rank == root && dist_node_numbers) {
      const int* new_node_numbers = dist_node_numbers->get_array();
      int root_size = root_vec->get_size();
      T1* root_array = root_vec->get_array();

      for (int i = 0; i < root_size; i++) {
        root_array[i] = reordered[new_node_numbers[i]];
      }
    }

    if (reordered) {
      delete[] reordered;
      delete[] counts;
    }
  }

  /**
   * @brief Scatter vector components from the root processor to a distributed
   * version of the vector
   *
   * @tparam T1 The type of the vector
   * @param root_vec The vector on the root processor
   * @param dist_prob The distributed version of the problem
   * @param dist_vec The distributed vector (output from the code)
   * @param root The root processor
   * @param distribute Boolean indicating whether to distribute external values
   */
  template <typename T1>
  void scatter_data_vector(
      const std::shared_ptr<Vector<T1>> root_vec,
      const std::shared_ptr<OptimizationProblem<T, policy>> dist_prob,
      std::shared_ptr<Vector<T1>> dist_vec, int root = 0,
      bool distribute = true) {
    int mpi_rank, mpi_size;
    MPI_Comm_rank(comm, &mpi_rank);
    MPI_Comm_size(comm, &mpi_size);

    T1* reordered = nullptr;
    int* counts = nullptr;
    const int* disp = nullptr;

    if (mpi_rank == root) {
      int size = root_vec->get_size();
      reordered = new T1[size];

      counts = new int[mpi_size];
      disp = dist_prob->data_owners->get_range();
      for (int i = 0; i < mpi_size; i++) {
        counts[i] = disp[i + 1] - disp[i];
      }

      if (dist_data_numbers) {
        const int* new_data_numbers = dist_data_numbers->get_array();
        const T1* array = root_vec->get_array();

        for (int i = 0; i < size; i++) {
          reordered[new_data_numbers[i]] = array[i];
        }
      }
    }

    MPI_Scatterv(reordered, counts, disp, get_mpi_type<T1>(),
                 dist_vec->get_array(), dist_vec->get_size(),
                 get_mpi_type<T1>(), root, comm);

    if (reordered) {
      delete[] reordered;
      delete[] counts;
    }

    if (distribute) {
      auto ctx = dist_prob->data_dist.template create_context<T1>();

      if constexpr (policy == ExecPolicy::CUDA) {
        dist_vec->copy_host_to_device();
      }

      dist_prob->data_dist.begin_forward(dist_vec, ctx);
      dist_prob->data_dist.end_forward(dist_vec, ctx);

      delete ctx;
    }
  }

  /**
   * @brief Create the CSR matrix object that can store the Hessian
   *
   * @return std::shared_ptr<CSRMat<T>>
   */
  std::shared_ptr<CSRMat<T>> create_matrix(
      MemoryLocation mem_loc = MemoryLocation::HOST_AND_DEVICE) {
    if (mat) {
      return mat->duplicate();
    } else {
      std::vector<int> intervals;
      auto element_nodes = get_element_nodes(intervals);
      int num_elements = intervals[components.size()];
      int num_variables =
          var_owners->get_local_size() + var_owners->get_ext_size();

      // Create the local CSR structure
      bool include_diagonal = true;

      // Columns sorted in MatrixDistribute initialization
      bool sort_columns = false;

      // Generate the non-zero pattern
      int *rowp, *cols;
      OrderingUtils::create_csr_from_element_conn(
          num_variables, num_variables, num_elements, element_nodes,
          include_diagonal, sort_columns, &rowp, &cols);

      // Now optionally add the components with a non-zero CSR sparsity pattern
      int num_csr_components = 0;
      std::vector<int> csr_components(components.size());
      for (size_t i = 0; i < components.size(); i++) {
        int nvars, ncon;
        components[i]->get_csr_data(&nvars, nullptr, &ncon, nullptr, nullptr,
                                    nullptr, nullptr, nullptr);
        if (nvars > 0 || ncon > 0) {
          csr_components[num_csr_components] = i;
          num_csr_components++;
        }
      }

      if (num_csr_components > 0) {
        auto get_csr_component =
            [&](int index, int* nvars, const int* vars[], int* ncon,
                const int* cons[], const int* jac_rowp[], const int* jac_cols[],
                const int* hess_rowp[], const int* hess_cols[]) {
              int comp_index = csr_components[index];
              components[comp_index]->get_csr_data(nvars, vars, ncon, cons,
                                                   jac_rowp, jac_cols,
                                                   hess_rowp, hess_cols);
              return;
            };

        int *new_rowp, *new_cols;
        OrderingUtils::add_extern_csr_pattern(
            num_variables, num_variables, rowp, cols, num_csr_components,
            get_csr_component, &new_rowp, &new_cols);

        delete[] rowp;
        delete[] cols;

        rowp = new_rowp;
        cols = new_cols;
      }

      // Distribute the pattern across matrices
      mat_dist = new MatrixDistribute<policy>(comm, mem_loc, var_owners,
                                              var_owners, num_variables,
                                              num_variables, rowp, cols, mat);
      mat_dist_ctx = mat_dist->template create_context<T>();

      delete[] rowp;
      delete[] cols;

      // Perform any component group initializations with the Hessian matrix
      for (size_t i = 0; i < components.size(); i++) {
        components[i]->initialize_hessian_pattern(*var_owners, *mat);
      }

      return mat;
    }
  }

  /**
   * @brief Update the design variables - needed for external components that
   * evaluate their terms independently
   *
   * @param x The design variable values
   */
  void update(std::shared_ptr<Vector<T>> x) {
    var_dist.begin_forward(x, var_ctx);
    var_dist.end_forward(x, var_ctx);

    for (size_t i = 0; i < components.size(); i++) {
      components[i]->update(*x);
    }
  }

  /**
   * @brief Compute the value of the Lagrangian
   *
   * @param alpha Scalar multiplier for the objective function
   * @param x The design variable values
   * @return The value of the Lagrangian
   */
  T lagrangian(T alpha, std::shared_ptr<Vector<T>> x) {
    var_dist.begin_forward(x, var_ctx);
    var_dist.end_forward(x, var_ctx);

    T lagrange = 0.0;
    for (size_t i = 0; i < components.size(); i++) {
      lagrange += components[i]->lagrangian(alpha, *data_vec, *x);
    }

    T value = lagrange;
    MPI_Allreduce(&value, &lagrange, 1, get_mpi_type<T>(), MPI_SUM, comm);
    return lagrange;
  }

  /**
   * @brief Compute the gradient of the Lagrangian for all components
   *
   * @param alpha Scalar multiplier for the objective function
   * @param x The design variable vector
   * @param g The output gradient vector
   */
  void gradient(T alpha, const std::shared_ptr<Vector<T>> x,
                std::shared_ptr<Vector<T>> g) {
    var_dist.begin_forward(x, var_ctx);
    var_dist.end_forward(x, var_ctx);

    g->zero();
    for (size_t i = 0; i < components.size(); i++) {
      components[i]->add_gradient(alpha, *data_vec, *x, *g);
    }

    var_dist.begin_reverse_add(g, var_ctx);
    var_dist.end_reverse_add(g, var_ctx);
  }

  /**
   * @brief Compute a Hessian-vector product with the Lagrangian
   *
   * @param alpha Scalar multiplier for the objective function
   * @param x The design variable vector
   * @param p The product direction
   * @param h The output Hessian-vector product
   */
  void hessian_product(T alpha, const std::shared_ptr<Vector<T>> x,
                       const std::shared_ptr<Vector<T>> p,
                       std::shared_ptr<Vector<T>> h) {
    var_dist.begin_forward(x, var_ctx);
    var_dist.end_forward(x, var_ctx);

    var_dist.begin_forward(p, var_ctx);
    var_dist.end_forward(p, var_ctx);

    h->zero();
    for (size_t i = 0; i < components.size(); i++) {
      components[i]->add_hessian_product(alpha, *data_vec, *x, *p, *h);
    }

    var_dist.begin_reverse_add(h, var_ctx);
    var_dist.end_reverse_add(h, var_ctx);
  }

  /**
   * @brief Compute the full Hessian matrix
   *
   * @param alpha Scalar multiplier for the objective function
   * @param x The design variable values
   * @param mat The full Hessian matrix
   */
  void hessian(T alpha, const std::shared_ptr<Vector<T>> x,
               std::shared_ptr<CSRMat<T>> matrix) {
    var_dist.begin_forward(x, var_ctx);
    var_dist.end_forward(x, var_ctx);

    // Compute the Hessian matrix entries
    matrix->zero();
    for (size_t i = 0; i < components.size(); i++) {
      components[i]->add_hessian(alpha, *data_vec, *x, *var_owners, *matrix);
    }

    mat_dist->begin_assembly(matrix, mat_dist_ctx);
    mat_dist->end_assembly(matrix, mat_dist_ctx);
  }

  /**
   * @brief Compute the Jacobian of the gradient of the Lagrangian wrt the input
   * data
   *
   * @param x The design variable vector
   * @param jac The Jacobian of the gradient wrt the data
   */
  void gradient_jacobian_wrt_data(const std::shared_ptr<Vector<T>> x,
                                  std::shared_ptr<CSRMat<T>> jac) {
    var_dist.begin_forward(x, var_ctx);
    var_dist.end_forward(x, var_ctx);

    jac->zero();
    for (size_t i = 0; i < components.size(); i++) {
      components[i]->add_grad_jac_wrt_data(*data_vec, *x, *data_owners, *jac);
    }

    grad_jac_dist->begin_assembly(jac, grad_jac_dist_ctx);
    grad_jac_dist->end_assembly(jac, grad_jac_dist_ctx);
  }

  /**
   * @brief Create a CSR matrix for the Jacobian of the gradient of the
   * Lagrangian wrt the data
   */
  std::shared_ptr<CSRMat<T>> create_gradient_jacobian_wrt_data() {
    if (grad_jac) {
      return grad_jac->duplicate();
    } else {
      std::vector<int> intervals(components.size() + 1);
      intervals[0] = 0;
      for (size_t i = 0; i < components.size(); i++) {
        int num_elems, inputs_per_elem;
        const int* inputs;
        components[i]->get_layout_data(&num_elems, &inputs_per_elem, &inputs);
        intervals[i + 1] = intervals[i] + num_elems;
      }

      auto element_output = [&](int element, int* nrow, int* ncol,
                                const int** rows, const int** cols) {
        // upper_bound finds the first index i such that intervals[i] >
        // element
        auto it = std::upper_bound(intervals.begin(), intervals.end(), element);

        // Decrement to get the interval where element fits: intervals[idx]
        // <= element < intervals[idx+1]
        int idx = static_cast<int>(it - intervals.begin()) - 1;

        int num_elems;
        const int *data, *inputs;
        components[idx]->get_layout_data(&num_elems, nrow, &inputs);
        components[idx]->get_data_layout_data(&num_elems, ncol, &data);

        int elem = element - intervals[idx];
        *rows = &inputs[(*nrow) * elem];
        *cols = &data[(*ncol) * elem];
      };

      int num_variables =
          var_owners->get_local_size() + var_owners->get_ext_size();

      int num_data =
          data_owners->get_local_size() + data_owners->get_ext_size();

      // Generate the non-zero pattern
      int *rowp, *cols;
      OrderingUtils::create_csr_from_output_data(num_variables, num_data,
                                                 intervals[components.size()],
                                                 element_output, &rowp, &cols);

      // Distribute the pattern across matrices
      MemoryLocation mem_loc = MemoryLocation::HOST_AND_DEVICE;
      grad_jac_dist = new MatrixDistribute<policy>(
          comm, mem_loc, var_owners, data_owners, num_variables, num_data, rowp,
          cols, grad_jac);
      grad_jac_dist_ctx = grad_jac_dist->template create_context<T>();

      delete[] rowp;
      delete[] cols;

      return grad_jac;
    }
  }

  /**
   * @brief Compute the output as a function of the inputs
   *
   * @param x The design variable vector
   * @param outputs The vector of outputs
   */
  void compute_output(const std::shared_ptr<Vector<T>> x,
                      std::shared_ptr<Vector<T>> outputs) {
    var_dist.begin_forward(x, var_ctx);
    var_dist.end_forward(x, var_ctx);

    outputs->zero();
    for (size_t i = 0; i < components.size(); i++) {
      components[i]->add_output(*data_vec, *x, *outputs);
    }

    output_dist.begin_reverse_add(outputs, output_ctx);
    output_dist.end_reverse_add(outputs, output_ctx);
  }

  /**
   * @brief Compute the Jacobian of the outputs wrt inputs
   *
   * @param x The design variable vector
   * @param jacobian The Jacobian matrix
   */
  void output_jacobian_wrt_input(const std::shared_ptr<Vector<T>> x,
                                 std::shared_ptr<CSRMat<T>> jacobian) {
    var_dist.begin_forward(x, var_ctx);
    var_dist.end_forward(x, var_ctx);

    jacobian->zero();
    for (size_t i = 0; i < components.size(); i++) {
      components[i]->add_output_jac_wrt_input(*data_vec, *x, *jacobian);
    }

    input_jac_dist->begin_assembly(jacobian, input_jac_dist_ctx);
    input_jac_dist->end_assembly(jacobian, input_jac_dist_ctx);
  }

  /**
   * @brief Compute the Jacobian of the outputs wrt data
   *
   * @param x The design variable vector
   * @param jacobian The Jacobian matrix
   */
  void output_jacobian_wrt_data(const std::shared_ptr<Vector<T>> x,
                                std::shared_ptr<CSRMat<T>> jacobian) {
    var_dist.begin_forward(x, var_ctx);
    var_dist.end_forward(x, var_ctx);

    jacobian->zero();
    for (size_t i = 0; i < components.size(); i++) {
      components[i]->add_output_jac_wrt_data(*data_vec, *x, *jacobian);
    }

    data_jac_dist->begin_assembly(jacobian, data_jac_dist_ctx);
    data_jac_dist->end_assembly(jacobian, data_jac_dist_ctx);
  }

  /**
   * @brief Create an instance of the input Jacobian matrix
   *
   * @return std::shared_ptr<CSRMat<T>>
   */
  std::shared_ptr<CSRMat<T>> create_output_jacobian_wrt_input() {
    if (input_jac) {
      return input_jac->duplicate();
    } else {
      std::vector<int> intervals(components.size() + 1);
      intervals[0] = 0;
      for (size_t i = 0; i < components.size(); i++) {
        int num_elems, outputs_per_elem;
        const int* outputs;
        components[i]->get_output_layout_data(&num_elems, &outputs_per_elem,
                                              &outputs);
        intervals[i + 1] = intervals[i] + num_elems;
      }

      auto element_output = [&](int element, int* nrow, int* ncol,
                                const int** rows, const int** cols) {
        // upper_bound finds the first index i such that intervals[i] >
        // element
        auto it = std::upper_bound(intervals.begin(), intervals.end(), element);

        // Decrement to get the interval where element fits: intervals[idx]
        // <= element < intervals[idx+1]
        int idx = static_cast<int>(it - intervals.begin()) - 1;

        int num_elems;
        const int *outputs, *inputs;
        components[idx]->get_output_layout_data(&num_elems, nrow, &outputs);
        components[idx]->get_layout_data(&num_elems, ncol, &inputs);

        int elem = element - intervals[idx];
        *rows = &outputs[(*nrow) * elem];
        *cols = &inputs[(*ncol) * elem];
      };

      int num_outputs =
          output_owners->get_local_size() + output_owners->get_ext_size();

      int num_variables =
          var_owners->get_local_size() + var_owners->get_ext_size();

      // Generate the non-zero pattern
      int *rowp, *cols;
      OrderingUtils::create_csr_from_output_data(num_outputs, num_variables,
                                                 intervals[components.size()],
                                                 element_output, &rowp, &cols);

      // Distribute the pattern across matrices
      MemoryLocation mem_loc = MemoryLocation::HOST_AND_DEVICE;
      input_jac_dist = new MatrixDistribute<policy>(
          comm, mem_loc, output_owners, var_owners, num_outputs, num_variables,
          rowp, cols, input_jac);
      input_jac_dist_ctx = input_jac_dist->template create_context<T>();

      delete[] rowp;
      delete[] cols;

      return input_jac;
    }
  }

  /**
   * @brief Create an instance of the input Jacobian matrix
   *
   * @return std::shared_ptr<CSRMat<T>>
   */
  std::shared_ptr<CSRMat<T>> create_output_jacobian_wrt_data() {
    if (data_jac) {
      return data_jac->duplicate();
    } else {
      std::vector<int> intervals(components.size() + 1);
      intervals[0] = 0;
      for (size_t i = 0; i < components.size(); i++) {
        int num_elems, outputs_per_elem;
        const int* outputs;
        components[i]->get_output_layout_data(&num_elems, &outputs_per_elem,
                                              &outputs);
        intervals[i + 1] = intervals[i] + num_elems;
      }

      auto element_output = [&](int element, int* nrow, int* ncol,
                                const int** rows, const int** cols) {
        // upper_bound finds the first index i such that intervals[i] >
        // element
        auto it = std::upper_bound(intervals.begin(), intervals.end(), element);

        // Decrement to get the interval where element fits: intervals[idx]
        // <= element < intervals[idx+1]
        int idx = static_cast<int>(it - intervals.begin()) - 1;

        int num_elems;
        const int *outputs, *data;
        components[idx]->get_output_layout_data(&num_elems, nrow, &outputs);
        components[idx]->get_data_layout_data(&num_elems, ncol, &data);

        int elem = element - intervals[idx];
        *rows = &outputs[(*nrow) * elem];
        *cols = &data[(*ncol) * elem];
      };

      int num_outputs =
          output_owners->get_local_size() + output_owners->get_ext_size();

      int num_data =
          data_owners->get_local_size() + data_owners->get_ext_size();

      // Generate the non-zero pattern
      int *rowp, *cols;
      OrderingUtils::create_csr_from_output_data(num_outputs, num_data,
                                                 intervals[components.size()],
                                                 element_output, &rowp, &cols);

      // Distribute the pattern across matrices
      MemoryLocation mem_loc = MemoryLocation::HOST_AND_DEVICE;
      data_jac_dist = new MatrixDistribute<policy>(
          comm, mem_loc, output_owners, data_owners, num_outputs, num_data,
          rowp, cols, data_jac);
      data_jac_dist_ctx = data_jac_dist->template create_context<T>();

      delete[] rowp;
      delete[] cols;

      return data_jac;
    }
  }

 private:
  /**
   * @brief Create a functor that returns the number of nodes and node
   * numbers, given an element index
   *
   * @param intervals Data structure that stores the intervals for the elements
   * @return The functor returning number of nodes per element and element nodes
   */
  auto get_element_nodes(std::vector<int>& intervals) const {
    intervals.resize(components.size() + 1);
    intervals[0] = 0;
    for (size_t i = 0; i < components.size(); i++) {
      int num_elems, nnodes_per_elem;
      const int* data;
      components[i]->get_layout_data(&num_elems, &nnodes_per_elem, &data);
      intervals[i + 1] = intervals[i] + num_elems;
    }

    auto element_nodes = [&](int element, const int** ptr) {
      // upper_bound finds the first index i such that intervals[i] >
      // element
      auto it = std::upper_bound(intervals.begin(), intervals.end(), element);

      // Decrement to get the interval where element fits: intervals[idx]
      // <= element < intervals[idx+1]
      int idx = static_cast<int>(it - intervals.begin()) - 1;

      int num_elems, nnodes_per_elem;
      const int* data;
      components[idx]->get_layout_data(&num_elems, &nnodes_per_elem, &data);

      int elem = element - intervals[idx];
      if (nnodes_per_elem > 0) {
        *ptr = &data[nnodes_per_elem * elem];
      } else {
        *ptr = nullptr;
      }
      return nnodes_per_elem;
    };

    return element_nodes;
  }

  /**
   * @brief Create a functor that returns the number of nodes and node numbers
   * for the data, given an element index
   *
   * @param intervals Data structure that stores the intervals for the elements
   * @return The functor returning the data per element and data indices
   */
  auto get_element_data(std::vector<int>& intervals) const {
    intervals.resize(components.size() + 1);
    intervals[0] = 0;
    for (size_t i = 0; i < components.size(); i++) {
      int num_elems, ndata_per_elem;
      const int* data;
      components[i]->get_data_layout_data(&num_elems, &ndata_per_elem, &data);
      intervals[i + 1] = intervals[i] + num_elems;
    }

    auto element_nodes = [&](int element, const int** ptr) {
      // upper_bound finds the first index i such that intervals[i] >
      // element
      auto it = std::upper_bound(intervals.begin(), intervals.end(), element);

      // Decrement to get the interval where element fits: intervals[idx]
      // <= element < intervals[idx+1]
      int idx = static_cast<int>(it - intervals.begin()) - 1;

      int num_elems, ndata_per_elem;
      const int* data;
      components[idx]->get_data_layout_data(&num_elems, &ndata_per_elem, &data);

      int elem = element - intervals[idx];
      if (ndata_per_elem > 0) {
        *ptr = &data[ndata_per_elem * elem];
      } else {
        *ptr = nullptr;
      }
      return ndata_per_elem;
    };

    return element_nodes;
  }

  /**
   * @brief Create a functor that returns the number of outputs and output
   * numbers given an element index
   *
   * @param intervals Data structure that stores the intervals for the elements
   * @return The functor returning the outputs per element and output indices
   */
  auto get_element_output(std::vector<int>& intervals) const {
    intervals.resize(components.size() + 1);
    intervals[0] = 0;
    for (size_t i = 0; i < components.size(); i++) {
      int num_elems, ouputs_per_elem;
      const int* outputs;
      components[i]->get_output_layout_data(&num_elems, &ouputs_per_elem,
                                            &outputs);
      intervals[i + 1] = intervals[i] + num_elems;
    }

    auto element_outputs = [&](int element, const int** ptr) {
      // upper_bound finds the first index i such that intervals[i] >
      // element
      auto it = std::upper_bound(intervals.begin(), intervals.end(), element);

      // Decrement to get the interval where element fits: intervals[idx]
      // <= element < intervals[idx+1]
      int idx = static_cast<int>(it - intervals.begin()) - 1;

      int num_elems, ouputs_per_elem;
      const int* outputs;
      components[idx]->get_output_layout_data(&num_elems, &ouputs_per_elem,
                                              &outputs);

      int elem = element - intervals[idx];
      if (ouputs_per_elem > 0) {
        *ptr = &outputs[ouputs_per_elem * elem];
      } else {
        *ptr = nullptr;
      }
      return ouputs_per_elem;
    };

    return element_outputs;
  }

  /**
   * @brief Compute a reordering from the global node numbers to a local
   * ordering
   *
   * If the node numbers fall in the range:
   *
   * lower <= node < upper
   *
   * Then the node ordered locally as
   *
   * local = node - lower
   *
   * Otherwise a unique, sorted list of external nodes (ext_nodes) is computed
   * so that
   *
   * node = ext_nodes[i]
   *
   * and
   *
   * local = (upper - lower) + i
   *
   * @param lower Lower global node index
   * @param upper Upper global node index
   * @param nodes Array of vectors for the local components
   * @param ext_nodes External nodes
   */
  void reorder_from_global_to_local(
      int lower, int upper, std::vector<std::shared_ptr<Vector<int>>>& nodes,
      std::vector<int>& ext_nodes) {
    // Estimate an upper bound for the size here - maybe there's a better
    // approach for this...
    ext_nodes.reserve(4 * (upper - lower));

    // Identify nodes that
    for (size_t i = 0; i < nodes.size(); i++) {
      int* array = nodes[i]->get_array();
      int size = nodes[i]->get_size();

      for (int j = 0; j < size; j++) {
        // Check if this is an external node
        if (array[j] < lower || array[j] >= upper) {
          ext_nodes.push_back(array[j]);
        }
      }
    }

    // Sort the array and remove duplicates
    std::sort(ext_nodes.begin(), ext_nodes.end());
    ext_nodes.erase(std::unique(ext_nodes.begin(), ext_nodes.end()),
                    ext_nodes.end());

    // Set the node numbers based on the local ordering
    for (size_t i = 0; i < nodes.size(); i++) {
      int* array = nodes[i]->get_array();
      int size = nodes[i]->get_size();

      for (int j = 0; j < size; j++) {
        // Check if this is an external node
        if (array[j] >= lower && array[j] < upper) {
          array[j] = array[j] - lower;
        } else {
          auto it =
              std::lower_bound(ext_nodes.begin(), ext_nodes.end(), array[j]);
          if (it != ext_nodes.end() && *it == array[j]) {
            array[j] = (upper - lower) + (it - ext_nodes.begin());
          }
        }
      }
    }
  }

  // The MPI communicator
  MPI_Comm comm;

  // Node owner data for the data and variables
  std::shared_ptr<NodeOwners> data_owners;
  std::shared_ptr<NodeOwners> var_owners;
  std::shared_ptr<NodeOwners> output_owners;

  // Array indicating which variables are multipliers
  std::shared_ptr<Vector<int>> is_multiplier;

  // Component groups for the optimization problem
  std::vector<std::shared_ptr<ComponentGroupBase<T, policy>>> components;

  // Variable information
  VectorDistribute<policy> data_dist;
  VectorDistribute<policy> var_dist;
  typename VectorDistribute<policy>::template VecDistributeContext<T>* data_ctx;
  typename VectorDistribute<policy>::template VecDistributeContext<T>* var_ctx;

  // Output information
  VectorDistribute<policy> output_dist;
  typename VectorDistribute<policy>::template VecDistributeContext<T>*
      output_ctx;

  // The shared data vector
  std::shared_ptr<Vector<T>> data_vec;

  // Node numbers created by
  std::shared_ptr<Vector<int>> dist_node_numbers;
  std::shared_ptr<Vector<int>> dist_data_numbers;
  std::shared_ptr<Vector<int>> dist_output_numbers;

  // Information about the Hessian matrix
  std::shared_ptr<CSRMat<T>> mat;
  MatrixDistribute<policy>* mat_dist;
  typename MatrixDistribute<policy>::template MatDistributeContext<T>*
      mat_dist_ctx;

  // Information about the output Jacobian
  std::shared_ptr<CSRMat<T>> input_jac;
  MatrixDistribute<policy>* input_jac_dist;
  typename MatrixDistribute<policy>::template MatDistributeContext<T>*
      input_jac_dist_ctx;

  std::shared_ptr<CSRMat<T>> data_jac;
  MatrixDistribute<policy>* data_jac_dist;
  typename MatrixDistribute<policy>::template MatDistributeContext<T>*
      data_jac_dist_ctx;

  std::shared_ptr<CSRMat<T>> grad_jac;
  MatrixDistribute<policy>* grad_jac_dist;
  typename MatrixDistribute<policy>::template MatDistributeContext<T>*
      grad_jac_dist_ctx;
};

}  // namespace amigo

#endif  // AMIGO_OPTIMIZATION_PROBLEM_H