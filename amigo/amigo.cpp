#include <mpi4py/mpi4py.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#ifdef _WIN32
#include <BaseTsd.h>
typedef SSIZE_T ssize_t;
#endif

#include "alias_tracker.h"
#include "amigo_include_paths.h"
#include "csr_matrix.h"
#include "optimization_problem.h"
#include "optimizer.h"
#include "sparse_cholesky.h"

namespace py = pybind11;

// Templated wrapper function
template <typename T>
void bind_vector(py::module_ &m, const std::string &name) {
  py::class_<amigo::Vector<T>, std::shared_ptr<amigo::Vector<T>>>(m,
                                                                  name.c_str())
      .def(py::init<int>())
      .def("zero", &amigo::Vector<T>::zero)
      .def("axpy", &amigo::Vector<T>::axpy)
      .def("scale", &amigo::Vector<T>::scale)
      .def("get_size", &amigo::Vector<T>::get_size)
      .def("__getitem__",
           [](const amigo::Vector<T> &v, py::object index) -> py::object {
             if (py::isinstance<py::int_>(index)) {
               ssize_t i = index.cast<ssize_t>();
               if (i < 0) i += v.get_size();
               if (i < 0 || i >= static_cast<ssize_t>(v.get_size())) {
                 throw py::index_error();
               }
               return py::cast(v[i]);
             } else if (py::isinstance<py::slice>(index)) {
               py::slice slice = index.cast<py::slice>();
               size_t start, stop, step, slicelength;
               if (!slice.compute(v.get_size(), &start, &stop, &step,
                                  &slicelength)) {
                 throw py::error_already_set();
               }

               std::vector<T> result;
               for (size_t i = 0; i < slicelength; i++) {
                 result.push_back(v[start + i * step]);
               }

               return py::cast(result);
             } else {
               throw py::type_error("Invalid index type");
             }
           })
      .def("__setitem__",
           [](amigo::Vector<T> &v, py::object index, py::object value) {
             if (py::isinstance<py::int_>(index)) {
               ssize_t i = index.cast<ssize_t>();
               if (i < 0) i += v.get_size();
               if (i < 0 || i >= static_cast<ssize_t>(v.get_size())) {
                 throw py::index_error();
               }
               v[i] = value.cast<T>();
             } else if (py::isinstance<py::slice>(index)) {
               py::slice slice = index.cast<py::slice>();
               size_t start, stop, step, slicelength;
               if (!slice.compute(v.get_size(), &start, &stop, &step,
                                  &slicelength)) {
                 throw py::error_already_set();
               }

               std::vector<T> values = value.cast<std::vector<T>>();
               if (values.size() != slicelength) {
                 throw std::runtime_error("Slice assignment size mismatch");
               }

               for (size_t i = 0; i < slicelength; i++) {
                 v[start + i * step] = values[i];
               }
             } else {
               throw py::type_error("Invalid index type");
             }
           })
      .def("__len__", &amigo::Vector<T>::get_size)
      .def("get_array",
           [](std::shared_ptr<amigo::Vector<T>> self) -> py::array_t<T> {
             return py::array_t<T>({self->get_size()}, {sizeof(T)},
                                   self->get_array(), py::cast(self));
           });
}

py::array_t<int> reorder_model(amigo::OrderingType order_type,
                               std::vector<py::array_t<int>> arrays,
                               py::object output_vars = py::none()) {
  std::vector<int> intervals(arrays.size() + 1);
  intervals[0] = 0;

  int max_node = 0;

  // Check the dimension of the arrays
  ssize_t max_columns = 0;
  for (size_t i = 0; i < arrays.size(); i++) {
    const auto &arr = arrays[i];
    if (arr.ndim() != 2) {
      throw std::runtime_error("Each input array must be 2D");
    }

    auto r = arr.unchecked<2>();
    intervals[i + 1] = intervals[i] + r.shape(0);
    int ncols = r.shape(1);

    if (ncols > max_columns) {
      max_columns = ncols;
    }

    for (int i = 0; i < r.shape(0); i++) {
      for (int j = 0; j < r.shape(1); j++) {
        if (r(i, j) > max_node) {
          max_node = r(i, j);
        }
      }
    }
  }
  max_node++;

  int nrows = max_node, ncols = max_node;
  int nelems = intervals[arrays.size()];

  std::vector<int> columns(max_columns);

  auto element_nodes = [&](int element, const int **ptr) {
    // upper_bound finds the first index i such that intervals[i] >
    // element
    auto it = std::upper_bound(intervals.begin(), intervals.end(), element);

    // Decrement to get the interval where element fits: intervals[idx]
    // <= element < intervals[idx+1]
    int idx = static_cast<int>(it - intervals.begin()) - 1;

    int elem = element - intervals[idx];

    auto r = arrays[idx].unchecked<2>();
    int ncols = r.shape(1);
    for (int j = 0; j < ncols; j++) {
      columns[j] = r(elem, j);
    }

    *ptr = columns.data();
    return ncols;
  };

  // Make a CSR structure
  bool include_diagonal = false;
  bool sort_columns = true;

  int *rowp, *cols;
  amigo::OrderingUtils::create_csr_from_element_conn(
      nrows, ncols, nelems, element_nodes, include_diagonal, sort_columns,
      &rowp, &cols);

  // Compute the reordering
  int *perm, *iperm;
  if (!output_vars.is_none()) {
    auto output_array = output_vars.cast<py::array_t<int>>();
    auto outputs_ = output_array.unchecked<1>();

    int num_outputs = outputs_.shape(0);
    int *outputs = new int[num_outputs];
    for (int i = 0; i < num_outputs; i++) {
      outputs[i] = outputs_[i];
    }

    amigo::OrderingUtils::reorder_block(order_type, nrows, rowp, cols,
                                        num_outputs, outputs, &perm, &iperm);

    delete[] outputs;
  } else {
    amigo::OrderingUtils::reorder(order_type, nrows, rowp, cols, &perm, &iperm);
  }

  delete[] rowp;
  delete[] cols;

  // Allocate the new partition
  py::array_t<int> iperm_output(nrows, iperm);

  delete[] perm;
  delete[] iperm;

  return iperm_output;
}

PYBIND11_MODULE(amigo, mod) {
  mod.doc() = "Amigo: A friendly library for MDO on HPC";

  // Import mpi4py
  import_mpi4py();

  mod.attr("A2D_INCLUDE_PATH") = A2D_INCLUDE_PATH;
  mod.attr("AMIGO_INCLUDE_PATH") = AMIGO_INCLUDE_PATH;

  py::enum_<amigo::OrderingType>(mod, "OrderingType")
      .value("NESTED_DISSECTION", amigo::OrderingType::NESTED_DISSECTION)
      .value("AMD", amigo::OrderingType::AMD)
      .value("MULTI_COLOR", amigo::OrderingType::MULTI_COLOR)
      .value("NATURAL", amigo::OrderingType::NATURAL)
      .export_values();

  mod.def("reorder_model", &reorder_model, py::arg("order_type"),
          py::arg("arrays"), py::arg("output_indices") = py::none());

  py::class_<amigo::CSRMat<double>, std::shared_ptr<amigo::CSRMat<double>>>(
      mod, "CSRMat")
      .def("get_nonzero_structure",
           [](amigo::CSRMat<double> &mat) {
             int nrows, ncols, nnz;
             const int *mat_rowp, *mat_cols;
             mat.get_data(&nrows, &ncols, &nnz, &mat_rowp, &mat_cols, nullptr);

             py::array_t<int> rowp(nrows + 1);
             std::memcpy(rowp.mutable_data(), mat_rowp,
                         (nrows + 1) * sizeof(int));

             py::array_t<int> cols(nnz);
             std::memcpy(cols.mutable_data(), mat_cols, nnz * sizeof(int));
             return py::make_tuple(nrows, ncols, nnz, rowp, cols);
           })
      .def("get_data",
           [](amigo::CSRMat<double> &mat) -> py::array_t<double> {
             int nnz;
             double *mat_data;
             mat.get_data(nullptr, nullptr, &nnz, nullptr, nullptr, &mat_data);

             py::array_t<double> data(nnz);
             std::memcpy(data.mutable_data(), mat_data, nnz * sizeof(double));
             return data;
           })
      .def("extract_submatrix",
           [](const amigo::CSRMat<double> &self, py::array_t<int> rows,
              py::array_t<int> cols) {
             return self.extract_submatrix(rows.size(), rows.data(),
                                           cols.size(), cols.data());
           })
      .def("extract_submatrix_values",
           [](const amigo::CSRMat<double> &self, py::array_t<int> rows,
              py::array_t<int> cols,
              std::shared_ptr<amigo::CSRMat<double>> mat) {
             self.extract_submatrix_values(rows.size(), rows.data(),
                                           cols.size(), cols.data(), mat);
           })
      .def("get_row_owners", &amigo::CSRMat<double>::get_row_owners)
      .def("get_column_owners", &amigo::CSRMat<double>::get_column_owners)
      .def("gauss_seidel", &amigo::CSRMat<double>::gauss_seidel)
      .def("mult", &amigo::CSRMat<double>::mult)
      .def("add_diagonal", &amigo::CSRMat<double>::add_diagonal);

  bind_vector<int>(mod, "VectorInt");
  bind_vector<double>(mod, "Vector");

  py::class_<amigo::ComponentGroupBase<double>,
             std::shared_ptr<amigo::ComponentGroupBase<double>>>(
      mod, "ComponentGroupBase");
  py::class_<amigo::OutputGroupBase<double>,
             std::shared_ptr<amigo::OutputGroupBase<double>>>(
      mod, "OutputGroupBase");

  py::class_<amigo::NodeOwners, std::shared_ptr<amigo::NodeOwners>>(
      mod, "NodeOwners")
      .def(py::init([](py::object pyobj, py::array_t<int> ranges) {
        int size = 1;
        MPI_Comm comm = MPI_COMM_SELF;
        if (!pyobj.is_none()) {
          comm = *PyMPIComm_Get(pyobj.ptr());
          MPI_Comm_size(comm, &size);
        }
        if (ranges.size() != size + 1) {
          throw std::runtime_error(
              "Ranges must be of length MPI_Comm_size + 1");
        }
        return std::make_shared<amigo::NodeOwners>(comm, ranges.data());
      }))
      .def("get_mpi_comm",
           [](const amigo::NodeOwners &self) {
             return py::reinterpret_steal<py::object>(
                 PyMPIComm_New(self.get_mpi_comm()));
           })
      .def("get_local_size", &amigo::NodeOwners::get_local_size);

  py::class_<amigo::OptimizationProblem<double>,
             std::shared_ptr<amigo::OptimizationProblem<double>>>(
      mod, "OptimizationProblem")
      .def(py::init([](py::object pyobj,
                       std::shared_ptr<amigo::NodeOwners> data_owners,
                       std::shared_ptr<amigo::NodeOwners> var_owners,
                       std::shared_ptr<amigo::Vector<int>> is_multiplier,
                       const std::vector<std::shared_ptr<
                           amigo::ComponentGroupBase<double>>> &components) {
        MPI_Comm comm = MPI_COMM_SELF;
        if (!pyobj.is_none()) {
          comm = *PyMPIComm_Get(pyobj.ptr());
        }
        return std::make_shared<amigo::OptimizationProblem<double>>(
            comm, data_owners, var_owners, is_multiplier, components);
      }))
      .def("get_num_variables",
           &amigo::OptimizationProblem<double>::get_num_variables)
      .def("partition_from_root",
           &amigo::OptimizationProblem<double>::partition_from_root,
           py::arg("root") = 0)
      .def("create_vector", &amigo::OptimizationProblem<double>::create_vector)
      .def("create_data_vector",
           &amigo::OptimizationProblem<double>::create_data_vector)
      .def("get_data_vector",
           &amigo::OptimizationProblem<double>::get_data_vector)
      .def("set_data_vector",
           &amigo::OptimizationProblem<double>::set_data_vector)
      .def(
          "get_local_to_global_node_numbers",
          &amigo::OptimizationProblem<double>::get_local_to_global_node_numbers)
      .def(
          "get_local_to_global_data_numbers",
          &amigo::OptimizationProblem<double>::get_local_to_global_data_numbers)
      .def("lagrangian", &amigo::OptimizationProblem<double>::lagrangian)
      .def("gradient", &amigo::OptimizationProblem<double>::gradient)
      .def("create_matrix", &amigo::OptimizationProblem<double>::create_matrix)
      .def("hessian", &amigo::OptimizationProblem<double>::hessian)
      .def("scatter_vector",
           &amigo::OptimizationProblem<double>::scatter_vector<double>,
           py::arg("root_vec"), py::arg("dist_problem"), py::arg("dist_vec"),
           py::arg("root") = 0, py::arg("distribute") = true)
      .def("gather_vector",
           &amigo::OptimizationProblem<double>::gather_vector<double>,
           py::arg("dist_problem"), py::arg("dist_vec"), py::arg("root_vec"),
           py::arg("root") = 0)
      .def("scatter_data_vector",
           &amigo::OptimizationProblem<double>::scatter_data_vector<double>,
           py::arg("root_vec"), py::arg("dist_problem"), py::arg("dist_vec"),
           py::arg("root") = 0, py::arg("distribute") = true);
  // .def("create_output_vector",
  //      &amigo::OptimizationProblem<double>::create_output_vector)
  // .def("analyze", &amigo::OptimizationProblem<double>::analyze)
  // .def("create_output_csr_matrix",
  //      &amigo::OptimizationProblem<double>::create_output_csr_matrix)
  // .def("analyze_jacobian",
  //      &amigo::OptimizationProblem<double>::analyze_jacobian);

  py::class_<amigo::AliasTracker<int>>(mod, "AliasTracker")
      .def(py::init<int>(), py::arg("size"))
      .def("alias", &amigo::AliasTracker<int>::alias, py::arg("var1"),
           py::arg("var2"))
      .def("get_alias_group", &amigo::AliasTracker<int>::get_alias_group,
           py::arg("var"))
      .def("assign_group_vars", [](amigo::AliasTracker<int> tracker) {
        py::array_t<int> array(tracker.size());
        int counter = tracker.assign_group_vars(array.mutable_data());
        return py::make_tuple(counter, array);
      });

  py::class_<amigo::SparseCholesky<double>,
             std::shared_ptr<amigo::SparseCholesky<double>>>(mod,
                                                             "SparseCholesky")
      .def(py::init<std::shared_ptr<amigo::CSRMat<double>>>())
      .def("factor", &amigo::SparseCholesky<double>::factor)
      .def("solve", &amigo::SparseCholesky<double>::solve);

  py::class_<amigo::OptVector<double>,
             std::shared_ptr<amigo::OptVector<double>>>(mod, "OptVector")
      .def_readwrite("x", &amigo::OptVector<double>::x)
      .def_readwrite("xs", &amigo::OptVector<double>::xs)
      .def_readwrite("zl", &amigo::OptVector<double>::zl)
      .def_readwrite("zu", &amigo::OptVector<double>::zu)
      .def("copy", &amigo::OptVector<double>::copy);

  py::class_<amigo::InteriorPointOptimizer<double>,
             std::shared_ptr<amigo::InteriorPointOptimizer<double>>>(
      mod, "InteriorPointOptimizer")
      .def(py::init<std::shared_ptr<amigo::OptimizationProblem<double>>,
                    std::shared_ptr<amigo::Vector<double>>,
                    std::shared_ptr<amigo::Vector<double>>>())
      .def(
          "create_opt_vector",
          [](const amigo::InteriorPointOptimizer<double> &self,
             py::object x = py::none()) {
            if (!x.is_none()) {
              return self.create_opt_vector(
                  x.cast<std::shared_ptr<amigo::Vector<double>>>());
            }
            return self.create_opt_vector();
          },
          py::arg("x") = py::none())
      .def("initialize_multipliers_and_slacks",
           &amigo::InteriorPointOptimizer<
               double>::initialize_multipliers_and_slacks)
      .def("make_vars_consistent",
           &amigo::InteriorPointOptimizer<double>::make_vars_consistent)
      .def("compute_residual",
           &amigo::InteriorPointOptimizer<double>::compute_residual)
      .def("compute_reduced_residual",
           &amigo::InteriorPointOptimizer<double>::compute_reduced_residual)
      .def("compute_update_from_reduced",
           &amigo::InteriorPointOptimizer<double>::compute_update_from_reduced)
      .def("compute_diagonal",
           &amigo::InteriorPointOptimizer<double>::compute_diagonal)
      .def("compute_max_step",
           [](const amigo::InteriorPointOptimizer<double> &self,
              const double tau,
              const std::shared_ptr<amigo::OptVector<double>> vars,
              const std::shared_ptr<amigo::OptVector<double>> update) {
             double alpha_x, alpha_z;
             self.compute_max_step(tau, vars, update, alpha_x, alpha_z);
             return py::make_tuple(alpha_x, alpha_z);
           })
      .def("apply_step_update",
           &amigo::InteriorPointOptimizer<double>::apply_step_update)
      .def("check_update",
           &amigo::InteriorPointOptimizer<double>::check_update);
}
