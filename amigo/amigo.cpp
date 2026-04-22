// Force rebuild after header changes
#include <mpi4py/mpi4py.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#ifdef _WIN32
#include <BaseTsd.h>
typedef SSIZE_T ssize_t;
#endif

#include "alias_tracker.h"
#include "csr_matrix.h"
#include "external_component.h"
#include "interior_point_optimizer.h"
#include "optimization_problem.h"
#include "slack_coupling.h"
#include "sparse_cholesky.h"
#include "sparse_ldl.h"

#ifdef AMIGO_USE_CUDA
#include "cuda/csr_factor_cuda.h"
#endif

namespace py = pybind11;

template <typename T>
class PyExternalCallback
    : public amigo::ExternalComponentEvaluation<T>,
      public std::enable_shared_from_this<PyExternalCallback<T>> {
 public:
  PyExternalCallback(int nvars, int ncon, const int rowp[], const int cols[],
                     py::object cb)
      : amigo::ExternalComponentEvaluation<T>(nvars, ncon, rowp, cols) {
    // Set the callback
    callback = std::move(cb);
  }

  void evaluate() {
    if (!callback.is_none()) {
      py::gil_scoped_acquire gil;
      // py::object base = py::cast(this->shared_from_this());

      // Wrap the design variable and multiplier values
      const std::shared_ptr<amigo::Vector<T>> x = this->get_variables();
      const T* x_array = x->get_array();
      int x_size = x->get_size();
      py::array_t<const T> xarr({x_size}, {ssize_t(sizeof(T))}, x_array,
                                py::cast(x));

      // Wrap the constraint vector
      std::shared_ptr<amigo::Vector<T>> constraints = this->get_constraints();
      T* con_array = constraints->get_array();
      int con_size = constraints->get_size();
      py::array_t<T> con({con_size}, {ssize_t(sizeof(T))}, con_array,
                         py::cast(constraints));

      // Wrap the objective gradient
      std::shared_ptr<amigo::Vector<T>> grad = this->get_objective_gradient();
      T* grad_array = grad->get_array();
      int grad_size = grad->get_size();
      py::array_t<T> g({grad_size}, {ssize_t(sizeof(T))}, grad_array,
                       py::cast(grad));

      // Wrap the constraint Jacobian
      std::shared_ptr<amigo::CSRMat<T>> jacobian = this->get_jacobian();
      int nnz;
      T* data_array;
      jacobian->get_data(nullptr, nullptr, &nnz, nullptr, nullptr, &data_array);
      py::array_t<T> data({nnz}, {ssize_t(sizeof(T))}, data_array,
                          py::cast(jacobian));

      py::object out = callback(xarr, con, g, data);

      if (!out.is_none()) {
        this->get_objective() = py::cast<T>(out);
      }
    }
  }

 private:
  py::object callback;
};

// Templated wrapper function
template <typename T>
void bind_vector(py::module_& m, const std::string& name) {
  py::class_<amigo::Vector<T>, std::shared_ptr<amigo::Vector<T>>>(m,
                                                                  name.c_str())
      .def(py::init<int>())
      .def("zero", &amigo::Vector<T>::zero)
      .def("get_size", &amigo::Vector<T>::get_size)
      .def("copy", [](amigo::Vector<T>& self,
                      amigo::Vector<T>& src) { self.copy(src); })
      .def("copy_host_to_device", &amigo::Vector<T>::copy_host_to_device)
      .def("copy_device_to_host", &amigo::Vector<T>::copy_device_to_host)
      .def("__getitem__",
           [](const amigo::Vector<T>& v, py::object index) -> py::object {
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
           [](amigo::Vector<T>& v, py::object index, py::object value) {
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
    const auto& arr = arrays[i];
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

  auto element_nodes = [&](int element, const int** ptr) {
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
    int* outputs = new int[num_outputs];
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

namespace detail {
#ifdef AMIGO_USE_OPENMP
inline constexpr amigo::ExecPolicy policy = amigo::ExecPolicy::OPENMP;
#elif defined(AMIGO_USE_CUDA)
inline constexpr amigo::ExecPolicy policy = amigo::ExecPolicy::CUDA;
#else
inline constexpr amigo::ExecPolicy policy = amigo::ExecPolicy::SERIAL;
#endif
}  // namespace detail

PYBIND11_MODULE(amigo, mod) {
  mod.doc() = "Amigo: A friendly library for MDO on HPC";

  // Import mpi4py
  if (import_mpi4py() < 0) {
    throw pybind11::error_already_set();
  }

  py::enum_<amigo::OrderingType>(mod, "OrderingType")
      .value("NESTED_DISSECTION", amigo::OrderingType::NESTED_DISSECTION)
      .value("AMD", amigo::OrderingType::AMD)
      .value("MULTI_COLOR", amigo::OrderingType::MULTI_COLOR)
      .value("NATURAL", amigo::OrderingType::NATURAL)
#ifdef AMIGO_USE_METIS
      .value("DEFAULT", amigo::OrderingType::NESTED_DISSECTION)
#else
      .value("DEFAULT", amigo::OrderingType::AMD)
#endif
      .export_values();

  py::enum_<amigo::MemoryLocation>(mod, "MemoryLocation")
      .value("HOST_AND_DEVICE", amigo::MemoryLocation::HOST_AND_DEVICE)
      .value("HOST_ONLY", amigo::MemoryLocation::HOST_ONLY)
      .value("DEVICE_ONLY", amigo::MemoryLocation::DEVICE_ONLY)
      .export_values();

  mod.def("reorder_model", &reorder_model, py::arg("order_type"),
          py::arg("arrays"), py::arg("output_indices") = py::none());

  py::class_<amigo::CSRMat<double>, std::shared_ptr<amigo::CSRMat<double>>>(
      mod, "CSRMat")
      .def(py::init(
               [](int nrows, int ncols,
                  py::array_t<int, py::array::c_style | py::array::forcecast>
                      rowp,
                  py::array_t<int, py::array::c_style | py::array::forcecast>
                      cols,
                  py::array_t<double, py::array::c_style | py::array::forcecast>
                      vals) {
                 if (rowp.ndim() != 1 || cols.ndim() != 1 || vals.ndim() != 1) {
                   throw std::runtime_error(
                       "rowp, cols, and vals must be 1D arrays");
                 }
                 if (rowp.shape(0) != nrows + 1) {
                   throw std::runtime_error("rowp must have length nrows + 1");
                 }
                 if (cols.shape(0) != vals.shape(0)) {
                   throw std::runtime_error(
                       "cols and vals must have the same length");
                 }
                 int nnz = static_cast<int>(vals.shape(0));
                 const int* rowp_ptr = rowp.data();
                 const int* cols_ptr = cols.data();
                 const double* vals_ptr = vals.data();
                 auto csr = amigo::CSRMat<double>::create_from_csr_data(
                     nrows, ncols, nnz, rowp_ptr, cols_ptr);
                 double* data = csr->get_data_ptr();
                 for (int i = 0; i < nnz; i++) {
                   data[i] = vals_ptr[i];
                 }
                 return csr;
               }),
           py::arg("nrows"), py::arg("ncols"), py::arg("rowp"), py::arg("cols"),
           py::arg("vals"))
      .def("get_nonzero_structure",
           [](amigo::CSRMat<double>& mat) {
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
           [](py::object self) -> py::array_t<double> {
             auto& mat = self.cast<amigo::CSRMat<double>&>();
             int nnz;
             double* mat_data;
             mat.get_data(nullptr, nullptr, &nnz, nullptr, nullptr, &mat_data);

             // Return a view (no copy) of the internal data array.
             // The `self` reference keeps the CSRMat alive.
             return py::array_t<double>({nnz}, {sizeof(double)}, mat_data,
                                        self);
           })
      .def("extract_submatrix",
           [](const amigo::CSRMat<double>& self, py::array_t<int> rows,
              py::array_t<int> cols) {
             return self.extract_submatrix(rows.size(), rows.data(),
                                           cols.size(), cols.data());
           })
      .def("extract_submatrix_values",
           [](const amigo::CSRMat<double>& self, py::array_t<int> rows,
              py::array_t<int> cols,
              std::shared_ptr<amigo::CSRMat<double>> mat) {
             self.extract_submatrix_values(rows.size(), rows.data(),
                                           cols.size(), cols.data(), mat);
           })
      .def("get_row_owners", &amigo::CSRMat<double>::get_row_owners)
      .def("get_column_owners", &amigo::CSRMat<double>::get_column_owners)
      .def("gauss_seidel", &amigo::CSRMat<double>::gauss_seidel)
      .def("mult", &amigo::CSRMat<double>::mult)
      .def("copy_data_device_to_host",
           &amigo::CSRMat<double>::copy_data_device_to_host);

  bind_vector<int>(mod, "VectorInt");
  bind_vector<double>(mod, "Vector");

  py::class_<
      amigo::ComponentGroupBase<double, detail::policy>,
      std::shared_ptr<amigo::ComponentGroupBase<double, detail::policy>>>(
      mod, "ComponentGroupBase");

  py::class_<
      amigo::ExternalComponentGroup<double, detail::policy>,
      amigo::ComponentGroupBase<double, detail::policy>,
      std::shared_ptr<amigo::ExternalComponentGroup<double, detail::policy>>>(
      mod, "ExternalComponentGroup")
      .def(py::init([](py::array_t<int> vars, py::array_t<int> cons,
                       py::array_t<int> rowp, py::array_t<int> cols,
                       py::object cb) {
        std::shared_ptr<PyExternalCallback<double>> extrn =
            std::make_shared<PyExternalCallback<double>>(
                vars.size(), cons.size(), rowp.data(), cols.data(), cb);

        return std::make_shared<
            amigo::ExternalComponentGroup<double, detail::policy>>(
            vars.size(), vars.data(), cons.size(), cons.data(), extrn);
      }));

  // SlackCouplingGroup: couples slack variables to inequality constraints
  // for the 2x2 augmented system (Wachter & Biegler 2006, eq. 13).
  py::class_<
      amigo::SlackCouplingGroup<double, detail::policy>,
      amigo::ComponentGroupBase<double, detail::policy>,
      std::shared_ptr<amigo::SlackCouplingGroup<double, detail::policy>>>(
      mod, "SlackCouplingGroup")
      .def(py::init(
          [](py::array_t<int> slack_indices, py::array_t<int> ineq_indices) {
            if (slack_indices.size() != ineq_indices.size()) {
              throw std::invalid_argument(
                  "slack_indices and ineq_indices must have the same length");
            }
            return std::make_shared<
                amigo::SlackCouplingGroup<double, detail::policy>>(
                static_cast<int>(slack_indices.size()), slack_indices.data(),
                ineq_indices.data());
          }));

  py::class_<amigo::NodeOwners, std::shared_ptr<amigo::NodeOwners>>(
      mod, "NodeOwners")
      .def(py::init([](py::object pyobj, py::array_t<int> ranges) {
        int size = 1;
        MPI_Comm comm = MPI_COMM_SELF;

        if (!pyobj.is_none()) {
          MPI_Comm* comm_ptr = PyMPIComm_Get(pyobj.ptr());
          if (!comm_ptr) {
            throw py::error_already_set();
          }

          comm = *comm_ptr;
          MPI_Comm_size(comm, &size);
        }
        if (ranges.size() != size + 1) {
          throw std::runtime_error(
              "Ranges must be of length MPI_Comm_size + 1");
        }
        return std::make_shared<amigo::NodeOwners>(comm, ranges.data());
      }))
      .def("get_mpi_comm",
           [](const amigo::NodeOwners& self) {
             return py::reinterpret_steal<py::object>(
                 PyMPIComm_New(self.get_mpi_comm()));
           })
      .def("get_local_size", &amigo::NodeOwners::get_local_size);

  py::class_<
      amigo::OptimizationProblem<double, detail::policy>,
      std::shared_ptr<amigo::OptimizationProblem<double, detail::policy>>>(
      mod, "OptimizationProblem")
      .def(py::init(
          [](py::object pyobj, std::shared_ptr<amigo::NodeOwners> data_owners,
             std::shared_ptr<amigo::NodeOwners> var_owners,
             std::shared_ptr<amigo::NodeOwners> output_owners,
             std::shared_ptr<amigo::Vector<int>> is_multiplier,
             const std::vector<std::shared_ptr<amigo::ComponentGroupBase<
                 double, detail::policy>>>& components,
             std::shared_ptr<amigo::Vector<int>> fixed_dofs) {
            MPI_Comm comm = MPI_COMM_SELF;
            if (!pyobj.is_none()) {
              comm = *PyMPIComm_Get(pyobj.ptr());
            }
            return std::make_shared<
                amigo::OptimizationProblem<double, detail::policy>>(
                comm, data_owners, var_owners, output_owners, is_multiplier,
                components, fixed_dofs);
          }))
      .def("get_num_variables",
           &amigo::OptimizationProblem<double,
                                       detail::policy>::get_num_variables)
      .def("partition_from_root",
           &amigo::OptimizationProblem<double,
                                       detail::policy>::partition_from_root,
           py::arg("root") = 0)
      .def("create_vector",
           &amigo::OptimizationProblem<double, detail::policy>::create_vector,
           py::arg("loc") = amigo::MemoryLocation::HOST_AND_DEVICE)
      .def("create_data_vector",
           &amigo::OptimizationProblem<double,
                                       detail::policy>::create_data_vector,
           py::arg("loc") = amigo::MemoryLocation::HOST_AND_DEVICE)
      .def("get_data_vector",
           &amigo::OptimizationProblem<double, detail::policy>::get_data_vector)
      .def("set_data_vector",
           &amigo::OptimizationProblem<double, detail::policy>::set_data_vector)
      .def(
          "get_multiplier_indicator",
          &amigo::OptimizationProblem<double,
                                      detail::policy>::get_multiplier_indicator)
      .def("get_local_to_global_node_numbers",
           &amigo::OptimizationProblem<
               double, detail::policy>::get_local_to_global_node_numbers)
      .def("get_local_to_global_data_numbers",
           &amigo::OptimizationProblem<
               double, detail::policy>::get_local_to_global_data_numbers)
      .def("update",
           &amigo::OptimizationProblem<double, detail::policy>::update)
      .def("add_diagonal",
           &amigo::OptimizationProblem<double, detail::policy>::add_diagonal)
      .def("lagrangian",
           &amigo::OptimizationProblem<double, detail::policy>::lagrangian,
           py::arg("alpha"), py::arg("x"))
      .def("gradient",
           &amigo::OptimizationProblem<double, detail::policy>::gradient,
           py::arg("alpha"), py::arg("x"), py::arg("grad"))
      .def("create_matrix",
           &amigo::OptimizationProblem<double, detail::policy>::create_matrix,
           py::arg("loc") = amigo::MemoryLocation::HOST_AND_DEVICE)
      .def("hessian",
           &amigo::OptimizationProblem<double, detail::policy>::hessian,
           py::arg("alpha"), py::arg("x"), py::arg("hess"))
      .def("scatter_vector",
           &amigo::OptimizationProblem<double,
                                       detail::policy>::scatter_vector<double>,
           py::arg("root_vec"), py::arg("dist_problem"), py::arg("dist_vec"),
           py::arg("root") = 0, py::arg("distribute") = true)
      .def("gather_vector",
           &amigo::OptimizationProblem<double,
                                       detail::policy>::gather_vector<double>,
           py::arg("dist_problem"), py::arg("dist_vec"), py::arg("root_vec"),
           py::arg("root") = 0)
      .def("scatter_data_vector",
           &amigo::OptimizationProblem<
               double, detail::policy>::scatter_data_vector<double>,
           py::arg("root_vec"), py::arg("dist_problem"), py::arg("dist_vec"),
           py::arg("root") = 0, py::arg("distribute") = true)
      .def("create_output_vector",
           &amigo::OptimizationProblem<double,
                                       detail::policy>::create_output_vector,
           py::arg("loc") = amigo::MemoryLocation::HOST_AND_DEVICE)
      .def("compute_output",
           &amigo::OptimizationProblem<double, detail::policy>::compute_output)
      .def("create_output_jacobian_wrt_input",
           &amigo::OptimizationProblem<
               double, detail::policy>::create_output_jacobian_wrt_input)
      .def("create_output_jacobian_wrt_data",
           &amigo::OptimizationProblem<
               double, detail::policy>::create_output_jacobian_wrt_data)
      .def("output_jacobian_wrt_input",
           &amigo::OptimizationProblem<
               double, detail::policy>::output_jacobian_wrt_input)
      .def(
          "output_jacobian_wrt_data",
          &amigo::OptimizationProblem<double,
                                      detail::policy>::output_jacobian_wrt_data)
      .def("create_gradient_jacobian_wrt_data",
           &amigo::OptimizationProblem<
               double, detail::policy>::create_gradient_jacobian_wrt_data)
      .def("gradient_jacobian_wrt_data",
           &amigo::OptimizationProblem<
               double, detail::policy>::gradient_jacobian_wrt_data);

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

  py::enum_<amigo::SparseLDL<double>::SolverType>(mod, "SolverType")
      .value("LDL", amigo::SparseLDL<double>::SolverType::LDL)
      .value("CHOLESKY", amigo::SparseLDL<double>::SolverType::CHOLESKY)
      .export_values();

  py::class_<amigo::SparseLDL<double>,
             std::shared_ptr<amigo::SparseLDL<double>>>(mod, "SparseLDL")
      .def(py::init<std::shared_ptr<amigo::CSRMat<double>>,
                    amigo::SparseLDL<double>::SolverType, double, double,
                    double, amigo::OrderingType>(),
           py::arg("mat"),
           py::arg("solver_type") = amigo::SparseLDL<double>::SolverType::LDL,
           py::arg("ustab") = 0.01, py::arg("pivot_tol") = 1e-14,
           py::arg("delay_growth") = 2.0,
           py::arg("order") = amigo::OrderingType::NATURAL)
      .def("factor", &amigo::SparseLDL<double>::factor)
      .def("solve", &amigo::SparseLDL<double>::solve)
      .def("get_inertia", [](const amigo::SparseLDL<double>& self) {
        int npos = 0, nneg = 0;
        self.get_inertia(&npos, &nneg);
        return py::make_tuple(npos, nneg);
      });

#ifdef AMIGO_USE_CUDA
  py::class_<amigo::CSRMatFactorCuda, std::shared_ptr<amigo::CSRMatFactorCuda>>(
      mod, "CSRMatFactorCuda")
      .def(py::init<std::shared_ptr<amigo::CSRMat<double>>, double>(),
           py::arg("mat"), py::arg("pivot_tol") = 1e-12)
      .def("factor", &amigo::CSRMatFactorCuda::factor)
      .def("solve", &amigo::CSRMatFactorCuda::solve);
#endif

  py::class_<amigo::OptVector<double>,
             std::shared_ptr<amigo::OptVector<double>>>(mod, "OptVector")
      .def("get_solution",
           py::overload_cast<>(&amigo::OptVector<double>::get_solution))
      .def("zero", &amigo::OptVector<double>::zero)
      .def("copy", &amigo::OptVector<double>::copy)
      .def("get_zl",
           [](const amigo::OptVector<double>& self) {
             const double *zl, *zu;
             self.get_bound_duals<detail::policy>(&zl, &zu);
             int n = self.get_n_primal();
             return std::vector<double>(zl, zl + n);
           })
      .def("get_zu",
           [](const amigo::OptVector<double>& self) {
             const double *zl, *zu;
             self.get_bound_duals<detail::policy>(&zl, &zu);
             int n = self.get_n_primal();
             return std::vector<double>(zu, zu + n);
           })
      .def("get_sl",
           [](const amigo::OptVector<double>& self) {
             const double *sl, *su;
             self.get_bound_slacks<detail::policy>(&sl, &su);
             int n = self.get_n_primal();
             return std::vector<double>(sl, sl + n);
           })
      .def("get_su",
           [](const amigo::OptVector<double>& self) {
             const double *sl, *su;
             self.get_bound_slacks<detail::policy>(&sl, &su);
             int n = self.get_n_primal();
             return std::vector<double>(su, su + n);
           })
      .def("get_slacks",
           [](amigo::OptVector<double>& self) { return self.get_slacks(); });

  using IPMOpt = amigo::InteriorPointOptimizer<double, detail::policy>;
  using OV = amigo::OptVector<double>;
  using Vec = amigo::Vector<double>;
  using CSR = amigo::CSRMat<double>;

  py::class_<IPMOpt, std::shared_ptr<IPMOpt>>(mod, "InteriorPointOptimizer")
      .def(py::init<
           std::shared_ptr<amigo::OptimizationProblem<double, detail::policy>>,
           std::shared_ptr<Vec>, std::shared_ptr<Vec>>())
      .def(
          "create_opt_vector",
          [](const IPMOpt& self, py::object x) {
            if (!x.is_none())
              return self.create_opt_vector(x.cast<std::shared_ptr<Vec>>());
            return self.create_opt_vector();
          },
          py::arg("x") = py::none())
      .def("set_multipliers_value", &IPMOpt::set_multipliers_value)
      .def("set_design_vars_value", &IPMOpt::set_design_vars_value)
      .def("copy_multipliers", &IPMOpt::copy_multipliers)
      .def("copy_design_vars", &IPMOpt::copy_design_vars)
      .def("initialize_multipliers_and_slacks",
           &IPMOpt::initialize_multipliers_and_slacks)
      .def("compute_residual", &IPMOpt::compute_residual)
      .def("compute_update", &IPMOpt::compute_update)
      .def("compute_diagonal", &IPMOpt::compute_diagonal)
      .def("compute_max_step",
           [](const IPMOpt& self, double tau, std::shared_ptr<OV> vars,
              std::shared_ptr<OV> upd) {
             double ax = 1.0, az = 1.0;
             int xi = -1, zi = -1;
             self.compute_max_step(tau, vars, upd, ax, xi, az, zi);
             return py::make_tuple(ax, xi, az, zi);
           })
      .def("apply_step_update", &IPMOpt::apply_step_update)
      .def(
          "compute_complementarity",
          [](const IPMOpt& self, std::shared_ptr<OV> vars) {
            double avg, xi;
            self.compute_complementarity(vars, avg, xi);
            return py::make_tuple(avg, xi);
          },
          py::arg("vars"))
      .def(
          "compute_complementarity_sq",
          [](const IPMOpt& self, std::shared_ptr<OV> vars) {
            double sq;
            self.compute_complementarity_sq(vars, sq);
            return sq;
          },
          py::arg("vars"))
      .def(
          "compute_max_comp_deviation",
          [](const IPMOpt& self, std::shared_ptr<OV> vars, double mu) {
            double md;
            self.compute_max_comp_deviation(vars, mu, md);
            return md;
          },
          py::arg("vars"), py::arg("mu"))
      .def("compute_barrier_log_sum", &IPMOpt::compute_barrier_log_sum,
           py::arg("barrier_param"), py::arg("vars"))
      .def("compute_barrier_dphi", &IPMOpt::compute_barrier_dphi,
           py::arg("barrier_param"), py::arg("vars"), py::arg("update"),
           py::arg("res"), py::arg("px"), py::arg("diag"))
      .def("compute_barrier_dphi_direct", &IPMOpt::compute_barrier_dphi_direct,
           py::arg("barrier_param"), py::arg("vars"), py::arg("grad"),
           py::arg("px"))
      .def("reset_bound_multipliers", &IPMOpt::reset_bound_multipliers,
           py::arg("barrier_param"), py::arg("kappa_sigma"), py::arg("vars"))
      .def("compute_constraint_violation_1norm",
           &IPMOpt::compute_constraint_violation_1norm, py::arg("vars"),
           py::arg("grad"))
      .def(
          "compute_kkt_error",
          [](const IPMOpt& self, std::shared_ptr<OV> vars,
             std::shared_ptr<Vec> grad) {
            double d, p, c;
            self.compute_kkt_error(vars, grad, d, p, c);
            return py::make_tuple(d, p, c);
          },
          py::arg("vars"), py::arg("grad"))
      .def(
          "compute_kkt_error_mu",
          [](const IPMOpt& self, double mu, std::shared_ptr<OV> vars,
             std::shared_ptr<Vec> grad) {
            double d, p, c;
            self.compute_kkt_error_mu(mu, vars, grad, d, p, c);
            return py::make_tuple(d, p, c);
          },
          py::arg("mu"), py::arg("vars"), py::arg("grad"))
      .def(
          "compute_residual_and_infeasibility",
          [](const IPMOpt& self, double mu, std::shared_ptr<OV> vars,
             std::shared_ptr<Vec> grad, std::shared_ptr<Vec> res) {
            double d, p;
            self.compute_residual_and_infeasibility(mu, vars, grad, res, d, p);
            return py::make_tuple(d, p);
          },
          py::arg("barrier_param"), py::arg("vars"), py::arg("grad"),
          py::arg("res"))
      .def("get_kkt_element_counts",
           [](const IPMOpt& self) {
             int d, p, c;
             self.get_kkt_element_counts(d, p, c);
             return py::make_tuple(d, p, c);
           })
      .def("compute_affine_start_point", &IPMOpt::compute_affine_start_point)
      .def("compute_dual_residual_vector",
           &IPMOpt::compute_dual_residual_vector, py::arg("vars"),
           py::arg("grad"), py::arg("output"))
      .def("check_update", &IPMOpt::check_update)
      .def("get_lbx",
           [](const IPMOpt& self) {
             const auto& v = *self.get_lbx();
             const double* a = v.template get_array<detail::policy>();
             return std::vector<double>(a, a + v.get_size());
           })
      .def("get_ubx",
           [](const IPMOpt& self) {
             const auto& v = *self.get_ubx();
             const double* a = v.template get_array<detail::policy>();
             return std::vector<double>(a, a + v.get_size());
           })
      .def("get_lbx_relaxed",
           [](const IPMOpt& self) {
             const auto& v = *self.get_lbx_relaxed();
             const double* a = v.template get_array<detail::policy>();
             return std::vector<double>(a, a + v.get_size());
           })
      .def("get_ubx_relaxed",
           [](const IPMOpt& self) {
             const auto& v = *self.get_ubx_relaxed();
             const double* a = v.template get_array<detail::policy>();
             return std::vector<double>(a, a + v.get_size());
           })
      .def("get_num_inequalities", &IPMOpt::get_num_inequalities)
      .def("get_num_design_variables", &IPMOpt::get_num_design_variables)
      .def("relax_bounds", &IPMOpt::relax_bounds, py::arg("factor") = 1e-8,
           py::arg("constr_viol_tol") = 1e-4)
      .def(
          "set_slack_mapping",
          [](IPMOpt& self, py::array_t<int> slack_indices,
             py::array_t<int> constr_indices) {
            if (slack_indices.size() != constr_indices.size()) {
              throw std::invalid_argument(
                  "slack_indices and constr_indices must have the same length");
            }
            self.set_slack_mapping(static_cast<int>(slack_indices.size()),
                                   slack_indices.data(), constr_indices.data());
          },
          py::arg("slack_indices"), py::arg("constr_indices"))
      .def("initialize_slacks", &IPMOpt::initialize_slacks, py::arg("grad"),
           py::arg("vars"))
      .def("has_slacks", &IPMOpt::has_slacks)
      // NLP scaling
      .def("compute_nlp_scaling", &IPMOpt::compute_nlp_scaling, py::arg("x"),
           py::arg("grad"), py::arg("max_gradient") = 100.0,
           py::arg("min_value") = 1e-8)
      .def("apply_gradient_scaling", &IPMOpt::apply_gradient_scaling,
           py::arg("grad"))
      .def("apply_hessian_scaling", &IPMOpt::apply_hessian_scaling,
           py::arg("hess"))
      .def("scale_multipliers", &IPMOpt::scale_multipliers, py::arg("x"))
      .def("unscale_multipliers", &IPMOpt::unscale_multipliers, py::arg("x"))
      .def("get_obj_scale", &IPMOpt::get_obj_scale)
      .def("has_scaling", &IPMOpt::has_scaling);
}
