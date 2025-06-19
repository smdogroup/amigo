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
#include "quasidef_cholesky.h"

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

py::array_t<int> reorder_model(std::vector<py::array_t<int>> arrays) {
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
  amigo::OrderingUtils::create_csr_from_elements(
      nrows, ncols, nelems, element_nodes, include_diagonal, sort_columns,
      &rowp, &cols);

  // Compute the reordering
  int *perm, *iperm;
  amigo::OrderingUtils::nested_disection(nrows, ncols, rowp, cols, &perm,
                                         &iperm);

  delete[] rowp;
  delete[] cols;
  delete[] perm;

  auto pyiperm = py::array_t<int>({nrows}, {sizeof(int)}, iperm);
  delete[] iperm;

  return pyiperm;
}

PYBIND11_MODULE(amigo, mod) {
  mod.doc() = "Amigo: A friendly library for MDO on GPUs";

  mod.attr("A2D_INCLUDE_PATH") = A2D_INCLUDE_PATH;
  mod.attr("AMIGO_INCLUDE_PATH") = AMIGO_INCLUDE_PATH;

  mod.def("reorder_model", &reorder_model);

  py::class_<amigo::CSRMat<double>, std::shared_ptr<amigo::CSRMat<double>>>(
      mod, "CSRMat")
      .def("get_nonzero_structure",
           [](amigo::CSRMat<double> &mat) {
             py::array_t<int> rowp(mat.nrows + 1);
             std::memcpy(rowp.mutable_data(), mat.rowp,
                         (mat.nrows + 1) * sizeof(int));

             py::array_t<int> cols(mat.nnz);
             std::memcpy(cols.mutable_data(), mat.cols, mat.nnz * sizeof(int));
             return py::make_tuple(mat.nrows, mat.ncols, mat.nnz, rowp, cols);
           })
      .def("get_data",
           [](amigo::CSRMat<double> &mat) -> py::array_t<double> {
             py::array_t<double> data(mat.nnz);
             std::memcpy(data.mutable_data(), mat.data,
                         mat.nnz * sizeof(double));
             return data;
           })
      .def("mult", &amigo::CSRMat<double>::mult);

  bind_vector<int>(mod, "VectorInt");
  bind_vector<double>(mod, "Vector");

  py::class_<amigo::ComponentGroupBase<double>,
             std::shared_ptr<amigo::ComponentGroupBase<double>>>(
      mod, "ComponentGroupBase");

  py::class_<amigo::OptimizationProblem<double>,
             std::shared_ptr<amigo::OptimizationProblem<double>>>(
      mod, "OptimizationProblem")
      .def(py::init<
           int, int,
           std::vector<std::shared_ptr<amigo::ComponentGroupBase<double>>>>())
      .def("get_data_vector",
           &amigo::OptimizationProblem<double>::get_data_vector)
      .def("create_vector", &amigo::OptimizationProblem<double>::create_vector)
      .def("lagrangian", &amigo::OptimizationProblem<double>::lagrangian)
      .def("gradient", &amigo::OptimizationProblem<double>::gradient)
      .def("create_csr_matrix",
           &amigo::OptimizationProblem<double>::create_csr_matrix)
      .def("hessian", &amigo::OptimizationProblem<double>::hessian);

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

  py::class_<amigo::QuasidefCholesky<double>,
             std::shared_ptr<amigo::QuasidefCholesky<double>>>(
      mod, "QuasidefCholesky")
      .def(py::init<std::shared_ptr<amigo::CSRMat<double>>>())
      .def("factor", &amigo::QuasidefCholesky<double>::factor)
      .def("solve", &amigo::QuasidefCholesky<double>::solve);
}
