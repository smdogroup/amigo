#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "amigo_include_paths.h"
#include "csr_matrix.h"
#include "optimization_problem.h"

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
                                   self->get_host_array(), py::cast(self));
           });
}

PYBIND11_MODULE(amigo, mod) {
  mod.doc() = "Amigo: A friendly library for MDO on GPUs";

  mod.attr("A2D_INCLUDE_PATH") = A2D_INCLUDE_PATH;
  mod.attr("AMIGO_INCLUDE_PATH") = AMIGO_INCLUDE_PATH;

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
      .def("get_data", [](amigo::CSRMat<double> &mat) -> py::array_t<double> {
        py::array_t<double> data(mat.nnz);
        std::memcpy(data.mutable_data(), mat.data, mat.nnz * sizeof(double));
        return data;
      });

  bind_vector<int>(mod, "VectorInt");
  bind_vector<double>(mod, "Vector");

  py::class_<amigo::ComponentGroupBase<double>,
             std::shared_ptr<amigo::ComponentGroupBase<double>>>(
      mod, "ComponentGroupBase");

  py::class_<amigo::OptimizationProblem<double>,
             std::shared_ptr<amigo::OptimizationProblem<double>>>(
      mod, "OptimizationProblem")
      .def(py::init<
           int,
           std::vector<std::shared_ptr<amigo::ComponentGroupBase<double>>>>())
      .def("create_vector", &amigo::OptimizationProblem<double>::create_vector)
      .def("lagrangian", &amigo::OptimizationProblem<double>::lagrangian)
      .def("gradient", &amigo::OptimizationProblem<double>::gradient)
      .def("create_csr_matrix",
           &amigo::OptimizationProblem<double>::create_csr_matrix)
      .def("hessian", &amigo::OptimizationProblem<double>::hessian);
}
