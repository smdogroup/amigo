#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "csr_matrix.h"
#include "optimization_problem.h"

namespace py = pybind11;

PYBIND11_MODULE(amigo, mod) {
  mod.doc() = "Amigo: A friendly library for MDO on GPUs";

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

  // py::class_<amigo::ComponentSet<double>>(mod, "ComponentSet")
  //     .def(py::init<>());

  py::class_<amigo::OptimizationProblem<double>>(mod, "OptimizationProblem")
      .def(
          py::init<std::vector<std::shared_ptr<amigo::ComponentSet<double>>>>())
      .def("get_num_dof", &amigo::OptimizationProblem<double>::get_num_dof)
      .def("lagrangian",
           [](amigo::OptimizationProblem<double> &self, py::array_t<double> x) {
             int size = self.get_num_dof();
             auto x_vec = self.create_vector();
             std::memcpy(x_vec->get_host_array(), x.data(),
                         size * sizeof(double));
             return self.lagrangian(x_vec);
           })
      .def("gradient",
           [](amigo::OptimizationProblem<double> &self,
              py::array_t<double> x) -> py::array_t<double> {
             int size = self.get_num_dof();
             auto g_vec = self.create_vector();
             auto x_vec = self.create_vector();

             std::memcpy(x_vec->get_host_array(), x.data(),
                         size * sizeof(double));

             self.gradient(x_vec, g_vec);

             py::array_t<double> g(size);
             std::memcpy(g.mutable_data(), g_vec->get_host_array(),
                         size * sizeof(double));
             return g;
           })
      .def("create_csr_matrix",
           &amigo::OptimizationProblem<double>::create_csr_matrix)
      .def("hessian", [](amigo::OptimizationProblem<double> &self,
                         py::array_t<double> x,
                         std::shared_ptr<amigo::CSRMat<double>> &mat) {
        int size = self.get_num_dof();
        auto x_vec = self.create_vector();
        std::memcpy(x_vec->get_host_array(), x.data(), size * sizeof(double));

        self.hessian(x_vec, mat);
      });
}
