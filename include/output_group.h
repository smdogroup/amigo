#ifndef AMIGO_OUTPUT_GROUP_H
#define AMIGO_OUTPUT_GROUP_H

#include "component_group_base.h"
#include "output_group_base.h"

namespace amigo {

template <typename T, int ncomp, int ndata, int noutputs, class... Components>
class SerialOutputBackend {
 public:
  SerialOutputBackend(IndexLayout<ndata> &data_layout,
                      IndexLayout<ncomp> &layout,
                      IndexLayout<noutputs> &output_layout) {}

  void add_output_kernel(const IndexLayout<ndata> &data_layout,
                         const IndexLayout<ncomp> &layout,
                         const IndexLayout<noutputs> &output_layout,
                         const Vector<T> &data_vec, const Vector<T> &vec,
                         Vector<T> &out) const {
    add_output_kernel<Components...>(data_layout, layout, output_layout,
                                     data_vec, vec, out);
  }

  template <class Component, class... Remain>
  void add_output_kernel(const IndexLayout<ndata> &data_layout,
                         const IndexLayout<ncomp> &layout,
                         const IndexLayout<noutputs> &output_layout,
                         const Vector<T> &data_vec, const Vector<T> &vec,
                         Vector<T> &out) const {
    typename Component::template Data<T> data;
    typename Component::template Input<T> input;
    typename Component::template Output<T> output;
    int length = layout.get_length();

    for (int i = 0; i < length; i++) {
      data_layout.get_values(i, data_vec, data);
      layout.get_values(i, vec, input);
      Component::analyze(data, input, output);
      output_layout.add_values(i, output, out);
    }

    if constexpr (sizeof...(Remain) > 0) {
      add_output_kernel<Remain...>(data_layout, layout, output_layout, data_vec,
                                   vec, out);
    }
  }

  void add_jacobian_kernel(const IndexLayout<ndata> &data_layout,
                           const IndexLayout<ncomp> &layout,
                           const IndexLayout<noutputs> &output_layout,
                           const Vector<T> &data_vec, const Vector<T> &vec,
                           CSRMat<T> &jac) const {
    add_jacobian_kernel<Components...>(data_layout, layout, output_layout,
                                       data_vec, vec, jac);
  }

  template <class Component, class... Remain>
  void add_jacobian_kernel(const IndexLayout<ndata> &data_layout,
                           const IndexLayout<ncomp> &layout,
                           const IndexLayout<noutputs> &output_layout,
                           const Vector<T> &data_vec, const Vector<T> &vec,
                           CSRMat<T> &jac) const {
    typename Component::template Data<A2D::ADScalar<T, 1>> data;
    typename Component::template Input<A2D::ADScalar<T, 1>> input;
    typename Component::template Output<A2D::ADScalar<T, 1>> output;
    int length = layout.get_length();

    for (int i = 0; i < length; i++) {
      data_layout.get_values(i, data_vec, data);

      T jac_elem[noutputs * ncomp];
      for (int j = 0; j < ncomp; j++) {
        layout.get_values(i, vec, input);
        input[j].deriv[0] = 1.0;
        Component::analyze(data, input, output);

        for (int k = 0; k < noutputs; k++) {
          jac_elem[ncomp * k + i] = output[k].deriv[0];
        }
      }

      int input_index[ncomp];
      layout.get_indices(i, input_index);

      int output_index[noutputs];
      output_layout.get_indices(i, output_index);

      for (int j = 0; j < noutputs; j++) {
        jac.add_row(output_index[j], ncomp, input_index, &jac_elem[ncomp * j]);
      }
    }

    if constexpr (sizeof...(Remain) > 0) {
      add_jacobian_kernel<Remain...>(data_layout, layout, output_layout,
                                     data_vec, vec, jac);
    }
  }
};

template <typename T, int ncomp, int ndata, int noutputs, class... Components>
using DefaultOutputBackend =
    SerialOutputBackend<T, ncomp, ndata, noutputs, Components...>;

template <typename T, class... Components>
class OutputGroup : public OutputGroupBase<T> {
 public:
  static constexpr int num_components = sizeof...(Components);

  static constexpr int ncomp = __get_collection_ncomp<Components...>::value;
  static constexpr int ndata = __get_collection_ndata<Components...>::value;
  static constexpr int noutputs =
      __get_collection_noutputs<Components...>::value;

  using Backend =
      DefaultOutputBackend<T, ncomp, ndata, noutputs, Components...>;

  OutputGroup(std::shared_ptr<Vector<int>> data_indices,
              std::shared_ptr<Vector<int>> indices,
              std::shared_ptr<Vector<int>> output_indices)
      : data_layout(data_indices),
        layout(indices),
        output_layout(output_indices),
        backend(data_layout, layout, output_layout) {}

  void add_output(const Vector<T> &data_vec, const Vector<T> &vec,
                  Vector<T> &output) const {
    backend.add_output_kernel(data_layout, layout, output_layout, data_vec, vec,
                              output);
  }

  void add_jacobian(const Vector<T> &data_vec, const Vector<T> &vec,
                    CSRMat<T> &jac) const {
    backend.add_jacobian_kernel(data_layout, layout, output_layout, data_vec,
                                vec, jac);
  }

  void get_layout_data(int *length_, int *nout_, int *ncomp_,
                       const int **outputs_, const int **inputs_) const {
    output_layout.get_data(length_, nout_, outputs_);
    layout.get_data(length_, ncomp_, inputs_);
  }

 private:
  IndexLayout<ndata> data_layout;
  IndexLayout<ncomp> layout;
  IndexLayout<noutputs> output_layout;

  Backend backend;
};

}  // namespace amigo

#endif  // AMIGO_OUTPUT_GROUP_H