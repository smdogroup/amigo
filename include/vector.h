#ifndef MDGO_VECTOR_H
#define MDGO_VECTOR_H

#include <random>

namespace mdgo {

enum class VectorType { HOST_AND_DEVICE, DEVICE_ONLY, HOST_ONLY };

template <typename T, int dim = 1>
class Vector {
 public:
  Vector(int length, VectorType vtype = VectorType::HOST_AND_DEVICE)
      : length(length), vtype(vtype), h_array(nullptr), d_array(nullptr) {
    if (vtype == VectorType::HOST_AND_DEVICE ||
        vtype == VectorType::HOST_ONLY) {
      h_array = new T[length * dim];
    }
    if (vtype == VectorType::HOST_AND_DEVICE ||
        vtype == VectorType::DEVICE_ONLY) {
      // cudaMalloc((void **)&d_array, length * dim * sizeof(T));
    }
  }

  ~Vector() {
    if (d_array) {
      // cudaFree(d_array);
    }
    if (h_array) {
      delete[] h_array;
    }
  }

  void host_to_device() {
    // cudaMemcpy(d_array, h_array, length * dim * sizeof(T),
    //            cudaMemcpyHostToDevice);
  }

  void device_to_host() {
    // cudaMemcpy(h_array, d_array, length * dim * sizeof(T),
    //            cudaMemcpyDeviceToHost);
  }

  void zero() {
    if (h_array) {
      memset(h_array, 0, length * dim * sizeof(T));
    }
  }

  void set_random(T low = -1.0, T high = 1.0) {
    if (h_array) {
      std::random_device rd;
      std::mt19937 gen(rd());
      std::uniform_real_distribution<T> dis(low, high);

      for (int i = 0; i < length * dim; i++) {
        h_array[i] = dis(gen);
      }
    }
  }

  void axpy(T alpha, Vector<T, dim>& x) {
    if (h_array && x.h_array) {
      for (int i = 0; i < length * dim; i++) {
        h_array[i] += alpha * x.h_array[i];
      }
    }
  }

  T dot(const Vector<T, dim>& x) const {
    T value = 0.0;
    if (h_array && x.h_array) {
      for (int i = 0; i < length * dim; i++) {
        value += h_array[i] * x.h_array[i];
      }
    }
    return value;
  }

  void scale(T alpha) {
    if (h_array) {
      for (int i = 0; i < length * dim; i++) {
        h_array[i] *= alpha;
      }
    }
  }

  int get_length() const { return length; }
  int get_size() const { return length * dim; }

  T* get_host_array() { return h_array; }
  const T* get_host_array() const { return h_array; }

  T* get_device_array() { return d_array; }
  const T* get_device_array() const { return d_array; }

 private:
  int length;
  VectorType vtype;
  T* h_array;  // Host data
  T* d_array;  // Device data
};

}  // namespace mdgo

#endif  // MDGO_VECTOR_H