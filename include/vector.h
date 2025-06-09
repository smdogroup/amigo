#ifndef AMIGO_VECTOR_H
#define AMIGO_VECTOR_H

#include <random>

namespace amigo {

enum class VectorType { HOST_AND_DEVICE, DEVICE_ONLY, HOST_ONLY };

template <typename T>
class Vector {
 public:
  Vector(int size, VectorType vtype = VectorType::HOST_AND_DEVICE)
      : size(size), vtype(vtype), h_array(nullptr), d_array(nullptr) {
    if (vtype == VectorType::HOST_AND_DEVICE ||
        vtype == VectorType::HOST_ONLY) {
      h_array = new T[size];
    }
    if (vtype == VectorType::HOST_AND_DEVICE ||
        vtype == VectorType::DEVICE_ONLY) {
      // cudaMalloc((void **)&d_array, size * sizeof(T));
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
    // cudaMemcpy(d_array, h_array, size * sizeof(T),
    //            cudaMemcpyHostToDevice);
  }

  void device_to_host() {
    // cudaMemcpy(h_array, d_array, size * sizeof(T),
    //            cudaMemcpyDeviceToHost);
  }

  void zero() {
    if (h_array) {
      memset(h_array, 0, size * sizeof(T));
    }
  }

  void set_random(T low = -1.0, T high = 1.0) {
    if (h_array) {
      std::random_device rd;
      std::mt19937 gen(rd());
      std::uniform_real_distribution<T> dis(low, high);

      for (int i = 0; i < size; i++) {
        h_array[i] = dis(gen);
      }
    }
  }

  void axpy(T alpha, Vector<T>& x) {
    if (h_array && x.h_array) {
      for (int i = 0; i < size; i++) {
        h_array[i] += alpha * x.h_array[i];
      }
    }
  }

  T dot(const Vector<T>& x) const {
    T value = 0.0;
    if (h_array && x.h_array) {
      for (int i = 0; i < size; i++) {
        value += h_array[i] * x.h_array[i];
      }
    }
    return value;
  }

  void scale(T alpha) {
    if (h_array) {
      for (int i = 0; i < size; i++) {
        h_array[i] *= alpha;
      }
    }
  }

  int get_size() const { return size; }

  T* get_host_array() { return h_array; }
  const T* get_host_array() const { return h_array; }

  T* get_device_array() { return d_array; }
  const T* get_device_array() const { return d_array; }

 private:
  int size;
  VectorType vtype;
  T* h_array;  // Host data
  T* d_array;  // Device data
};

}  // namespace amigo

#endif  // AMIGO_VECTOR_H