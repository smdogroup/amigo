#ifndef AMIGO_H
#define AMIGO_H

#ifdef AMIGO_USE_CUDA
#define AMIGO_KERNEL __global__
#define AMGIO_DEVICE __device__
#endif  // AMIGO_USE_CUDA

#endif  // AMIGO_H