#ifndef GEMM_LOWBIT_KERNEL // include guard
#define GEMM_LOWBIT_KERNEL

#include <torch/extension.h>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
void gemm_lowbit(at::Tensor a, at::Tensor b, at::Tensor c, float w_scale, float x_scale);

#endif