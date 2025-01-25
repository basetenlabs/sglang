// Adapted from
// https://github.com/InternLM/lmdeploy/blob/800b6010c0bf76aadf678bc38a507b749fb9774c/src/turbomind/kernels/norm/rms_norm.cu

#include <turbomind/kernels/core/array_ops.h>
#include <turbomind/kernels/core/common.h>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <pytorch_extension_utils.h>
#include <THC/THCAtomics.cuh>

#include "utils.h"

#include <cub/block/block_reduce.cuh>

using namespace turbomind;

template <class T, class Tacc, int block_dim, int vec_size>
__global__ void BiasResidualRMSNormKernel(T* __restrict__ residual, T* __restrict__ hidden_states,
                                          const T* __restrict__ weights, const T* __restrict__ bias, int dims, int num,
                                          float eps, float inv_dims) {
  const int ti = blockIdx.x;
  const int di = threadIdx.x * vec_size;

  if (ti >= num) {
    return;
  }

  residual += dims * ti;
  hidden_states += dims * ti;

  Array<Tacc, vec_size> accum{};

  Array<T, vec_size> r_vec;
  Array<T, vec_size> h_vec;
  Array<T, vec_size> b_vec;

  for (int i = di; i < dims; i += block_dim * vec_size) {
    Load(r_vec, &residual[i]);
    Load(h_vec, &hidden_states[i]);

    using namespace ops;
    r_vec = r_vec + h_vec;

    if (bias) {
      Ldg(b_vec, &bias[i]);
      r_vec = r_vec + b_vec;
    }

    Store(&residual[i], r_vec);

    Array<Tacc, vec_size> tmp = cast<Tacc>(r_vec);

    accum = accum + tmp * tmp;
  }

  float sum{};
  PRAGMA_UNROLL
  for (int i = 0; i < vec_size; ++i) {
    sum += accum[i];
  }

  using BlockReduce = cub::BlockReduce<Tacc, block_dim>;
  __shared__ typename BlockReduce::TempStorage temp_storage;

  sum = BlockReduce{temp_storage}.Sum(sum);

  __shared__ float shared_sum;

  if (threadIdx.x == 0) {
    shared_sum = rsqrtf(sum * inv_dims + eps);
  }

  __syncthreads();

  sum = shared_sum;

  Array<T, vec_size> w_vec;
  for (int i = di; i < dims; i += block_dim * vec_size) {
    Load(r_vec, &residual[i]);
    Ldg(w_vec, &weights[i]);
    PRAGMA_UNROLL
    for (int c = 0; c < vec_size; ++c) {
      r_vec[c] = (T)((float)r_vec[c] * sum) * w_vec[c];
    }
    Store(&hidden_states[i], r_vec);
  }
}

void invokeBiasResidualRMSNorm(torch::Tensor residual, torch::Tensor hidden_states, 
                               const torch::Tensor& weights, std::optional<torch::Tensor>& bias,
                               float eps) {
  const auto dims = residual.size(-1);
  const auto num = residual.numel() / dims;
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  DISPATCH_PYTORCH_DTYPE_TO_CTYPE_FP16(residual.scalar_type(), scalar_t, [&] {
    constexpr int vec_size = 16 / sizeof(scalar_t);
    constexpr int threads = 512;
    const int blocks = num;

    BiasResidualRMSNormKernel<scalar_t, float, threads, vec_size>
        <<<blocks, threads, 0, stream>>>(
            static_cast<scalar_t*>(residual.data_ptr()),
            static_cast<scalar_t*>(hidden_states.data_ptr()),
            static_cast<scalar_t*>(weights.data_ptr()),
            bias.has_value() ? static_cast<scalar_t*>(bias.value().data_ptr()) : nullptr,
            dims, num, eps, 1.f / dims);
    return true;
  });
}
