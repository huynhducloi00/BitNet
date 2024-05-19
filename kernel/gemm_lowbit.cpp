#include "gemm_lowbit_kernel.h"
#include <iostream>
using namespace std;
// The wrapper function to be called from Python
void gemm_lowbit_running(at::Tensor a, at::Tensor b, at::Tensor c, float w_scale, float x_scale)
{
    auto M = a.size(0);
    auto K = a.size(1);
    auto N = b.size(1);

    // Ensure inputs are on the correct device and are of half precision
    a = a.to(at::device(at::kCUDA).dtype(at::kHalf));
    b = b.to(at::device(at::kCUDA).dtype(at::kHalf));
    c = c.to(at::device(at::kCUDA).dtype(at::kHalf));
    // Call the CUDA kernel wrapper
    gemm_lowbit(a, b, c, w_scale, x_scale);
}

int main(void)
{
    printf("abc");
    auto a = torch::randn({10, 20}, torch::kCUDA);
    auto b = torch::randn({20, 30}, torch::kCUDA);
    auto c = torch::randn({10, 30}, torch::kCUDA);

    // # Example usage
    // a = torch.randn(10, 20, dtype=torch.half, device='cuda')  # Example tensor
    // b = torch.randn(20, 30, dtype=torch.half, device='cuda')  # Example tensor
    // c = torch.empty(10, 30, dtype=torch.half, device='cuda')  # Output tensor

    auto w_scale = 1.0f;
    auto x_scale = 1.0f;
    gemm_lowbit_running(a, b, c, w_scale, x_scale);
    // std::cout << c;
}
// // The PyBind11 module definition
// PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
//     m.def("gemm_lowbit", &gemm_lowbit, "A low precision GEMM operation with scaling");
// }
