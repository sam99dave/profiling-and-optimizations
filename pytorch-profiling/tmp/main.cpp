#include <torch/extension.h>
/* ---- Python ---- */
// def batched_dot_mul_sum(a, b):
//     return a.mul(b).sum(-1)

torch::Tensor batched_dot_mul_sum_v0(
    const torch::Tensor a,
    const torch::Tensor b) {
  return a.mul(b).sum(-1);
}

torch::Tensor batched_dot_mul_sum_v1(
    const torch::Tensor& a,
    const torch::Tensor& b) {
  return a.mul(b).sum(-1);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
m.def("batched_dot_mul_sum_v0", torch::wrap_pybind_function(batched_dot_mul_sum_v0), "batched_dot_mul_sum_v0");
m.def("batched_dot_mul_sum_v1", torch::wrap_pybind_function(batched_dot_mul_sum_v1), "batched_dot_mul_sum_v1");
}