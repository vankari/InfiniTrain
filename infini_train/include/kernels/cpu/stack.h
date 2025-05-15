#pragma once

#include <memory>
#include <vector>

#include "infini_train/include/tensor.h"

namespace infini_train::kernels::cpu {
std::shared_ptr<Tensor> StackForward(const std::vector<std::shared_ptr<Tensor>> &inputs, int64_t dim);
std::vector<std::shared_ptr<Tensor>> StackBackward(const std::vector<int64_t> &input_dims, int64_t dim,
                                                   const std::shared_ptr<Tensor> &grad_output);
} // namespace infini_train::kernels::cpu
