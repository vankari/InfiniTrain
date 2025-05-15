#pragma once

#include <memory>
#include <tuple>

#include "infini_train/include/tensor.h"

namespace infini_train::kernels::cuda {
std::shared_ptr<Tensor> OuterForward(const std::shared_ptr<Tensor> &input, const std::shared_ptr<Tensor> &other);
std::tuple<std::shared_ptr<Tensor>, std::shared_ptr<Tensor>> OuterBackward(const std::shared_ptr<Tensor> &input,
                                                                           const std::shared_ptr<Tensor> &other,
                                                                           const std::shared_ptr<Tensor> &grad_output);

} // namespace infini_train::kernels::cuda
