#pragma once

#include <memory>
#include <vector>

#include "infini_train/include/tensor.h"

namespace infini_train::kernels::cuda {
std::shared_ptr<Tensor> MeanForward(const std::shared_ptr<Tensor> &input, const int64_t dim, const bool keep_dim);
std::shared_ptr<Tensor> MeanBackward(const std::shared_ptr<Tensor> &grad_output, const std::vector<int64_t> &input_dims,
                                     const int64_t dim, bool keep_dim);

std::shared_ptr<Tensor> SumForward(const std::shared_ptr<Tensor> &input, const int64_t dim, const bool keep_dim);
std::shared_ptr<Tensor> SumBackward(const std::shared_ptr<Tensor> &grad_output, const std::vector<int64_t> &input_dims,
                                    const int64_t dim, bool keep_dim);

std::shared_ptr<Tensor> MaxForward(const std::shared_ptr<Tensor> &input, const int64_t dim, const bool keep_dim);
std::shared_ptr<Tensor> MaxBackward(const std::shared_ptr<Tensor> &grad_output, const std::shared_ptr<Tensor> &input,
                                    const std::shared_ptr<Tensor> &reduced, const int64_t dim, bool keep_dim);

std::shared_ptr<Tensor> MinForward(const std::shared_ptr<Tensor> &input, const int64_t dim, const bool keep_dim);
std::shared_ptr<Tensor> MinBackward(const std::shared_ptr<Tensor> &grad_output, const std::shared_ptr<Tensor> &input,
                                    const std::shared_ptr<Tensor> &reduced, const int64_t dim, bool keep_dim);

} // namespace infini_train::kernels::cuda
