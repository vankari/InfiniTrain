#include "infini_train/include/op.h"

#include <cmath>
#include <cstdint>
#include <random>
#include <vector>

#include "glog/logging.h"

#include "infini_train/include/tensor.h"

namespace infini_train {
Op::Op(const std::vector<Tensor *> input_tensors)
    : input_tensors_(input_tensors) {}

void Op::OutputTensorsUseGradient() {
    for (auto &tensor : output_tensors_) {
        tensor.UseGradient();
    }
}

void Op::Backward() {
    BackwardImpl();
    for (auto *tensor : input_tensors_) {
        tensor->Backward();
    }
}

void Op::AddWeight(const std::vector<int64_t> &dims, const DataType dtype) {
    weights_.emplace_back(dims, dtype);
    weights_.back().UseGradient();

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(-1.0f, 1.0f);
    weights_.back().Fill(dis(gen));
}

std::vector<Tensor> &Op::Weights() { return weights_; }
const std::vector<Tensor> &Op::Weights() const { return weights_; }

std::vector<Tensor> &Op::OutputTensors() { return output_tensors_; }
const std::vector<Tensor> &Op::OutputTensors() const { return output_tensors_; }

namespace ops {
Linear::Linear(const std::vector<Tensor *> &input_tensors, int64_t out_dim)
    : Op(input_tensors), out_dim_(out_dim) {
    CHECK_EQ(input_tensors.size(), 1);
    const auto &input_dims = input_tensors[0]->Dims();
    CHECK_EQ(input_dims.size(), 2);
    bs_ = input_dims[0];
    in_dim_ = input_dims[1];
    AddWeight({in_dim_, out_dim}, DataType::kFLOAT32);
    AddWeight({out_dim}, DataType::kFLOAT32);

    output_tensors_.emplace_back(Tensor{{bs_, out_dim}, DataType::kFLOAT32});
    OutputTensorsUseGradient();
}

void Linear::Forward() {
    const Tensor &input = *input_tensors_[0];
    Tensor &output = output_tensors_[0];
    auto &w_ = weights_[0];
    auto &b_ = weights_[1];

    for (int64_t i = 0; i < bs_; ++i) {
        for (int64_t j = 0; j < out_dim_; ++j) {
            float sum = 0.0f;
            for (int64_t k = 0; k < in_dim_; ++k) {
                sum += reinterpret_cast<const float *>(input.DataPtr())[i * in_dim_ + k] * reinterpret_cast<float *>(w_.DataPtr())[k * out_dim_ + j];
            }
            reinterpret_cast<float *>(output.DataPtr())[i * out_dim_ + j] = sum + reinterpret_cast<float *>(b_.DataPtr())[j];
        }
    }
}

void Linear::BackwardImpl() {
    // Get the input tensor and output tensor
    Tensor &input = *input_tensors_[0];
    Tensor &output = output_tensors_[0];
    auto &w_ = weights_[0];
    auto &b_ = weights_[1];

    // Compute the gradient of the linear transformation
    if (input.Gradient()) {
        for (int64_t i = 0; i < bs_; ++i) {
            for (int64_t j = 0; j < in_dim_; ++j) {
                float sum = 0.0f;
                for (int64_t k = 0; k < out_dim_; ++k) {
                    sum += reinterpret_cast<float *>(
                               w_.Gradient()->DataPtr())[j * out_dim_ + k]
                         * reinterpret_cast<float *>(
                               output.Gradient()->DataPtr())[i * out_dim_ + k];
                }
                reinterpret_cast<float *>(
                    input.Gradient()->DataPtr())[i * in_dim_ + j]
                    = sum;
            }
        }
    }

    // Compute the gradient of the weights and biases
    for (int64_t i = 0; i < bs_; ++i) {
        for (int64_t j = 0; j < in_dim_; ++j) {
            for (int64_t k = 0; k < out_dim_; ++k) {
                reinterpret_cast<float *>(w_.Gradient()->DataPtr())[j * out_dim_ + k] += reinterpret_cast<float *>(input.DataPtr())[i * in_dim_ + j] * reinterpret_cast<float *>(output.Gradient()->DataPtr())[i * out_dim_ + k];
            }
        }
        for (int64_t k = 0; k < out_dim_; ++k) {
            reinterpret_cast<float *>(b_.Gradient()->DataPtr())[k] += reinterpret_cast<float *>(
                output.Gradient()->DataPtr())[i * out_dim_ + k];
        }
    }
}

Sigmoid::Sigmoid(const std::vector<Tensor *> input_tensors)
    : Op(input_tensors) {
    CHECK_EQ(input_tensors.size(), 1);
    const auto &input_dims = input_tensors[0]->Dims();
    CHECK_EQ(input_dims.size(), 2);
    bs_ = input_dims[0];
    out_dim_ = input_dims[1];

    output_tensors_.emplace_back(Tensor{{bs_, out_dim_}, DataType::kFLOAT32});
    OutputTensorsUseGradient();
}

void Sigmoid::Forward() {
    const Tensor &input = *input_tensors_[0];
    Tensor &output = output_tensors_[0];

    for (int64_t i = 0; i < bs_; ++i) {
        for (int64_t j = 0; j < out_dim_; ++j) {
            const float x = reinterpret_cast<const float *>(input.DataPtr())[i * out_dim_ + j];
            reinterpret_cast<float *>(output.DataPtr())[i * out_dim_ + j] = 1.0f / (1.0f + exp(-x));
        }
    }
}

void Sigmoid::BackwardImpl() {
    // Get the input tensor and output tensor
    Tensor &input = *input_tensors_[0];
    Tensor &output = output_tensors_[0];

    // Compute the gradient of the sigmoid function
    for (int64_t i = 0; i < bs_; ++i) {
        for (int64_t j = 0; j < out_dim_; ++j) {
            const float x = reinterpret_cast<float *>(output.DataPtr())[i * out_dim_ + j];
            const float grad = reinterpret_cast<float *>(
                output.Gradient()->DataPtr())[i * out_dim_ + j];
            reinterpret_cast<float *>(input.Gradient()->DataPtr())[i * out_dim_ + j] = grad * x * (1.0f - x);
        }
    }
}

CrossEntropy::CrossEntropy(const std::vector<Tensor *> input_tensors)
    : Op(input_tensors) {
    CHECK_EQ(input_tensors.size(), 2);

    const auto &input_dims = input_tensors[0]->Dims();
    CHECK_EQ(input_dims.size(), 2);
    bs_ = input_dims[0];
    num_classes_ = input_dims[1];

    const auto &target_dims = input_tensors[1]->Dims();
    CHECK_EQ(target_dims.size(), 1);
    CHECK_EQ(target_dims[0], bs_);

    output_tensors_.emplace_back(Tensor{{}, DataType::kFLOAT32});
    OutputTensorsUseGradient();
}

void CrossEntropy::Forward() {
    const Tensor &input = *input_tensors_[0];  // Logits (before softmax)
    const Tensor &target = *input_tensors_[1]; // raw target index
    Tensor &output = output_tensors_[0];

    float loss = 0.0f;
    std::vector<float> log_softmax_probs(bs_ * num_classes_);

    // compute log_softmax
    for (int64_t i = 0; i < bs_; ++i) {
        // extract logits and compute softmax denominator
        float max_logit = -std::numeric_limits<float>::infinity();
        for (int64_t j = 0; j < num_classes_; ++j) {
            max_logit = std::max(max_logit, reinterpret_cast<const float *>(
                                                input.DataPtr())[i * num_classes_ + j]);
        }

        // compute softmax denominator
        float sum_exp = 0.0f;
        for (int64_t j = 0; j < num_classes_; ++j) {
            float exp_val = exp(reinterpret_cast<const float *>(
                                    input.DataPtr())[i * num_classes_ + j]
                                - max_logit);
            sum_exp += exp_val;
        }

        // compute log_softmax and store
        for (int64_t j = 0; j < num_classes_; ++j) {
            float exp_val = exp(reinterpret_cast<const float *>(
                                    input.DataPtr())[i * num_classes_ + j]
                                - max_logit);
            log_softmax_probs[i * num_classes_ + j] = log(exp_val / sum_exp);
        }

        // compute cross-entropy loss
        int64_t target_idx = target.DataPtr()[i];
        loss -= log_softmax_probs[i * num_classes_ + target_idx];
    }

    // compute average loss
    reinterpret_cast<float *>(output.DataPtr())[0] = loss / bs_;
}

void CrossEntropy::BackwardImpl() {
    Tensor &input = *input_tensors_[0];
    const Tensor &target = *input_tensors_[1];

    if (input.Gradient()) {
        std::vector<float> softmax_probs(bs_ * num_classes_);

        // compute softmax
        for (int64_t i = 0; i < bs_; ++i) {
            float max_logit = -std::numeric_limits<float>::infinity();
            for (int64_t j = 0; j < num_classes_; ++j) {
                max_logit = std::max(max_logit, reinterpret_cast<const float *>(
                                                    input.DataPtr())[i * num_classes_ + j]);
            }

            float sum_exp = 0.0f;
            for (int64_t j = 0; j < num_classes_; ++j) {
                float exp_val = exp(reinterpret_cast<const float *>(
                                        input.DataPtr())[i * num_classes_ + j]
                                    - max_logit);
                sum_exp += exp_val;
            }

            for (int64_t j = 0; j < num_classes_; ++j) {
                float exp_val = exp(reinterpret_cast<const float *>(
                                        input.DataPtr())[i * num_classes_ + j]
                                    - max_logit);
                softmax_probs[i * num_classes_ + j] = exp_val / sum_exp;
            }
        }

        // compute dL/dx = softmax_probs - one_hot(target)
        for (int64_t i = 0; i < bs_; ++i) {
            int64_t target_idx = target.DataPtr()[i];
            for (int j = 0; j < num_classes_; ++j) {
                float grad = softmax_probs[i * num_classes_ + j] - (j == target_idx ? 1.0f : 0.0f);
                reinterpret_cast<float *>(
                    input.Gradient()->DataPtr())[i * num_classes_ + j]
                    = grad / bs_; // normalize by batch size
            }
        }
    }
}

} // namespace ops
} // namespace infini_train
