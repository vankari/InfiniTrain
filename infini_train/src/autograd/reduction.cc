#include "infini_train/include/autograd/reduction.h"

#include <memory>
#include <vector>

#include "glog/logging.h"

#include "infini_train/include/kernels/cpu/reduction.h"
#include "infini_train/include/tensor.h"
#ifdef USE_CUDA
#include "infini_train/include/kernels/cuda/reduction.h"
#endif

namespace infini_train::autograd {
std::vector<std::shared_ptr<Tensor>> Mean::Forward(const std::vector<std::shared_ptr<Tensor>> &input_tensors) {
    CHECK_EQ(input_tensors.size(), 1);
    const auto &input = input_tensors[0];

    std::shared_ptr<Tensor> output = nullptr;
    switch (input->GetDevice().Type()) {
    case DeviceType::kCPU: {
        output = kernels::cpu::MeanForward(input, dim_, keep_dim_);
        break;
    }
#ifdef USE_CUDA
    case DeviceType::kCUDA: {
        output = kernels::cuda::MeanForward(input, dim_, keep_dim_);
        break;
    }
#endif
    default:
        LOG(FATAL) << "Unsupported device type: " << static_cast<int>(input->GetDevice().Type());
        break;
    }
    return {output};
}

void Mean::SetupContext(const std::vector<std::shared_ptr<Tensor>> &input_tensors,
                        const std::vector<std::shared_ptr<Tensor>> &output_tensors) {
    const auto &input = input_tensors[0];
    input_dims_ = input->Dims();
}

std::vector<std::shared_ptr<Tensor>> Mean::Backward(const std::vector<std::shared_ptr<Tensor>> &grad_outputs) {
    CHECK_EQ(grad_outputs.size(), 1);
    const auto &grad_output = grad_outputs[0];

    std::shared_ptr<Tensor> grad_input = nullptr;
    switch (grad_output->GetDevice().Type()) {
    case DeviceType::kCPU: {
        grad_input = kernels::cpu::MeanBackward(grad_output, input_dims_, dim_, keep_dim_);
        return {grad_input};
    }
#ifdef USE_CUDA
    case DeviceType::kCUDA: {
        grad_input = kernels::cuda::MeanBackward(grad_output, input_dims_, dim_, keep_dim_);
        return {grad_input};
    }
#endif
    default:
        LOG(FATAL) << "Unsupported device type: " << static_cast<int>(grad_output->GetDevice().Type());
        break;
    }
    return {grad_input};
}

std::vector<std::shared_ptr<Tensor>> Sum::Forward(const std::vector<std::shared_ptr<Tensor>> &input_tensors) {
    CHECK_EQ(input_tensors.size(), 1);
    const auto &input = input_tensors[0];

    std::shared_ptr<Tensor> output = nullptr;
    switch (input->GetDevice().Type()) {
    case DeviceType::kCPU: {
        output = kernels::cpu::SumForward(input, dim_, keep_dim_);
        break;
    }
#ifdef USE_CUDA
    case DeviceType::kCUDA: {
        output = kernels::cuda::SumForward(input, dim_, keep_dim_);
        break;
    }
#endif
    default:
        LOG(FATAL) << "Unsupported device type: " << static_cast<int>(input->GetDevice().Type());
        break;
    }
    return {output};
}

void Sum::SetupContext(const std::vector<std::shared_ptr<Tensor>> &input_tensors,
                       const std::vector<std::shared_ptr<Tensor>> &output_tensors) {
    const auto &input = input_tensors[0];
    input_dims_ = input->Dims();
}

std::vector<std::shared_ptr<Tensor>> Sum::Backward(const std::vector<std::shared_ptr<Tensor>> &grad_outputs) {
    CHECK_EQ(grad_outputs.size(), 1);
    const auto &grad_output = grad_outputs[0];

    std::shared_ptr<Tensor> grad_input = nullptr;
    switch (grad_output->GetDevice().Type()) {
    case DeviceType::kCPU: {
        grad_input = kernels::cpu::SumBackward(grad_output, input_dims_, dim_, keep_dim_);
        return {grad_input};
    }
#ifdef USE_CUDA
    case DeviceType::kCUDA: {
        grad_input = kernels::cuda::SumBackward(grad_output, input_dims_, dim_, keep_dim_);
        return {grad_input};
    }
#endif
    default:
        LOG(FATAL) << "Unsupported device type: " << static_cast<int>(grad_output->GetDevice().Type());
        break;
    }
    return {grad_input};
}

std::vector<std::shared_ptr<Tensor>> Max::Forward(const std::vector<std::shared_ptr<Tensor>> &input_tensors) {
    CHECK_EQ(input_tensors.size(), 1);
    const auto &input = input_tensors[0];

    std::shared_ptr<Tensor> output = nullptr;
    switch (input->GetDevice().Type()) {
    case DeviceType::kCPU: {
        output = kernels::cpu::MaxForward(input, dim_, keep_dim_);
        break;
    }
#ifdef USE_CUDA
    case DeviceType::kCUDA: {
        output = kernels::cuda::MaxForward(input, dim_, keep_dim_);
        break;
    }
#endif
    default:
        LOG(FATAL) << "Unsupported device type: " << static_cast<int>(input->GetDevice().Type());
        break;
    }
    return {output};
}

void Max::SetupContext(const std::vector<std::shared_ptr<Tensor>> &input_tensors,
                       const std::vector<std::shared_ptr<Tensor>> &output_tensors) {
    const auto &input = input_tensors[0];
    const auto &output = output_tensors[0];
    saved_tensors_ = {input, output};
}

std::vector<std::shared_ptr<Tensor>> Max::Backward(const std::vector<std::shared_ptr<Tensor>> &grad_outputs) {
    CHECK_EQ(grad_outputs.size(), 1);
    CHECK_EQ(saved_tensors_.size(), 2);
    const auto &grad_output = grad_outputs[0];
    const auto &input = saved_tensors_[0];
    const auto &reduced = saved_tensors_[1];

    std::shared_ptr<Tensor> grad_input = nullptr;
    switch (grad_output->GetDevice().Type()) {
    case DeviceType::kCPU: {
        grad_input = kernels::cpu::MaxBackward(grad_output, input, reduced, dim_, keep_dim_);
        return {grad_input};
    }
#ifdef USE_CUDA
    case DeviceType::kCUDA: {
        grad_input = kernels::cuda::MaxBackward(grad_output, input, reduced, dim_, keep_dim_);
        return {grad_input};
    }
#endif
    default:
        LOG(FATAL) << "Unsupported device type: " << static_cast<int>(grad_output->GetDevice().Type());
        break;
    }
    return {grad_input};
}

std::vector<std::shared_ptr<Tensor>> Min::Forward(const std::vector<std::shared_ptr<Tensor>> &input_tensors) {
    CHECK_EQ(input_tensors.size(), 1);
    const auto &input = input_tensors[0];

    std::shared_ptr<Tensor> output = nullptr;
    switch (input->GetDevice().Type()) {
    case DeviceType::kCPU: {
        output = kernels::cpu::MinForward(input, dim_, keep_dim_);
        break;
    }
#ifdef USE_CUDA
    case DeviceType::kCUDA: {
        output = kernels::cuda::MinForward(input, dim_, keep_dim_);
        break;
    }
#endif
    default:
        LOG(FATAL) << "Unsupported device type: " << static_cast<int>(input->GetDevice().Type());
        break;
    }
    return {output};
}

void Min::SetupContext(const std::vector<std::shared_ptr<Tensor>> &input_tensors,
                       const std::vector<std::shared_ptr<Tensor>> &output_tensors) {
    const auto &input = input_tensors[0];
    const auto &output = output_tensors[0];
    saved_tensors_ = {input, output};
}

std::vector<std::shared_ptr<Tensor>> Min::Backward(const std::vector<std::shared_ptr<Tensor>> &grad_outputs) {
    CHECK_EQ(grad_outputs.size(), 1);
    CHECK_EQ(saved_tensors_.size(), 2);
    const auto &grad_output = grad_outputs[0];
    const auto &input = saved_tensors_[0];
    const auto &reduced = saved_tensors_[1];

    std::shared_ptr<Tensor> grad_input = nullptr;
    switch (grad_output->GetDevice().Type()) {
    case DeviceType::kCPU: {
        grad_input = kernels::cpu::MinBackward(grad_output, input, reduced, dim_, keep_dim_);
        return {grad_input};
    }
#ifdef USE_CUDA
    case DeviceType::kCUDA: {
        grad_input = kernels::cuda::MinBackward(grad_output, input, reduced, dim_, keep_dim_);
        return {grad_input};
    }
#endif
    default:
        LOG(FATAL) << "Unsupported device type: " << static_cast<int>(grad_output->GetDevice().Type());
        break;
    }
    return {grad_input};
}
} // namespace infini_train::autograd
