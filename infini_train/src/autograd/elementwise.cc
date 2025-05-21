#include "infini_train/include/autograd/elementwise.h"

#include "infini_train/include/dispatcher.h"
#include "infini_train/include/tensor.h"

namespace infini_train::autograd {
std::vector<std::shared_ptr<Tensor>> Neg::Forward(const std::vector<std::shared_ptr<Tensor>> &input_tensors) {
    CHECK_EQ(input_tensors.size(), 1);
    const auto &input = input_tensors[0];

    std::shared_ptr<Tensor> output = nullptr;
    switch (input->GetDevice().Type()) {
    case DeviceType::kCPU: {
        output = kernels::cpu::NegForward(input);
        break;
    }
#ifdef USE_CUDA
    case DeviceType::kCUDA: {
        output = kernels::cuda::NegForward(input);
        break;
    }
#endif
    default:
        LOG(FATAL) << "Unsupported device type: " << static_cast<int>(input->GetDevice().Type());
        break;
    }
    return {output};
}

std::vector<std::shared_ptr<Tensor>> Neg::Backward(const std::vector<std::shared_ptr<Tensor>> &grad_outputs) {
    CHECK_EQ(grad_outputs.size(), 1);
    const auto &grad_output = grad_outputs[0];

    std::shared_ptr<Tensor> grad_input = nullptr;
    switch (grad_output->GetDevice().Type()) {
    case DeviceType::kCPU: {
        grad_input = kernels::cpu::NegBackward(grad_output);
        break;
    }
#ifdef USE_CUDA
    case DeviceType::kCUDA: {
        grad_input = kernels::cuda::NegBackward(grad_output);
        break;
    }
#endif
    default:
        LOG(FATAL) << "Unsupported device type: " << static_cast<int>(grad_output->GetDevice().Type());
        break;
    }
    return {grad_input};
}

std::vector<std::shared_ptr<Tensor>> Reciprocal::Forward(const std::vector<std::shared_ptr<Tensor>> &input_tensors) {
    CHECK_EQ(input_tensors.size(), 1);
    const auto &input = input_tensors[0];

    std::shared_ptr<Tensor> output = nullptr;
    switch (input->GetDevice().Type()) {
    case DeviceType::kCPU: {
        output = kernels::cpu::ReciprocalForward(input);
        break;
    }
#ifdef USE_CUDA
    case DeviceType::kCUDA: {
        output = kernels::cuda::ReciprocalForward(input);
        break;
    }
#endif
    default:
        LOG(FATAL) << "Unsupported device type: " << static_cast<int>(input->GetDevice().Type());
        break;
    }
    return {output};
}

void Reciprocal::SetupContext(const std::vector<std::shared_ptr<Tensor>> &input_tensors,
                              const std::vector<std::shared_ptr<Tensor>> &) {
    const auto &input = input_tensors[0];
    saved_tensors_ = {input};
}

std::vector<std::shared_ptr<Tensor>> Reciprocal::Backward(const std::vector<std::shared_ptr<Tensor>> &grad_outputs) {
    CHECK_EQ(saved_tensors_.size(), 1);
    const auto &input = saved_tensors_[0];
    CHECK_EQ(grad_outputs.size(), 1);
    const auto &grad_output = grad_outputs[0];

    std::shared_ptr<Tensor> grad_input = nullptr;
    switch (input->GetDevice().Type()) {
    case DeviceType::kCPU: {
        grad_input = kernels::cpu::ReciprocalBackward(grad_output, input);
        break;
    }
#ifdef USE_CUDA
    case DeviceType::kCUDA: {
        grad_input = kernels::cuda::ReciprocalBackward(grad_output, input);
        break;
    }
#endif
    default:
        LOG(FATAL) << "Unsupported device type: " << static_cast<int>(input->GetDevice().Type());
        break;
    }
    return {grad_input};
}

std::vector<std::shared_ptr<Tensor>> Sin::Forward(const std::vector<std::shared_ptr<Tensor>> &input_tensors) {
    CHECK_EQ(input_tensors.size(), 1);
    const auto &input = input_tensors[0];

    std::shared_ptr<Tensor> output = nullptr;
    switch (input->GetDevice().Type()) {
    case DeviceType::kCPU: {
        output = kernels::cpu::SinForward(input);
        break;
    }
#ifdef USE_CUDA
    case DeviceType::kCUDA: {
        output = kernels::cuda::SinForward(input);
        break;
    }
#endif
    default:
        LOG(FATAL) << "Unsupported device type: " << static_cast<int>(input->GetDevice().Type());
        break;
    }
    return {output};
}

void Sin::SetupContext(const std::vector<std::shared_ptr<Tensor>> &input_tensors,
                       const std::vector<std::shared_ptr<Tensor>> &) {
    const auto &input = input_tensors[0];
    saved_tensors_ = {input};
}

std::vector<std::shared_ptr<Tensor>> Sin::Backward(const std::vector<std::shared_ptr<Tensor>> &grad_outputs) {
    CHECK_EQ(saved_tensors_.size(), 1);
    const auto &input = saved_tensors_[0];
    CHECK_EQ(grad_outputs.size(), 1);
    const auto &grad_output = grad_outputs[0];

    std::shared_ptr<Tensor> grad_input = nullptr;
    switch (input->GetDevice().Type()) {
    case DeviceType::kCPU: {
        grad_input = kernels::cpu::SinBackward(grad_output, input);
        break;
    }
#ifdef USE_CUDA
    case DeviceType::kCUDA: {
        grad_input = kernels::cuda::SinBackward(grad_output, input);
        break;
    }
#endif
    default:
        LOG(FATAL) << "Unsupported device type: " << static_cast<int>(input->GetDevice().Type());
        break;
    }
    return {grad_input};
}

std::vector<std::shared_ptr<Tensor>> Cos::Forward(const std::vector<std::shared_ptr<Tensor>> &input_tensors) {
    CHECK_EQ(input_tensors.size(), 1);
    const auto &input = input_tensors[0];

    std::shared_ptr<Tensor> output = nullptr;
    switch (input->GetDevice().Type()) {
    case DeviceType::kCPU: {
        output = kernels::cpu::CosForward(input);
        break;
    }
#ifdef USE_CUDA
    case DeviceType::kCUDA: {
        output = kernels::cuda::CosForward(input);
        break;
    }
#endif
    default:
        LOG(FATAL) << "Unsupported device type: " << static_cast<int>(input->GetDevice().Type());
        break;
    }
    return {output};
}

void Cos::SetupContext(const std::vector<std::shared_ptr<Tensor>> &input_tensors,
                       const std::vector<std::shared_ptr<Tensor>> &) {
    const auto &input = input_tensors[0];
    saved_tensors_ = {input};
}

std::vector<std::shared_ptr<Tensor>> Cos::Backward(const std::vector<std::shared_ptr<Tensor>> &grad_outputs) {
    CHECK_EQ(saved_tensors_.size(), 1);
    const auto &input = saved_tensors_[0];
    CHECK_EQ(grad_outputs.size(), 1);
    const auto &grad_output = grad_outputs[0];

    std::shared_ptr<Tensor> grad_input = nullptr;
    switch (input->GetDevice().Type()) {
    case DeviceType::kCPU: {
        grad_input = kernels::cpu::CosBackward(grad_output, input);
        break;
    }
#ifdef USE_CUDA
    case DeviceType::kCUDA: {
        grad_input = kernels::cuda::CosBackward(grad_output, input);
        break;
    }
#endif
    default:
        LOG(FATAL) << "Unsupported device type: " << static_cast<int>(input->GetDevice().Type());
        break;
    }
    return {grad_input};
}

std::vector<std::shared_ptr<Tensor>> Tanh::Forward(const std::vector<std::shared_ptr<Tensor>> &input_tensors) {
    CHECK_EQ(input_tensors.size(), 1);
    const auto &input = input_tensors[0];

    auto device = input->GetDevice().Type();
    auto kernel = Dispatcher::Instance().GetKernel({device, "TanhForward"});
    return {kernel.Call<std::shared_ptr<Tensor>>(input)};
}

void Tanh::SetupContext(const std::vector<std::shared_ptr<Tensor>> &,
                        const std::vector<std::shared_ptr<Tensor>> &output_tensors) {
    const auto &output = output_tensors[0];
    saved_tensors_ = {output};
}

std::vector<std::shared_ptr<Tensor>> Tanh::Backward(const std::vector<std::shared_ptr<Tensor>> &grad_outputs) {
    CHECK_EQ(saved_tensors_.size(), 1);
    const auto &output = saved_tensors_[0];
    CHECK_EQ(grad_outputs.size(), 1);
    const auto &grad_output = grad_outputs[0];

    auto device = output->GetDevice().Type();
    auto kernel = Dispatcher::Instance().GetKernel({device, "TanhBackward"});
    return {kernel.Call<std::shared_ptr<Tensor>>(grad_output, output)};
}

std::vector<std::shared_ptr<Tensor>> Pow::Forward(const std::vector<std::shared_ptr<Tensor>> &input_tensors) {
    CHECK_EQ(input_tensors.size(), 1);
    const auto &input = input_tensors[0];

    auto device = input->GetDevice().Type();
    auto kernel = Dispatcher::Instance().GetKernel({device, "PowForward"});
    return {kernel.Call<std::shared_ptr<Tensor>>(input, exponent_, scalar_is_base_)};
}

void Pow::SetupContext(const std::vector<std::shared_ptr<Tensor>> &input_tensors,
                       const std::vector<std::shared_ptr<Tensor>> &) {
    const auto &input = input_tensors[0];
    saved_tensors_ = {input};
}

std::vector<std::shared_ptr<Tensor>> Pow::Backward(const std::vector<std::shared_ptr<Tensor>> &grad_outputs) {
    CHECK_EQ(saved_tensors_.size(), 1);
    const auto &input = saved_tensors_[0];
    CHECK_EQ(grad_outputs.size(), 1);
    const auto &grad_output = grad_outputs[0];

    auto device = input->GetDevice().Type();
    auto kernel = Dispatcher::Instance().GetKernel({device, "PowBackward"});
    return {kernel.Call<std::shared_ptr<Tensor>>(grad_output, input, exponent_, scalar_is_base_)};
}

std::vector<std::shared_ptr<Tensor>> EqualsScalar::Forward(const std::vector<std::shared_ptr<Tensor>> &input_tensors) {
    CHECK_EQ(input_tensors.size(), 1);
    const auto &input = input_tensors[0];

    auto device = input->GetDevice().Type();
    auto kernel = Dispatcher::Instance().GetKernel({device, "EqualsScalarForward"});
    return {kernel.Call<std::shared_ptr<Tensor>>(input, scalar_)};
}

std::vector<std::shared_ptr<Tensor>> EqualsScalar::Backward(const std::vector<std::shared_ptr<Tensor>> &grad_outputs) {
    LOG(FATAL) << "EqualsScalar::Backward shall not be called anytime";
    return {};
}

std::vector<std::shared_ptr<Tensor>> Add::Forward(const std::vector<std::shared_ptr<Tensor>> &input_tensors) {
    CHECK_EQ(input_tensors.size(), 2);
    const auto &a = input_tensors[0];
    const auto &b = input_tensors[1];

    auto device = a->GetDevice().Type();
    auto kernel = Dispatcher::Instance().GetKernel({device, "AddForward"});
    return {kernel.Call<std::shared_ptr<Tensor>>(a, b)};
}

void Add::SetupContext(const std::vector<std::shared_ptr<Tensor>> &input_tensors,
                       const std::vector<std::shared_ptr<Tensor>> &output_tensors) {
    a_dims_ = input_tensors[0]->Dims();
    b_dims_ = input_tensors[1]->Dims();
}

std::vector<std::shared_ptr<Tensor>> Add::Backward(const std::vector<std::shared_ptr<Tensor>> &grad_outputs) {
    CHECK_EQ(grad_outputs.size(), 1);
    const auto &grad_output = grad_outputs[0];

    auto device = grad_output->GetDevice().Type();
    auto kernel = Dispatcher::Instance().GetKernel({device, "AddBackward"});
    auto [grad_a, grad_b]
        = kernel.Call<std::pair<std::shared_ptr<Tensor>, std::shared_ptr<Tensor>>>(grad_output, a_dims_, b_dims_);
    return {grad_a, grad_b};
}

std::vector<std::shared_ptr<Tensor>> AddScalar::Forward(const std::vector<std::shared_ptr<Tensor>> &input_tensors) {
    CHECK_EQ(input_tensors.size(), 1);
    const auto &input = input_tensors[0];

    auto device = input->GetDevice().Type();
    auto kernel = Dispatcher::Instance().GetKernel({device, "AddScalarForward"});
    return {kernel.Call<std::shared_ptr<Tensor>>(input, scalar_)};
}

std::vector<std::shared_ptr<Tensor>> AddScalar::Backward(const std::vector<std::shared_ptr<Tensor>> &grad_outputs) {
    CHECK_EQ(grad_outputs.size(), 1);
    const auto &grad_output = grad_outputs[0];

    auto device = grad_output->GetDevice().Type();
    auto kernel = Dispatcher::Instance().GetKernel({device, "AddScalarBackward"});
    return {kernel.Call<std::shared_ptr<Tensor>>(grad_output)};
}

std::vector<std::shared_ptr<Tensor>> Mul::Forward(const std::vector<std::shared_ptr<Tensor>> &input_tensors) {
    CHECK_EQ(input_tensors.size(), 2);
    const auto &a = input_tensors[0];
    const auto &b = input_tensors[1];

    auto device = a->GetDevice().Type();
    auto kernel = Dispatcher::Instance().GetKernel({device, "MulForward"});
    return {kernel.Call<std::shared_ptr<Tensor>>(a, b)};
}

void Mul::SetupContext(const std::vector<std::shared_ptr<Tensor>> &input_tensors,
                       const std::vector<std::shared_ptr<Tensor>> &) {
    const auto &a = input_tensors[0];
    const auto &b = input_tensors[1];
    saved_tensors_ = {a, b};
}

std::vector<std::shared_ptr<Tensor>> Mul::Backward(const std::vector<std::shared_ptr<Tensor>> &grad_outputs) {
    CHECK_EQ(saved_tensors_.size(), 2);
    const auto &a = saved_tensors_[0];
    const auto &b = saved_tensors_[1];
    CHECK_EQ(grad_outputs.size(), 1);
    const auto &grad_output = grad_outputs[0];

    auto device = grad_output->GetDevice().Type();
    auto kernel = Dispatcher::Instance().GetKernel({device, "MulBackward"});
    auto [grad_a, grad_b] = kernel.Call<std::pair<std::shared_ptr<Tensor>, std::shared_ptr<Tensor>>>(grad_output, a, b);
    return {grad_a, grad_b};
}

std::vector<std::shared_ptr<Tensor>> MulScalar::Forward(const std::vector<std::shared_ptr<Tensor>> &input_tensors) {
    CHECK_EQ(input_tensors.size(), 1);
    const auto &input = input_tensors[0];

    auto device = input->GetDevice().Type();
    auto kernel = Dispatcher::Instance().GetKernel({device, "MulScalarForward"});
    return {kernel.Call<std::shared_ptr<Tensor>>(input, scalar_)};
}

std::vector<std::shared_ptr<Tensor>> MulScalar::Backward(const std::vector<std::shared_ptr<Tensor>> &grad_outputs) {
    CHECK_EQ(grad_outputs.size(), 1);
    const auto &grad_output = grad_outputs[0];

    auto device = grad_output->GetDevice().Type();
    auto kernel = Dispatcher::Instance().GetKernel({device, "MulScalarBackward"});
    return {kernel.Call<std::shared_ptr<Tensor>>(grad_output, scalar_)};
}
} // namespace infini_train::autograd
