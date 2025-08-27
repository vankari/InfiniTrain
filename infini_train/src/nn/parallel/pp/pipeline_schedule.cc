// pipeline_schedule.cc
#include "infini_train/include/nn/parallel/pp/pipeline_schedule.h"

#include "glog/logging.h"
#include <cuda_runtime.h>
#include <vector>

#include "infini_train/include/device.h"
#include "infini_train/include/nn/modules/module.h"
#include "infini_train/include/tensor.h"
#include "infini_train/src/nn/parallel/pp/send_recv.h"

namespace infini_train::nn::pipeline {

std::vector<std::vector<std::shared_ptr<Tensor>>>
PipelineSchedule::SplitTensor(const std::vector<std::shared_ptr<Tensor>> &full_inputs) {

    if (full_inputs.empty()) {
        LOG(FATAL) << "SplitTensor: no input tensors provided.";
    }

    const auto &first_dims = full_inputs[0]->Dims();
    if (first_dims.empty()) {
        LOG(FATAL) << "SplitTensor: tensor has no dimensions.";
    }
    int64_t batch_size = first_dims[0];

    int microbatch_size = batch_size / num_microbatches_;
    int remainder = batch_size % num_microbatches_;

    std::vector<std::vector<std::shared_ptr<Tensor>>> micro_batches(num_microbatches_);

    int start_idx = 0;
    for (int mb_idx = 0; mb_idx < num_microbatches_; ++mb_idx) {
        int current_size = microbatch_size + (mb_idx == num_microbatches_ - 1 ? remainder : 0);

        for (const auto &tensor : full_inputs) {
            if (tensor->Dims()[0] != batch_size) {
                LOG(FATAL) << "SplitTensor: tensor size mismatch on dim 0.";
            }

            auto sliced = tensor->Slice(0, start_idx, current_size); // 切片：返回 view（共享数据，支持反向传播）
            micro_batches[mb_idx].push_back(sliced);
        }

        start_idx += current_size;
    }

    return micro_batches;
}

float PipelineSchedule::Step(const std::vector<std::shared_ptr<Tensor>> &input,
                             const std::vector<std::shared_ptr<Tensor>> &target,
                             const std::shared_ptr<Module> &loss_fn) {

    auto micro_batches = SplitTensor(input);
    auto target_mbs = !target.empty() ? SplitTensor(target)[0] : std::vector<std::shared_ptr<Tensor>>();

    // 验证 microbatch 数量
    if (micro_batches.empty() || micro_batches[0].empty()) {
        LOG(FATAL) << "No microbatches to process.";
    }

    float lossf = StepMicrobatches(micro_batches, target_mbs, loss_fn);

    return lossf;
}

float Schedule1F1B::StepMicrobatches(const std::vector<std::vector<std::shared_ptr<Tensor>>> &microbatch_inputs,
                                     const std::vector<std::shared_ptr<Tensor>> &microbatch_targets,
                                     const std::shared_ptr<Module> &loss_fn) {
    const int n_microbatches = NumMicrobatches();
    if (n_microbatches == 0) {
        return 0.0;
    }

    std::vector<std::shared_ptr<Tensor>> outputs(n_microbatches);

    int mb_idx = 0;       // 前向mrcro_batch索引
    int bwd_mb_index = 0; // 反向mrcro_batch索引

    // Warmup 阶段
    int warmup_steps = stage_->IsLastStage() ? n_microbatches
                                             : std::min(n_microbatches, stage_->num_stages() - stage_->stage_index());

    for (int i = 0; i < warmup_steps; ++i) {
        std::vector<std::shared_ptr<Tensor>> input_tensors;
        if (stage_->IsFirstStage()) {
            input_tensors = microbatch_inputs[mb_idx];
        } else {
            auto shape = stage_->recv_shape();
            auto recv_tensor
                = std::make_shared<Tensor>(std::vector<int64_t>{shape.batch_size, shape.seq_len, shape.hidden_size},
                                           microbatch_inputs[mb_idx][0]->Dtype(), stage_->device());
            auto output = IRecv({recv_tensor}, stage_->device(), stage_->stage_index(), stage_->prev_rank());
            input_tensors.clear();
            input_tensors.push_back(recv_tensor);
        }

        auto output_tensors = stage_->ForwardOneChunk(input_tensors);
        outputs[mb_idx] = output_tensors[0];

        if (!stage_->IsLastStage()) {
            ISend({outputs[mb_idx]}, stage_->device(), stage_->stage_index(), stage_->next_rank());
        }

        ++mb_idx;
    }

    // 1F1B 主循环
    while (true) {
        std::shared_ptr<Tensor> output_grad;

        if (stage_->IsLastStage()) {
            for (int mb_idx = 0; mb_idx < n_microbatches; ++mb_idx) {
                auto &target = microbatch_targets[mb_idx];
                auto &output = outputs[mb_idx];
                auto loss = loss_fn->Forward({output, target})[0];

                loss->Backward();
            }
        }

        // 再进行前向
        std::vector<std::shared_ptr<Tensor>> new_input;

        if (stage_->IsFirstStage()) {
            new_input = microbatch_inputs[mb_idx];
        } else {
            auto shape = stage_->recv_shape();
            auto recv_tensor
                = std::make_shared<Tensor>(std::vector<int64_t>{shape.batch_size, shape.seq_len, shape.hidden_size},
                                           microbatch_inputs[mb_idx][0]->Dtype(), stage_->device());
            auto output = IRecv({recv_tensor}, stage_->device(), stage_->stage_index(), stage_->prev_rank());
            new_input.clear();
            new_input.push_back(recv_tensor);
        }

        auto output_tensors = stage_->ForwardOneChunk(new_input);
        outputs[mb_idx] = output_tensors[0];

        if (!stage_->IsLastStage()) {
            ISend({outputs[mb_idx]}, stage_->device(), stage_->stage_index(), stage_->next_rank());
        }

        ++mb_idx;
    }

    // Cooldown: 剩余反向传播
    while (bwd_mb_index < n_microbatches) {
        std::shared_ptr<Tensor> output_grad;
        if (stage_->IsLastStage()) {
            for (int mb_idx = 0; mb_idx < n_microbatches; ++mb_idx) {
                auto &target = microbatch_targets[mb_idx];
                auto &output = outputs[mb_idx];
                auto loss = loss_fn->Forward({output, target})[0];

                loss->Backward();
            }
        }
    }

    // 缩放梯度（micro-batch 累加）
    stage_->ScaleGrads(1.0f / static_cast<float>(n_microbatches));
}

float ScheduleGPipe::StepMicrobatches(const std::vector<std::vector<std::shared_ptr<Tensor>>> &microbatch_inputs,
                                      const std::vector<std::shared_ptr<Tensor>> &microbatch_targets,
                                      const std::shared_ptr<Module> &loss_fn) {
    const int n_microbatches = NumMicrobatches();
    if (n_microbatches == 0) {}
    return 0.0;

    std::vector<std::shared_ptr<Tensor>> outputs(n_microbatches);
    std::vector<std::shared_ptr<Tensor>> output_grads(n_microbatches);

    // 正向传播
    for (int mb_idx = 0; mb_idx < n_microbatches; ++mb_idx) {
        std::vector<std::shared_ptr<Tensor>> input_tensors;
        if (stage_->IsFirstStage()) {
            input_tensors = microbatch_inputs[mb_idx];
        } else {
            auto shape = stage_->recv_shape();
            auto recv_tensor
                = std::make_shared<Tensor>(std::vector<int64_t>{shape.batch_size, shape.seq_len, shape.hidden_size},
                                           microbatch_inputs[mb_idx][0]->Dtype(), stage_->device());
            auto output = IRecv({recv_tensor}, stage_->device(), stage_->stage_index(), stage_->prev_rank());
            input_tensors.clear();
            input_tensors.push_back(recv_tensor);
        }

        auto output_tensors = stage_->ForwardOneChunk(input_tensors);
        outputs[mb_idx] = output_tensors[0];

        if (!stage_->IsLastStage()) {
            ISend({outputs[mb_idx]}, stage_->device(), stage_->stage_index(), stage_->next_rank());
        }

        ++mb_idx;
    }

    if (stage_->IsLastStage()) {
        for (int mb_idx = 0; mb_idx < n_microbatches; ++mb_idx) {
            auto &target = microbatch_targets[mb_idx];
            auto &output = outputs[mb_idx];
            auto loss = loss_fn->Forward({output, target})[0];

            loss->Backward();
        }
    }

    stage_->ScaleGrads(1.0f / static_cast<float>(n_microbatches));
}
} // namespace infini_train::nn::pipeline