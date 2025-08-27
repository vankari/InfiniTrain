// pipeline_parallel.cpp
#include "infini_train/include/nn/parallel/pp/pipeline_parallel.h"

#include "infini_train/include/nn/modules/container.h"
#include "infini_train/include/nn/modules/module.h"
#include "infini_train/include/nn/parallel/pp/pipeline_schedule.h"
#include "infini_train/include/nn/parallel/pp/pipeline_stage.h"
#include <memory>

namespace infini_train::nn::pipeline {
namespace {

std::vector<std::vector<std::shared_ptr<Module>>> SplitLayersIntoStages(std::vector<std::shared_ptr<Module>> layers,
                                                                        int num_stages) {
    const int total_layers = layers.size();
    CHECK_GT(total_layers, 0) << "Model has no layers to split!";
    CHECK_GE(num_stages, 1) << "num_stages must be >= 1";
    CHECK_LE(num_stages, total_layers) << "num_stages (" << num_stages << ") cannot be greater than total layers ("
                                       << total_layers << ")";

    std::vector<std::vector<std::shared_ptr<Module>>> stages(num_stages);

    int base_layers_per_stage = total_layers / num_stages; // 每个 stage 至少这么多层
    int remainder = total_layers % num_stages; // 如果不能平均分配，前 remainder 个 stage 需要 +1 层

    int layer_idx = 0;
    for (int s = 0; s < num_stages; ++s) {
        int layers_in_this_stage = base_layers_per_stage + (s < remainder ? 1 : 0);
        for (int i = 0; i < layers_in_this_stage; i++) {
            auto layer = layers[layer_idx];
            stages[s].emplace_back(layer);
            layer_idx++;
        }
    }

    return stages;
}

} // namespace
PipelineParallel::PipelineParallel(const std::shared_ptr<Module> &model, const std::vector<Device> &devices,
                                   std::shared_ptr<Optimizer> optimizer, int num_gpus, int num_microbatches,
                                   const int batch_size, const int seq_len, const int hidden_size)
    : original_model_(model), devices_(devices), optimizer_(std::move(optimizer)), num_stages_(num_gpus) {
    CHECK(!devices_.empty()) << "Devices list is empty";

    SplitModel(batch_size, seq_len, hidden_size); // 执行模型切分，生成 split_stages_
    SetupSchedules(num_microbatches);             // 生成 schedules_
}

void PipelineParallel::SplitModel(const int batch_size, const int seq_len, const int hidden_size) {
    auto layers = original_model_->GetPipelineLayers();
    auto stage_layers = SplitLayersIntoStages(layers, num_stages_);

    for (int i = 0; i < num_stages_; i++) {
        for (auto layer : stage_layers[i]) { layer->To(&devices_[i]); }
    }

    ActivationShape recv_shape{.batch_size = batch_size, .seq_len = seq_len, .hidden_size = hidden_size};

    for (int s = 0; s < num_stages_; ++s) {
        auto stage = std::make_shared<PipelineStage>(stage_layers[s], s, num_stages_, optimizer_, recv_shape);
        pipeline_stages_.push_back(stage);
    }
}

void PipelineParallel::SetupSchedules(int num_microbatches) {
    for (int stage_idx = 0; stage_idx < world_size_; ++stage_idx) {
        auto schedule
            = std::make_shared<ScheduleGPipe>(pipeline_stages_[stage_idx], num_stages_, num_microbatches, stage_idx);
        schedules_.push_back(schedule);
    }
}

float PipelineParallel::TrainStep(const std::vector<std::shared_ptr<Tensor>> &input,
                                  const std::vector<std::shared_ptr<Tensor>> &target,
                                  const std::shared_ptr<Module> &loss_fn) {

    std::vector<std::vector<float>> local_losses(num_stages_);
    std::vector<std::thread> stage_threads;
    stage_threads.reserve(num_stages_);

    for (int si = 0; si < num_stages_; ++si) {
        auto schedule = schedules_[si];

        stage_threads.emplace_back([si, schedule, input, target, loss_fn, &local_losses, this]() {
            devices_[si].SetDevice();

            std::vector<std::shared_ptr<Tensor>> stage_input;
            if (si == 0) {
                stage_input = input;
            }

            // 调度器内部完成：Forward + loss 计算 + Backward
            auto stage_losses = schedule->Step(stage_input, target, loss_fn);

            // // 只有最后一stage 有 loss
            // if (si == num_stages_ - 1 && !stage_losses.empty()) {
            //     auto& loss_vec = local_losses[si];  // 引用本地 vector
            //     loss_vec.reserve(stage_losses.size());

            //     // 将每个 loss 张量转为 float 值并存入
            //     for (const auto& loss : stage_losses) {
            //         auto loss_cpu = loss->To(DeviceManager::Instance()->GetDefaultDevice());
            //         float value = static_cast<const float *>(loss_cpu.DataPtr())[0];
            //         loss_vec.push_back(value);
            //     }
            // }
        });
    }

    for (auto &t : stage_threads) {
        if (t.joinable()) {
            t.join();
        }
    }

    float total_loss = 0.0f;
    int total_count = 0;
    for (const auto &per_stage_losses : local_losses) {
        for (float loss_val : per_stage_losses) {
            total_loss += loss_val;
            total_count++;
        }
    }

    if (total_count > 0) {
        return total_loss / total_count; // 返回平均 loss
    }
    LOG(FATAL) << "No loss collected";
    return 0.0f;
}

} // namespace infini_train::nn::pipeline