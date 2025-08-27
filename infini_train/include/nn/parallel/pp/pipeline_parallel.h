// pipeline_parallel.h
#pragma once

#include <memory>
#include <vector>

#include "infini_train/include/device.h"
#include "infini_train/include/nn/modules/module.h"
#include "infini_train/include/tensor.h"

#include "infini_train/include/nn/parallel/pp/pipeline_schedule.h"
#include "infini_train/include/nn/parallel/pp/pipeline_stage.h"

namespace infini_train::nn::pipeline {

class PipelineParallel : public Module {
public:
    PipelineParallel(const std::shared_ptr<Module> &model, const std::vector<Device> &devices,
                     std::shared_ptr<Optimizer> optimizer, int num_gpus, int num_microbatches, const int batch_size,
                     const int seq_len, const int hidden_size);

    float TrainStep(const std::vector<std::shared_ptr<Tensor>> &input,
                    const std::vector<std::shared_ptr<Tensor>> &target, const std::shared_ptr<Module> &loss_fn);

    std::vector<std::shared_ptr<Tensor>> Parameters() const override;

private:
    int world_size_;
    int num_stages_;
    int rank_;
    std::shared_ptr<Optimizer> optimizer_;
    std::vector<Device> devices_;
    std::shared_ptr<Module> original_model_;
    std::vector<std::shared_ptr<PipelineStage>> pipeline_stages_;
    std::vector<std::shared_ptr<PipelineSchedule>> schedules_;

    void SplitModel(const int batch_size, const int seq_len, const int hidden_size);
    void SetupSchedules(int num_microbatches);
};

} // namespace infini_train::nn::pipeline