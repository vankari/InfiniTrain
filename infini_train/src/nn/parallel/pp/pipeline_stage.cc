#include "infini_train/include/nn/parallel/pp/pipeline_stage.h"

#include "glog/logging.h"

#include <memory>

#include "infini_train/include/device.h"
#include "infini_train/include/dispatcher.h"

namespace infini_train::nn::pipeline {

PipelineStage::PipelineStage(std::vector<std::shared_ptr<Module>> layers, int stage_index, int num_stages,
                             std::shared_ptr<Optimizer> opt, const ActivationShape &recvShape)
    : stage_index_(stage_index), num_stages_(num_stages), layers_(layers),
      prev_rank_(stage_index > 0 ? stage_index - 1 : -1),
      next_rank_(stage_index < num_stages - 1 ? stage_index + 1 : -1), optimizer_(opt),
      devices_(DeviceManager::Instance()->GetAllAvailableDevices(DeviceType::kCUDA)), device_(devices_.at(stage_index)),
      recv_shape_(recvShape) {}

std::vector<std::shared_ptr<Tensor>>
PipelineStage::ForwardOneChunk(const std::vector<std::shared_ptr<Tensor>> &inputs) {
    std::vector<std::shared_ptr<Tensor>> current = inputs;

    forward_outputs_.clear();

    for (const auto &layer : layers_) {
        auto outputs = layer->Forward(current);
        current = outputs;
    }

    return current;
}

} // namespace infini_train::nn::pipeline