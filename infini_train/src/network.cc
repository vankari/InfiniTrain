#include "infini_train/include/network.h"

#include <vector>

#include "infini_train/include/tensor.h"

namespace infini_train {
Network::Network(const std::vector<Tensor *> input_tensors)
    : input_tensors_(input_tensors) {}

void Network::AddOutputTensor(Tensor *tensor) {
    output_tensors_.emplace_back(tensor);
}

std::vector<Tensor> &Network::AddLayer(std::unique_ptr<Op> op) {
    layers_.emplace_back(std::move(op));
    auto &output_tensors = layers_.back()->OutputTensors();
    for (auto &tensor : output_tensors) {
        tensor.SetProducer(layers_.back().get());
    }
    return output_tensors;
}

std::vector<Tensor *> Network::Parameters() const {
    std::vector<Tensor *> params;
    for (auto &layer : layers_) {
        for (auto &tensor : layer->Weights()) {
            params.push_back(&tensor);
        }
    }
    return params;
}

std::vector<Tensor *> &Network::InputTensors() { return input_tensors_; }

const std::vector<Tensor *> &Network::InputTensors() const {
    return input_tensors_;
}

std::vector<Tensor *> &Network::OutputTensors() { return output_tensors_; }

const std::vector<Tensor *> &Network::OutputTensors() const {
    return output_tensors_;
}

void Network::Forward() {
    for (auto &layer : layers_) {
        layer->Forward();
    }
}

namespace loss {
CrossEntropyLoss::CrossEntropyLoss(const std::vector<Tensor *> &input_tensors)
    : Network(input_tensors) {
    CHECK_EQ(input_tensors.size(), 2);
    auto *input_tensor = input_tensors_.at(0);
    auto *target_tensor = input_tensors_.at(1);
    auto cross_entropy = std::make_unique<ops::CrossEntropy>(
        std::vector<Tensor *>{input_tensor, target_tensor});
    auto &x1 = AddLayer(std::move(cross_entropy));
    AddOutputTensor(&x1.at(0));
}
} // namespace loss
} // namespace infini_train
