#pragma once

#include <memory>
#include <vector>

#include "infini_train/include/nn/modules/module.h"
#include "infini_train/include/tensor.h"

namespace infini_train::nn {
class CrossEntropyLoss : public CloneableModule<CrossEntropyLoss> {
public:
    CrossEntropyLoss() = default;

    std::vector<std::shared_ptr<Tensor>> Forward(const std::vector<std::shared_ptr<Tensor>> &input_tensors) override;
};

class MSELoss : public CloneableModule<MSELoss> {
public:
    MSELoss() = default;

    std::vector<std::shared_ptr<Tensor>> Forward(const std::vector<std::shared_ptr<Tensor>> &input_tensors) override;
};
} // namespace infini_train::nn
