#pragma once

#include <memory>
#include <unordered_map>
#include <vector>

#include "infini_train/include/nn/modules/module.h"

namespace infini_train {
class Tensor;
}

namespace infini_train::nn {
class Sequential : public CloneableModule<Sequential> {
public:
    // TODO(dcj): Use better ctor signature later.
    explicit Sequential(std::vector<std::shared_ptr<Module>> &&layers);

    std::vector<std::shared_ptr<Tensor>> Forward(const std::vector<std::shared_ptr<Tensor>> &input_tensors) override;
};

class ModuleDict : public CloneableModule<ModuleDict> {
public:
    // TODO(dcj): in torch, there is a dict with the order of insertion
    explicit ModuleDict(std::unordered_map<std::string, std::shared_ptr<Module>> modules);

    std::vector<std::shared_ptr<Tensor>> Forward(const std::vector<std::shared_ptr<Tensor>> &input_tensors) override;
};
} // namespace infini_train::nn
