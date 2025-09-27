#pragma once

#include <memory>
#include <vector>

#include "infini_train/include/nn/modules/module.h"

namespace infini_train {
class Tensor;
class Device;
} // namespace infini_train

namespace infini_train::nn {
class LayerNorm : public CloneableModule<LayerNorm> {
public:
    static constexpr char kParamWeightName[] = "weight";
    static constexpr char kParamBiasName[] = "bias";

    LayerNorm(const std::vector<int64_t> &normalized_shape, float eps = 1e-5f, const Device *device = nullptr);
    std::vector<std::shared_ptr<Tensor>> Forward(const std::vector<std::shared_ptr<Tensor>> &input_tensors) override;

private:
    void ResetParameters();

    const float eps_ = 1e-5f;
};
} // namespace infini_train::nn
