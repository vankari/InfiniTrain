#pragma once

#include <vector>

#include "infini_train/include/op.h"
#include "infini_train/include/tensor.h"

namespace infini_train {
class Network {
public:
    explicit Network(const std::vector<Tensor *> input_tensors);
    virtual ~Network() = default;

    void AddOutputTensor(Tensor *tensor);

    std::vector<Tensor> &AddLayer(std::unique_ptr<Op> op);

    std::vector<Tensor *> Parameters() const;

    std::vector<Tensor *> &InputTensors();
    const std::vector<Tensor *> &InputTensors() const;

    std::vector<Tensor *> &OutputTensors();
    const std::vector<Tensor *> &OutputTensors() const;

    void Forward();

protected:
    std::vector<Tensor *> input_tensors_;
    std::vector<Tensor *> output_tensors_;
    std::vector<std::unique_ptr<Op>> layers_;
};

namespace loss {
class CrossEntropyLoss : public Network {
public:
    explicit CrossEntropyLoss(const std::vector<Tensor *> &input_tensors);
};
} // namespace loss
} // namespace infini_train
