#pragma once

#include <cmath>
#include <cstdint>
#include <vector>

#include "glog/logging.h"

#include "infini_train/include/tensor.h"

namespace infini_train {
class Op : public AllowBackward {
public:
    explicit Op(const std::vector<Tensor *> input_tensors);
    virtual ~Op() = default;

    void OutputTensorsUseGradient();

    virtual void Forward() = 0;
    virtual void BackwardImpl() = 0;
    void Backward() override;

    void AddWeight(const std::vector<int64_t> &dims, const DataType dtype);

    std::vector<Tensor> &Weights();
    const std::vector<Tensor> &Weights() const;

    std::vector<Tensor> &OutputTensors();
    const std::vector<Tensor> &OutputTensors() const;

protected:
    std::vector<Tensor *> input_tensors_; // not owned
    std::vector<Tensor> output_tensors_;
    std::vector<Tensor> weights_;
};

namespace ops {
/*
  x: [bs, in_dim]
  -> Linear (w: [in_dim, out_dim], b: [out_dim])
  -> o: [bs, out_dim]
 */
class Linear : public Op {
public:
    Linear(const std::vector<Tensor *> &input_tensors, int64_t out_dim);

    void Forward() override;

    void BackwardImpl() override;

private:
    int64_t bs_ = 0;
    int64_t in_dim_ = 0;
    int64_t out_dim_ = 0;
};

/*
  -> Sigmoid
  -> input: [bs, out_dim]
  -> output: [bs, out_dim]
 */
class Sigmoid : public Op {
public:
    Sigmoid(const std::vector<Tensor *> input_tensors);

    void Forward() override;

    void BackwardImpl() override;

private:
    int64_t bs_ = 0;
    int64_t out_dim_ = 0;
};

class CrossEntropy : public Op {
public:
    CrossEntropy(const std::vector<Tensor *> input_tensors);

    void Forward() override;

    void BackwardImpl() override;

private:
    int64_t bs_ = 0;
    int64_t num_classes_ = 0;
};
} // namespace ops
} // namespace infini_train
