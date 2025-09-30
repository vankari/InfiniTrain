#pragma once

#include <cstdint>
#include <memory>
#include <vector>
#include "infini_train/include/checkpoint.h" 
namespace infini_train {
class Tensor;
class Checkpoint;
}
namespace infini_train {
class Optimizer {
    
public:
    explicit Optimizer(const std::vector<std::shared_ptr<Tensor>> &params);

    void ZeroGrad();

    virtual void Step() = 0;

protected:
    std::vector<std::shared_ptr<Tensor>> params_;
};

namespace optimizers {
class SGD : public Optimizer {
    friend class infini_train::Checkpoint;
public:
    SGD(const std::vector<std::shared_ptr<Tensor>> &params, float learning_rate);

    void Step() override;

private:
    float learning_rate_ = 0.0;
};

class Adam : public Optimizer {
    friend class infini_train::Checkpoint;
public:
    Adam(const std::vector<std::shared_ptr<Tensor>> &params, float learning_rate = 1e-3, float beta1 = 0.9,
         float beta2 = 0.999, float eps = 1e-8);

    void Step() override;

private:
    int64_t t_;
    float learning_rate_;
    float beta1_;
    float beta2_;
    float eps_;
    std::vector<std::shared_ptr<Tensor>> m_;
    std::vector<std::shared_ptr<Tensor>> v_;
};
} // namespace optimizers
} // namespace infini_train
