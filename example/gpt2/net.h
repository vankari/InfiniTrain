#pragma once

#include <cstdint>
#include <cstdlib>
#include <memory>
#include <vector>

#include "glog/logging.h"

#include "infini_train/include/device.h"
#include "infini_train/include/nn/modules/module.h"
#include "infini_train/include/tensor.h"

struct GPT2Config {
    int64_t block_size = 1024;
    int64_t vocab_size = 50257;
    int64_t n_layer = 12;
    int64_t n_head = 12;
    int64_t n_embd = 768;
};

class NewGELU : public infini_train::nn::CloneableModule<NewGELU> {
public:
    std::vector<std::shared_ptr<infini_train::Tensor>>
    Forward(const std::vector<std::shared_ptr<infini_train::Tensor>> &x) override;
};

class CausalSelfAttention : public infini_train::nn::CloneableModule<CausalSelfAttention> {
public:
    static constexpr char kCAttnLayerName[] = "c_attn";
    static constexpr char kCProjLayerName[] = "c_proj";

    static constexpr char kParamBiasName[] = "bias";

    explicit CausalSelfAttention(const GPT2Config &config);

    std::vector<std::shared_ptr<infini_train::Tensor>>
    Forward(const std::vector<std::shared_ptr<infini_train::Tensor>> &x) override;

private:
    GPT2Config config_;
    int64_t n_head_ = 0;
    int64_t n_embd_ = 0;
};

class MLP : public infini_train::nn::CloneableModule<MLP> {
public:
    static constexpr char kCFclayerName[] = "c_fc";
    static constexpr char kGeluLayerName[] = "gelu";
    static constexpr char kCProjLayerName[] = "c_proj";

    explicit MLP(const GPT2Config &config);

    std::vector<std::shared_ptr<infini_train::Tensor>>
    Forward(const std::vector<std::shared_ptr<infini_train::Tensor>> &x) override;
};

class Block : public infini_train::nn::CloneableModule<Block> {
public:
    static constexpr char kLn1LayerName[] = "ln_1";
    static constexpr char kAttnLayerName[] = "attn";
    static constexpr char kLn2LayerName[] = "ln_2";
    static constexpr char kMlpLayerName[] = "mlp";

    explicit Block(const GPT2Config &config);

    std::vector<std::shared_ptr<infini_train::Tensor>>
    Forward(const std::vector<std::shared_ptr<infini_train::Tensor>> &x) override;
};

class GPT2 : public infini_train::nn::CloneableModule<GPT2> {
public:
    static constexpr char kWTELayerName[] = "wte";
    static constexpr char kWPELayerName[] = "wpe";
    static constexpr char kHLayerName[] = "h";
    static constexpr char kLnFLayerName[] = "ln_f";
    static constexpr char kTransformerLayerName[] = "transformer";
    static constexpr char kLMHeadLayerName[] = "lm_head";

    enum class ModelType : int8_t {
        kGPT2,
        kGPT2Medium,
        kGPT2Large,
        kGPT2XL,
    };

    explicit GPT2(const GPT2Config &config);

    int GetHiddenSize() const;
    std::vector<std::shared_ptr<infini_train::nn::Module>> GetPipelineLayers() override;

    std::vector<std::shared_ptr<infini_train::Tensor>>
    Forward(const std::vector<std::shared_ptr<infini_train::Tensor>> &x) override;

    static std::shared_ptr<GPT2> FromPretrained(ModelType model_type);
    static std::shared_ptr<GPT2> FromLLMC(const std::string &filepath);

private:
    GPT2Config config_;
};
