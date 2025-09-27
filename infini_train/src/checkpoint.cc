#include "infini_train/include/checkpoint.h"  
#include <filesystem>  
#include <fstream>  
#include <nlohmann/json.hpp>  
#include "glog/logging.h"  
  
namespace infini_train {  
  
void Checkpoint::Save(const std::string& path, const nn::Module& model,   
                     const Optimizer& optimizer, int global_step,   
                     float best_loss, float last_lr) {  
    // 创建checkpoint目录  
    std::filesystem::create_directories(path);  
      
    // 1. 保存模型权重 (model.bin)  
    SaveModelBin(path + "/model.bin", model);  
      
    // 2. 保存优化器状态 (optimizer.bin)    
    SaveOptimizerBin(path + "/optimizer.bin", optimizer);  
      
    // 3. 保存训练状态 (trainer_state.json)  
    SaveTrainerState(path + "/trainer_state.json", global_step, best_loss, last_lr);  
      
    LOG(INFO) << "Checkpoint saved to: " << path;  
}  
void Checkpoint::Load(const std::string& path,  
                  nn::Module& model,  
                  Optimizer& optimizer) {  
    // 检查checkpoint目录是否存在  
    CHECK(std::filesystem::exists(path)) << "Checkpoint directory does not exist: " << path;  
      
    // 1. 加载模型权重 (model.bin)  
    LoadModelBin(path + "/model.bin", model);  
      
    // 2. 加载优化器状态 (optimizer.bin)  
    LoadOptimizerBin(path + "/optimizer.bin", optimizer);  
      
    // 3. 加载训练状态 (trainer_state.json) 并返回相关信息  
    auto [global_step, best_loss, last_lr] = LoadTrainerState(path + "/trainer_state.json");  
      
    LOG(INFO) << "Checkpoint loaded from: " << path;  
    return std::make_tuple(global_step, best_loss, last_lr);  
}
void Checkpoint::SaveModelBin(const std::string& filepath, const nn::Module& model) {  
    std::ofstream ofs(filepath, std::ios::binary);  
    CHECK(ofs.is_open()) << "Failed to open file: " << filepath;  
      
    // 写入LLMC格式header  
    WriteHeader(ofs, model);  
      
    // 获取模型状态字典并按顺序写入权重  
    auto state_dict = model.StateDict();  
    WriteWeightsInOrder(ofs, state_dict, model);  
      
    ofs.close();  
    LOG(INFO) << "Model weights saved to: " << filepath;  
}  
  
void Checkpoint::WriteHeader(std::ofstream& ofs, const nn::Module& model) {  
    // 基于GPT2/LLaMA3的header格式实现  
    std::vector<int32_t> header(256, 0);  
      
    // 检测模型类型并设置相应的magic number和配置  
    if (model.type() == "GPT2") {  
        header[0] = 20240326; // kHeaderMagic from GPT2  
        header[1] = 3;        // kHeaderFP32Version  
        // 从模型中提取配置参数（需要扩展Module接口）  
        // header[2] = block_size; header[3] = vocab_size; 等等  
    } else if (model.type() == "LLaMA3") {  
        header[0] = 20240803; // kLLaMA3Magic  
        header[1] = 3;        // kLLaMA3FP32Version  
        // 设置LLaMA3特有的配置参数  
    }  
      
    // 写入256个int32_t的header  
    ofs.write(reinterpret_cast<const char*>(header.data()), 256 * sizeof(int32_t));  
}  
  
void Checkpoint::WriteWeightsInOrder(std::ofstream& ofs,   
                                   const std::unordered_map<std::string, std::shared_ptr<Tensor>>& state_dict,  
                                   const nn::Module& model) {  
    // 基于现有FromLLMC的读取顺序，实现相反的写入过程  
    if (model.type() == "GPT2") {  
        WriteGPT2Weights(ofs, state_dict);  
    } else if (model.type() == "LLaMA3") {  
        WriteLLaMA3Weights(ofs, state_dict);  
    }  
}  
  
void Checkpoint::WriteGPT2Weights(std::ofstream& ofs,   
                                const std::unordered_map<std::string, std::shared_ptr<Tensor>>& state_dict) {  
    // 按照GPT2::FromLLMC的读取顺序写入 [2](#4-1)   
      
    // transformer.wte.weight  
    WriteTensorToFile(ofs, state_dict.at("transformer.wte.weight"));  
      
    // transformer.wpe.weight    
    WriteTensorToFile(ofs, state_dict.at("transformer.wpe.weight"));  
      
    // 按层写入权重  
    // transformer.h.{i}.ln_1.weight, ln_1.bias, attn.c_attn.weight, 等等  
    // 参考example/gpt2/net.cc:298-388的顺序  
}  
  
void Checkpoint::WriteLLaMA3Weights(std::ofstream& ofs,  
                                  const std::unordered_map<std::string, std::shared_ptr<Tensor>>& state_dict) {  
    // 按照LLaMA3::FromLLMC的读取顺序写入 [3](#4-2)   
      
    // transformer.wte.weight  
    WriteTensorToFile(ofs, state_dict.at("transformer.wte.weight"));  
      
    // 按层写入权重，参考example/llama3/net.cc:431-491的顺序  
}  
  
void Checkpoint::WriteTensorToFile(std::ofstream& ofs, const std::shared_ptr<Tensor>& tensor) {  
    // 将tensor数据写入文件  
    const size_t num_bytes = tensor->SizeInBytes();  
      
    if (tensor->GetDevice()->Type() == DeviceType::kCPU) {  
        ofs.write(reinterpret_cast<const char*>(tensor->DataPtr()), num_bytes);  
    }  
#ifdef USE_CUDA  
    else if (tensor->GetDevice()->Type() == DeviceType::kCUDA) {  
        // 先拷贝到CPU再写入  
        auto cpu_tensor = tensor->To(DeviceManager::Instance()->GetDevice(DeviceType::kCPU));  
        ofs.write(reinterpret_cast<const char*>(cpu_tensor.DataPtr()), num_bytes);  
    }  
#endif  
}  
  
void Checkpoint::SaveOptimizerBin(const std::string& filepath, const Optimizer& optimizer) {  
    std::ofstream ofs(filepath, std::ios::binary);  
    CHECK(ofs.is_open()) << "Failed to open file: " << filepath;  
      
    // 写入优化器类型标识  
    if (auto adam = dynamic_cast<const optimizers::Adam*>(&optimizer)) {  
        SaveAdamState(ofs, *adam);  
    } else if (auto sgd = dynamic_cast<const optimizers::SGD*>(&optimizer)) {  
        SaveSGDState(ofs, *sgd);  
    }  
      
    ofs.close();  
    LOG(INFO) << "Optimizer state saved to: " << filepath;  
}  
  
void Checkpoint::SaveAdamState(std::ofstream& ofs, const optimizers::Adam& adam) {  
    // 写入优化器类型标识  
    std::string optimizer_type = "Adam";  
    uint32_t type_len = optimizer_type.length();  
    ofs.write(reinterpret_cast<const char*>(&type_len), sizeof(type_len));  
    ofs.write(optimizer_type.c_str(), type_len);  
      
    // 写入Adam参数  
    ofs.write(reinterpret_cast<const char*>(&adam.t_), sizeof(adam.t_));  
    ofs.write(reinterpret_cast<const char*>(&adam.learning_rate_), sizeof(adam.learning_rate_));  
    ofs.write(reinterpret_cast<const char*>(&adam.beta1_), sizeof(adam.beta1_));  
    ofs.write(reinterpret_cast<const char*>(&adam.beta2_), sizeof(adam.beta2_));  
    ofs.write(reinterpret_cast<const char*>(&adam.eps_), sizeof(adam.eps_));  
      
    // 写入momentum和variance状态 [4](#4-3)   
    uint32_t num_params = adam.m_.size();  
    ofs.write(reinterpret_cast<const char*>(&num_params), sizeof(num_params));  
      
    for (size_t i = 0; i < adam.m_.size(); ++i) {  
        WriteTensorToFile(ofs, adam.m_[i]);  
        WriteTensorToFile(ofs, adam.v_[i]);  
    }  
}  
  
void Checkpoint::SaveSGDState(std::ofstream& ofs, const optimizers::SGD& sgd) {  
    // SGD状态相对简单，只需保存learning_rate  
    std::string optimizer_type = "SGD";  
    uint32_t type_len = optimizer_type.length();  
    ofs.write(reinterpret_cast<const char*>(&type_len), sizeof(type_len));  
    ofs.write(optimizer_type.c_str(), type_len);  
      
    ofs.write(reinterpret_cast<const char*>(&sgd.learning_rate_), sizeof(sgd.learning_rate_));  
}  
  
void Checkpoint::SaveTrainerState(const std::string& filepath, int global_step,   
                                float best_loss, float last_lr) {  
    nlohmann::json state;  
    state["global_step"] = global_step;  
    state["best_loss"] = best_loss;  
    state["last_lr"] = last_lr;  
    state["timestamp"] = std::time(nullptr);  
      
    std::ofstream ofs(filepath);  
    CHECK(ofs.is_open()) << "Failed to open file: " << filepath;  
    ofs << state.dump(4);  
    ofs.close();  
      
    LOG(INFO) << "Training state saved to: " << filepath;  
}  
void Checkpoint::LoadOptimizerBin(const std::string& filepath, Optimizer& optimizer) {  
    std::ifstream ifs(filepath, std::ios::binary);  
    CHECK(ifs.is_open()) << "Failed to open optimizer file: " << filepath;  
      
    // 读取优化器类型  
    uint32_t type_len;  
    ifs.read(reinterpret_cast<char*>(&type_len), sizeof(type_len));  
      
    std::string optimizer_type(type_len, '\0');  
    ifs.read(optimizer_type.data(), type_len);  
      
    if (optimizer_type == "Adam") {  
        auto* adam = dynamic_cast<optimizers::Adam*>(&optimizer);  
        CHECK(adam != nullptr) << "Optimizer type mismatch: expected Adam";  
        LoadAdamState(ifs, *adam);  
    } else if (optimizer_type == "SGD") {  
        auto* sgd = dynamic_cast<optimizers::SGD*>(&optimizer);  
        CHECK(sgd != nullptr) << "Optimizer type mismatch: expected SGD";  
        LoadSGDState(ifs, *sgd);  
    }  
      
    ifs.close();  
    LOG(INFO) << "Optimizer state loaded from: " << filepath;  
}  
  
void Checkpoint::LoadAdamState(std::ifstream& ifs, optimizers::Adam& adam) {  
    // 读取Adam参数  
    ifs.read(reinterpret_cast<char*>(&adam.t_), sizeof(adam.t_));  
    ifs.read(reinterpret_cast<char*>(&adam.learning_rate_), sizeof(adam.learning_rate_));  
    ifs.read(reinterpret_cast<char*>(&adam.beta1_), sizeof(adam.beta1_));  
    ifs.read(reinterpret_cast<char*>(&adam.beta2_), sizeof(adam.beta2_));  
    ifs.read(reinterpret_cast<char*>(&adam.eps_), sizeof(adam.eps_));  
      
    // 读取momentum和variance状态  
    uint32_t num_params;  
    ifs.read(reinterpret_cast<char*>(&num_params), sizeof(num_params));  
      
    CHECK_EQ(num_params, adam.m_.size()) << "Parameter count mismatch";  
      
    for (size_t i = 0; i < adam.m_.size(); ++i) {  
        LoadTensorFromFile(ifs, adam.m_[i]);  
        LoadTensorFromFile(ifs, adam.v_[i]);  
    }  
}  
  
void Checkpoint::LoadSGDState(std::ifstream& ifs, optimizers::SGD& sgd) {  
    // SGD只需要读取learning_rate  
    ifs.read(reinterpret_cast<char*>(&sgd.learning_rate_), sizeof(sgd.learning_rate_));  
}  
  
// 辅助函数：从状态字典推断模型层数  
int Checkpoint::GetModelLayerCount(const std::unordered_map<std::string, std::shared_ptr<Tensor>>& state_dict) {  
    int max_layer = -1;  
    for (const auto& [key, tensor] : state_dict) {  
        if (key.find("transformer.h.") == 0) {  
            // 提取层号，例如从"transformer.h.5.ln_1.weight"中提取5  
            size_t start = key.find("transformer.h.") + 14;  
            size_t end = key.find(".", start);  
            if (end != std::string::npos) {  
                int layer_idx = std::stoi(key.substr(start, end - start));  
                max_layer = std::max(max_layer, layer_idx);  
            }  
        }  
    }  
    return max_layer + 1; // 层数 = 最大层索引 + 1  
}  
  
} // namespace infini_train