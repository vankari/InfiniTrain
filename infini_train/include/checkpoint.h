#pragma once  
  
#include <string>  
#include <tuple>  
#include <fstream>  
#include <unordered_map>  
#include <memory>  
  
#include "infini_train/include/nn/modules/module.h"  
#include "infini_train/include/optimizer.h"  
#include "infini_train/include/tensor.h"  
  
namespace infini_train {  
  
class Checkpoint {  
public:  
    static void Save(const std::string& path, const nn::Module& model,  
                    const Optimizer& optimizer, int global_step,   
                    float best_loss, float last_lr);  
      
    static std::tuple<int, float, float> Load(const std::string& path,  
                                             nn::Module& model,  
                                             Optimizer& optimizer);  
  
private:  
    // 保存相关的辅助函数  
    static void SaveModelBin(const std::string& filepath, const nn::Module& model);  
    static void SaveOptimizerBin(const std::string& filepath, const Optimizer& optimizer);  
    static void SaveTrainerState(const std::string& filepath, int global_step,   
                                float best_loss, float last_lr);  
      
    // 模型权重写入函数  
    static void WriteHeader(std::ofstream& ofs, const nn::Module& model);  
    static void WriteWeightsInOrder(std::ofstream& ofs,   
                                   const std::unordered_map<std::string, std::shared_ptr<Tensor>>& state_dict,  
                                   const nn::Module& model);  
    static void WriteGPT2Weights(std::ofstream& ofs,   
                                const std::unordered_map<std::string, std::shared_ptr<Tensor>>& state_dict);  
    static void WriteLLaMA3Weights(std::ofstream& ofs,  
                                  const std::unordered_map<std::string, std::shared_ptr<Tensor>>& state_dict);  
    static void WriteTensorToFile(std::ofstream& ofs, const std::shared_ptr<Tensor>& tensor);  
      
    // 优化器状态写入函数  
    static void SaveAdamState(std::ofstream& ofs, const optimizers::Adam& adam);  
    static void SaveSGDState(std::ofstream& ofs, const optimizers::SGD& sgd);  
     // 加载相关的辅助函数  
    static std::tuple<int, float, float> LoadTrainerState(const std::string& filepath);  
    static void LoadModelBin(const std::string& filepath, nn::Module& model);  
    static void LoadOptimizerBin(const std::string& filepath, Optimizer& optimizer);  
      
    // 模型权重读取函数  
    static void ValidateHeader(std::ifstream& ifs, const nn::Module& model);  
    static void LoadWeightsInOrder(std::ifstream& ifs,   
                                  const std::unordered_map<std::string, std::shared_ptr<Tensor>>& state_dict,  
                                  const nn::Module& model);  
    static void LoadGPT2Weights(std::ifstream& ifs,  
                               const std::unordered_map<std::string, std::shared_ptr<Tensor>>& state_dict);  
    static void LoadLLaMA3Weights(std::ifstream& ifs,  
                                 const std::unordered_map<std::string, std::shared_ptr<Tensor>>& state_dict);  
    static void LoadTensorFromFile(std::ifstream& ifs, const std::shared_ptr<Tensor>& tensor);  
      
    // 优化器状态读取函数  
    static void LoadAdamState(std::ifstream& ifs, optimizers::Adam& adam);  
    static void LoadSGDState(std::ifstream& ifs, optimizers::SGD& sgd);  
      
    // 辅助函数  
    static int GetModelLayerCount(const std::unordered_map<std::string, std::shared_ptr<Tensor>>& state_dict);
};  

  
} // namespace infini_train