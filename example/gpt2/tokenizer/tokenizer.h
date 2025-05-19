#pragma once
#include <string>
#include <vector>
#include <unordered_map>
#include "infini_train/include/tensor.h"
#include "infini_train/include/device.h"


#include "example/gpt2/tokenizer/tokenizer.h"
#include <fstream>
#include <sstream>
#include <algorithm>

namespace infini_train {

class GPT2Tokenizer {
public:
    
    explicit GPT2Tokenizer(const std::string& vocab_path, 
                        const std::string& merges_path);
    
    std::vector<int> Encode(const std::string& text) const;
    
    std::string Decode(const std::vector<int>& tokens) const;

    std::vector<std::string> byteEncode(const std::string& text) const;

private:
    std::vector<std::string> BytePairEncoding(const std::string& text) const;
    
    void LoadVocab(const std::string& path);
    
    void LoadMerges(const std::string& path);

    void InitializeByteEncoder();

    std::unordered_map<std::string, int> vocab_;
    std::vector<std::pair<std::string, std::string>> merges_;
    std::unordered_map<std::string, std::string> byte_encoder_;
};

}