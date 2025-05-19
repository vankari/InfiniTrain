#include <fstream>
#include <sstream>
#include <algorithm>
#include "infini_train/include/device.h"
#include "example/gpt2/tokenizer/tokenizer.h"
#include "example/gpt2/tokenizer/json.hpp"

namespace infini_train {

GPT2Tokenizer::GPT2Tokenizer(const std::string& vocab_file, 
                    const std::string& merges_file) {
    LoadVocab(vocab_file);
    LoadMerges(merges_file);
    InitializeByteEncoder();
}

std::vector<int> GPT2Tokenizer::Encode(const std::string& text) const {
    std::vector<std::string> tokens = byteEncode(text);
    
    // std::cout<< "应用合并规则:" <<std::endl;
    for (const auto& merge : merges_) {
        for (size_t i = 0; i < tokens.size() - 1; ) {
            if (tokens[i] == merge.first && tokens[i+1] == merge.second) {
                tokens[i] = merge.first + merge.second;
                tokens.erase(tokens.begin() + i + 1);
            } else {
                ++i;
            }
        }
    }
    
    std::vector<int> token_ids;
    std::cout << vocab_.size() << std::endl;
    for (const auto& token : tokens) {
        // std::cout<< "转换为IDs:" << token <<std::endl;
        if (vocab_.count(token)) {
            // std::cout << "ID: " << vocab_.at(token) << std::endl;
            token_ids.push_back(vocab_.at(token));
        } else {
            // std::cout << "找不到的ID: " << token<< std::endl;
            token_ids.push_back(vocab_.at("unk"));
        }
    }
    return token_ids;
}

void GPT2Tokenizer::LoadVocab(const std::string& vocab_file) {
    std::ifstream file(vocab_file);
    CHECK(file.is_open()) << "open vocab file failed" << vocab_file;

    nlohmann::json vocab_json;
    file >> vocab_json;

    for (auto& [token, id] : vocab_json.items()) {
        vocab_[token] = id.is_string() ? std::stoi(id.get<std::string>()) : id.get<int>();
    }

    // 验证必要token
    // std::cout << "LoadVocab OK，词汇表大小: " << vocab_.size() << std::endl;
    CHECK(vocab_.count("unk")) << "vocab must contain unk";
}

void GPT2Tokenizer::LoadMerges(const std::string& merges_file) {
    std::ifstream file(merges_file);
    std::string line;
    while (std::getline(file, line)) {
        if (line.empty() || line.starts_with("#")) continue;
        auto pos = line.find(' ');
        merges_.emplace_back(line.substr(0, pos), line.substr(pos+1));
    }
    std::cout<<"LoadMerges OK"<<std::endl;
}

void GPT2Tokenizer::InitializeByteEncoder() {
    for (int i=0; i<256; ++i) {
        const unsigned char c = static_cast<unsigned char>(i);
        const std::string byte_str(1, c);
        byte_encoder_[byte_str] = byte_str;
    }
    // 空格处理
    byte_encoder_[" "] = "Ġ";
}

std::vector<std::string> GPT2Tokenizer::byteEncode(const std::string& text) const {
    std::vector<std::string> tokens;
    tokens.reserve(text.size());
    // std::cout << "byte_encoder_大小: " << byte_encoder_.size() << std::endl;
    for (uint8_t byte : text) {
        // 构造单字符字符串（兼容所有字节值）
        const std::string key(1, static_cast<char>(byte));
        const auto it = byte_encoder_.find(key);
        if (it != byte_encoder_.end()) {
            tokens.push_back(it->second);
        } else {
            tokens.push_back("unk");
        }
    }
    return tokens;
}
}