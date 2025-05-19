#include <fstream>
#include <vector>
#include "glog/logging.h"
#include "example/gpt2/tokenizer/tokenizer.h" 

const int MAGIC_NUMBER = 20240520;
const int VERSION = 1;

void ConvertTextToBin(const std::string& txt_path,
                      const std::string& bin_path,
                      size_t sequence_length,
                      const infini_train::GPT2Tokenizer& tokenizer) {
    std::ifstream ifs(txt_path);
    CHECK(ifs.is_open()) << "无法打开文件: " << txt_path;
    // std::cout<< "文件：" <<txt_path <<std::endl;
    const std::string text(
        (std::istreambuf_iterator<char>(ifs)),
        std::istreambuf_iterator<char>());
    std::cout << "text获取完成"  <<std::endl;
    const auto tokens = tokenizer.Encode(text);
    std::cout << "sequence_length" << tokens.size() <<std::endl;
    int i = 0;
    for (const auto& token : tokens) {
        if (i++ == 100)
            break;
        std::cout << token << std::endl;
    }
    // CHECK_GE(tokens.size(), sequence_length) 
    //     << "文本长度不足最小序列要求";
    
    std::vector<uint16_t> token_ids;
    token_ids.reserve(tokens.size());
    for (const auto& token : tokens) {
        CHECK_LE(token, 65535) << "Toekn溢出" << token;
        token_ids.push_back(static_cast<uint16_t>(token));
    }
    // std::cout<< "转换为指定类型" <<tokens.size() <<std::endl;

    const int32_t num_tokens = token_ids.size();
    const int32_t reserved = 0;
    // std::cout<< " 构造头部完成" <<std::endl;
    std::vector<uint8_t> header(1024, 0);
    std::memcpy(&header[0], &MAGIC_NUMBER, sizeof(MAGIC_NUMBER));
    std::memcpy(&header[4], &VERSION, sizeof(VERSION));
    std::memcpy(&header[8], &num_tokens, sizeof(num_tokens));
    std::memcpy(&header[12], &reserved, sizeof(reserved));

    std::ofstream ofs(bin_path, std::ios::binary);
    // CHECK(ofs.is_open()) << "无法创建文件: " << bin_path;
    
    ofs.write(reinterpret_cast<const char*>(header.data()), 1024);
    ofs.write(reinterpret_cast<const char*>(token_ids.data()), 
             token_ids.size() * sizeof(uint16_t));
}

int main() {
    const infini_train::GPT2Tokenizer tokenizer("/home/wanghaojie/jiyiming/InfiniTrain/example/gpt2/tokenizer/vocab.json",
        "/home/wanghaojie/jiyiming/InfiniTrain/example/gpt2/tokenizer/merges.txt");
    ConvertTextToBin(
        "/home/wanghaojie/jiyiming/InfiniTrain/example/gpt2/tokenizer/my_novel.txt", 
        "shakespeare_format.bin",
        100, 
        tokenizer
    );
    return 0;
}