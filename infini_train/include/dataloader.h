#pragma once

#include <cstddef>
#include <memory>
#include <utility>

#include "infini_train/include/dataset.h"
#include "infini_train/include/tensor.h"

namespace infini_train {
class DataLoaderIterator {
public:
    DataLoaderIterator(const Dataset &dataset, size_t batch_size,
                       size_t batch_idx, size_t max_batch_idx,
                       Tensor *data_tensor, Tensor *label_tensor);

    std::pair<Tensor *, Tensor *> operator*() const;

    DataLoaderIterator &operator++();
    DataLoaderIterator operator++(int);

    friend bool operator<(const DataLoaderIterator &lhs,
                          const DataLoaderIterator &rhs);
    friend bool operator!=(const DataLoaderIterator &lhs,
                           const DataLoaderIterator &rhs);
    friend bool operator==(const DataLoaderIterator &lhs,
                           const DataLoaderIterator &rhs);

private:
    const Dataset *dataset_ = nullptr; // not owned
    size_t batch_size_ = 0;
    size_t batch_idx_ = 0;
    size_t max_batch_idx_ = 0;
    Tensor *data_tensor_ = nullptr;
    Tensor *label_tensor_ = nullptr;
};

class DataLoader {
public:
    DataLoader(const std::shared_ptr<Dataset> &dataset, size_t batch_size,
               Tensor *data_tensor, Tensor *label_tensor);

    DataLoaderIterator begin() const;
    DataLoaderIterator end() const;

private:
    std::shared_ptr<Dataset> dataset_;
    size_t batch_size_ = 0;
    size_t max_batch_idx_ = 0;
    Tensor *data_tensor_ = nullptr;
    Tensor *label_tensor_ = nullptr;
};
} // namespace infini_train
