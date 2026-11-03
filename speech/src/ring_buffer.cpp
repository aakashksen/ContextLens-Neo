#include "ring_buffer.h"

#include <stdexcept>

RingBuffer::RingBuffer(size_t capacity) : buf_(capacity) {
  if (capacity == 0) throw std::invalid_argument("RingBuffer capacity must be > 0");
}

void RingBuffer::push(const float* data, size_t n) {
  for (size_t i = 0; i < n; ++i) {
    size_t tail = (head_ + size_) % buf_.size();
    buf_[tail] = data[i];
    if (size_ == buf_.size()) {
      head_ = (head_ + 1) % buf_.size();
    } else {
      ++size_;
    }
  }
}

std::vector<float> RingBuffer::snapshot() const {
  std::vector<float> out;
  out.reserve(size_);
  for (size_t i = 0; i < size_; ++i) out.push_back(buf_[(head_ + i) % buf_.size()]);
  return out;
}

void RingBuffer::clear() {
  head_ = 0;
  size_ = 0;
}
