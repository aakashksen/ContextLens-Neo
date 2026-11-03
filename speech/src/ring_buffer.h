#pragma once
#include <cstddef>
#include <vector>

// Fixed-capacity buffer of samples. Pushing past capacity drops the oldest samples.
class RingBuffer {
 public:
  explicit RingBuffer(size_t capacity);

  void push(const float* data, size_t n);
  size_t size() const { return size_; }
  size_t capacity() const { return buf_.size(); }
  // Contents, oldest first.
  std::vector<float> snapshot() const;
  void clear();

 private:
  std::vector<float> buf_;
  size_t head_ = 0;  // index of the oldest sample
  size_t size_ = 0;
};
