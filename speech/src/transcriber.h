#pragma once
#include <mutex>
#include <string>
#include <vector>

#include "vad.h"

struct whisper_context;

struct Segment {
  std::string text;
  int64_t start_ms;
  int64_t end_ms;
};

// One whisper.cpp model shared by all streams; calls are serialised.
class Transcriber {
 public:
  explicit Transcriber(const std::string& model_path);
  ~Transcriber();
  Transcriber(const Transcriber&) = delete;
  Transcriber& operator=(const Transcriber&) = delete;

  // "cpu", or "gpu:<device name>" when a GPU backend is loaded and in use.
  const std::string& backend() const { return backend_; }
  // Segment times are relative to the whole stream (the utterance start is added).
  std::vector<Segment> transcribe(const Utterance& u);

 private:
  whisper_context* ctx_ = nullptr;
  std::string backend_;
  std::mutex mu_;
};
