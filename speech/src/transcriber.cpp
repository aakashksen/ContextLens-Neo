#include "transcriber.h"

#include <algorithm>
#include <stdexcept>
#include <thread>

#include "ggml-backend.h"
#include "whisper.h"

namespace {

std::string detect_backend() {
  for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
    ggml_backend_dev_t dev = ggml_backend_dev_get(i);
    if (ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_GPU) {
      return std::string("gpu:") + ggml_backend_dev_name(dev);
    }
  }
  return "cpu";
}

std::string trim(const std::string& s) {
  size_t a = s.find_first_not_of(" \t\r\n");
  if (a == std::string::npos) return "";
  size_t b = s.find_last_not_of(" \t\r\n");
  return s.substr(a, b - a + 1);
}

}  // namespace

Transcriber::Transcriber(const std::string& model_path) {
  whisper_context_params cp = whisper_context_default_params();
  cp.use_gpu = true;  // falls back to CPU by itself when no GPU backend is present
  ctx_ = whisper_init_from_file_with_params(model_path.c_str(), cp);
  if (!ctx_) throw std::runtime_error("cannot load whisper model: " + model_path);
  backend_ = detect_backend();
}

Transcriber::~Transcriber() {
  if (ctx_) whisper_free(ctx_);
}

std::vector<Segment> Transcriber::transcribe(const Utterance& u) {
  std::vector<float> samples = u.samples;
  // whisper.cpp refuses clips shorter than one second.
  const size_t min_samples = 16000 * 11 / 10;
  if (samples.size() < min_samples) samples.resize(min_samples, 0.0f);

  whisper_full_params p = whisper_full_default_params(WHISPER_SAMPLING_GREEDY);
  p.language = "en";
  p.print_progress = false;
  p.print_realtime = false;
  p.print_special = false;
  p.print_timestamps = false;
  p.n_threads = static_cast<int>(std::max(1u, std::min(4u, std::thread::hardware_concurrency())));

  std::lock_guard<std::mutex> lock(mu_);
  if (whisper_full(ctx_, p, samples.data(), static_cast<int>(samples.size())) != 0) {
    throw std::runtime_error("whisper_full failed");
  }
  std::vector<Segment> out;
  int n = whisper_full_n_segments(ctx_);
  for (int i = 0; i < n; ++i) {
    std::string text = trim(whisper_full_get_segment_text(ctx_, i));
    if (text.empty() || text[0] == '[' || text[0] == '(') continue;  // [BLANK_AUDIO], (music), ...
    // whisper times are in units of 10 ms
    int64_t t0 = whisper_full_get_segment_t0(ctx_, i) * 10;
    int64_t t1 = whisper_full_get_segment_t1(ctx_, i) * 10;
    int64_t dur = u.end_ms - u.start_ms;
    out.push_back({text, u.start_ms + std::min(t0, dur), u.start_ms + std::min(t1, dur)});
  }
  return out;
}
