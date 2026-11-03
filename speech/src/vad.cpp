#include "vad.h"

#include <cmath>

float VoiceGate::rms(const float* frame, size_t n) {
  if (n == 0) return 0.0f;
  double sum = 0.0;
  for (size_t i = 0; i < n; ++i) sum += static_cast<double>(frame[i]) * frame[i];
  return static_cast<float>(std::sqrt(sum / static_cast<double>(n)));
}

bool VoiceGate::update(const float* frame, size_t n) {
  last_loud_ = rms(frame, n) >= threshold_;
  if (last_loud_) {
    quiet_run_ = 0;
  } else if (quiet_run_ < (1 << 30)) {
    ++quiet_run_;
  }
  return quiet_run_ <= hangover_frames_;
}

Segmenter::Segmenter(const SegmenterConfig& cfg)
    : cfg_(cfg),
      frame_len_(static_cast<size_t>(cfg.sample_rate) * cfg.frame_ms / 1000),
      gate_(cfg.threshold, cfg.hangover_ms / cfg.frame_ms),
      preroll_(static_cast<size_t>(cfg.sample_rate) * cfg.preroll_ms / 1000) {}

std::vector<Utterance> Segmenter::feed(const float* samples, size_t n) {
  std::vector<Utterance> out;
  pending_.insert(pending_.end(), samples, samples + n);
  size_t pos = 0;
  while (pending_.size() - pos >= frame_len_) {
    process_frame(pending_.data() + pos, out);
    pos += frame_len_;
  }
  pending_.erase(pending_.begin(), pending_.begin() + static_cast<std::ptrdiff_t>(pos));
  return out;
}

std::vector<Utterance> Segmenter::flush() {
  std::vector<Utterance> out;
  if (in_utt_) close(out);
  pending_.clear();
  return out;
}

void Segmenter::process_frame(const float* frame, std::vector<Utterance>& out) {
  bool open = gate_.update(frame, frame_len_);
  if (open) {
    if (!in_utt_) {
      in_utt_ = true;
      utt_ = preroll_.snapshot();
      utt_start_sample_ = frames_done_ * static_cast<int64_t>(frame_len_) -
                          static_cast<int64_t>(utt_.size());
      loud_samples_ = 0;
      preroll_.clear();
    }
    utt_.insert(utt_.end(), frame, frame + frame_len_);
    if (gate_.last_frame_loud()) loud_samples_ += frame_len_;
    if (utt_.size() >= static_cast<size_t>(cfg_.sample_rate) * cfg_.max_utterance_ms / 1000) {
      close(out);
    }
  } else {
    if (in_utt_) close(out);
    preroll_.push(frame, frame_len_);
  }
  ++frames_done_;
}

void Segmenter::close(std::vector<Utterance>& out) {
  size_t min_loud = static_cast<size_t>(cfg_.sample_rate) * cfg_.min_speech_ms / 1000;
  if (loud_samples_ >= min_loud) {
    Utterance u;
    u.start_ms = utt_start_sample_ * 1000 / cfg_.sample_rate;
    u.end_ms = (utt_start_sample_ + static_cast<int64_t>(utt_.size())) * 1000 / cfg_.sample_rate;
    u.samples = std::move(utt_);
    out.push_back(std::move(u));
  }
  utt_.clear();
  in_utt_ = false;
  loud_samples_ = 0;
}
