#pragma once
#include <cstddef>
#include <cstdint>
#include <vector>

#include "ring_buffer.h"

// Energy gate: open while frames are loud, and for hangover_frames after the last loud one.
class VoiceGate {
 public:
  VoiceGate(float threshold, int hangover_frames)
      : threshold_(threshold), hangover_frames_(hangover_frames) {}

  // Feed one frame; returns whether the gate is open after it.
  bool update(const float* frame, size_t n);
  static float rms(const float* frame, size_t n);
  bool last_frame_loud() const { return last_loud_; }

 private:
  float threshold_;
  int hangover_frames_;
  int quiet_run_ = 1 << 30;
  bool last_loud_ = false;
};

struct SegmenterConfig {
  int sample_rate = 16000;
  int frame_ms = 20;
  float threshold = 0.01f;  // RMS on samples in [-1, 1]
  int hangover_ms = 500;
  int preroll_ms = 300;
  int min_speech_ms = 250;  // utterances with less loud audio than this are dropped
  int max_utterance_ms = 15000;
};

struct Utterance {
  int64_t start_ms = 0;
  int64_t end_ms = 0;
  std::vector<float> samples;
};

// Cuts a sample stream into utterances: runs of speech plus a little audio before and after.
class Segmenter {
 public:
  explicit Segmenter(const SegmenterConfig& cfg = SegmenterConfig());

  std::vector<Utterance> feed(const float* samples, size_t n);
  // End of stream: closes an utterance that is still open.
  std::vector<Utterance> flush();

 private:
  void process_frame(const float* frame, std::vector<Utterance>& out);
  void close(std::vector<Utterance>& out);

  SegmenterConfig cfg_;
  size_t frame_len_;
  VoiceGate gate_;
  RingBuffer preroll_;
  std::vector<float> pending_;
  int64_t frames_done_ = 0;
  bool in_utt_ = false;
  int64_t utt_start_sample_ = 0;
  std::vector<float> utt_;
  size_t loud_samples_ = 0;
};
