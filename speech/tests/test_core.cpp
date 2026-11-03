// Plain-assert tests for the ring buffer and the voice gate; no framework needed.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <functional>
#include <vector>

#include "backend_log.h"
#include "ring_buffer.h"
#include "vad.h"

static int failures = 0;
#define CHECK(cond)                                                   \
  do {                                                                \
    if (!(cond)) {                                                    \
      std::printf("  CHECK failed: %s (line %d)\n", #cond, __LINE__); \
      ++failures;                                                     \
    }                                                                 \
  } while (0)

static std::vector<float> tone(size_t n, float amp) {
  std::vector<float> v(n);
  for (size_t i = 0; i < n; ++i) v[i] = amp * std::sin(2.0f * 3.14159265f * 220.0f * i / 16000.0f);
  return v;
}

static void ring_buffer_keeps_order_at_wraparound() {
  RingBuffer rb(5);
  float a[] = {1, 2, 3, 4};
  float b[] = {5, 6, 7};
  rb.push(a, 4);
  CHECK((rb.snapshot() == std::vector<float>{1, 2, 3, 4}));
  rb.push(b, 3);  // wraps; 1 and 2 are dropped
  CHECK(rb.size() == 5);
  CHECK((rb.snapshot() == std::vector<float>{3, 4, 5, 6, 7}));
  rb.clear();
  CHECK(rb.size() == 0);
}

static void silence_yields_no_segments() {
  Segmenter seg;
  std::vector<float> silence(16000 * 5, 0.0f);
  auto got = seg.feed(silence.data(), silence.size());
  auto tail = seg.flush();
  CHECK(got.empty());
  CHECK(tail.empty());
}

static void low_noise_yields_no_segments() {
  Segmenter seg;
  auto noise = tone(16000 * 3, 0.002f);
  CHECK(seg.feed(noise.data(), noise.size()).empty());
  CHECK(seg.flush().empty());
}

static void gate_opens_on_loud_and_closes_after_hangover() {
  VoiceGate gate(0.01f, 2);
  auto loud = tone(320, 0.3f);
  std::vector<float> quiet(320, 0.0f);
  CHECK(!gate.update(quiet.data(), 320));
  CHECK(gate.update(loud.data(), 320));
  CHECK(gate.update(quiet.data(), 320));   // hangover 1
  CHECK(gate.update(quiet.data(), 320));   // hangover 2
  CHECK(!gate.update(quiet.data(), 320));  // closed
}

static void speech_between_silence_yields_one_utterance() {
  Segmenter seg;
  std::vector<float> audio(16000, 0.0f);  // 1 s silence
  auto speech = tone(16000, 0.3f);        // 1 s tone
  audio.insert(audio.end(), speech.begin(), speech.end());
  audio.insert(audio.end(), 16000 * 2, 0.0f);  // 2 s silence
  // Odd-sized pieces exercise the carry-over of partial frames.
  std::vector<Utterance> got;
  for (size_t pos = 0; pos < audio.size(); pos += 1000) {
    size_t n = std::min<size_t>(1000, audio.size() - pos);
    auto u = seg.feed(audio.data() + pos, n);
    got.insert(got.end(), u.begin(), u.end());
  }
  auto tail = seg.flush();
  got.insert(got.end(), tail.begin(), tail.end());
  CHECK(got.size() == 1);
  if (got.size() == 1) {
    CHECK(got[0].start_ms >= 600 && got[0].start_ms <= 800);  // 1000 ms minus 300 ms preroll
    CHECK(got[0].end_ms >= 2400 && got[0].end_ms <= 2600);    // 2000 ms plus 500 ms hangover
  }
}

static void backend_is_gpu_only_when_whisper_reports_using_it() {
  CHECK(backend_from_log("whisper_init: loading\n") == "cpu");
  CHECK(backend_from_log("whisper_backend_init_gpu: no GPU found\n") == "cpu");
  CHECK(backend_from_log("whisper_backend_init_gpu: using CUDA0 backend\n") == "gpu:CUDA0");
  // a GPU that was registered but failed to initialise is not in use
  CHECK(backend_from_log("whisper_backend_init_gpu: using CUDA0 backend\n"
                         "whisper_backend_init_gpu: failed to initialize CUDA0 backend\n") == "cpu");
  // other backends (accelerators) are not the GPU
  CHECK(backend_from_log("whisper_backend_init: using BLAS backend\n") == "cpu");
}

int main() {
  struct T {
    const char* name;
    std::function<void()> fn;
  };
  std::vector<T> tests = {
      {"ring buffer keeps order at wraparound", ring_buffer_keeps_order_at_wraparound},
      {"silence yields no segments", silence_yields_no_segments},
      {"low noise yields no segments", low_noise_yields_no_segments},
      {"gate opens on loud and closes after hangover", gate_opens_on_loud_and_closes_after_hangover},
      {"speech between silence yields one utterance", speech_between_silence_yields_one_utterance},
      {"backend is gpu only when whisper reports using it", backend_is_gpu_only_when_whisper_reports_using_it},
  };
  for (auto& t : tests) {
    int before = failures;
    t.fn();
    std::printf("%s: %s\n", failures == before ? "PASS" : "FAIL", t.name);
  }
  return failures == 0 ? 0 : 1;
}
