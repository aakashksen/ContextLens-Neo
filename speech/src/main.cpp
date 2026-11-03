#include <grpcpp/grpcpp.h>

#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <string>

#include "speech.grpc.pb.h"
#include "transcriber.h"
#include "vad.h"

namespace pb = contextlens::speech;

class SpeechService final : public pb::Speech::Service {
 public:
  explicit SpeechService(Transcriber& t) : transcriber_(t) {}

  grpc::Status Transcribe(grpc::ServerContext* ctx,
                          grpc::ServerReaderWriter<pb::TranscriptSegment, pb::AudioChunk>* stream)
      override {
    ctx->AddInitialMetadata("x-compute-backend", transcriber_.backend());
    Segmenter seg;
    pb::AudioChunk chunk;
    std::string carry;  // an odd trailing byte from the previous chunk
    try {
      while (stream->Read(&chunk)) {
        carry += chunk.pcm();
        size_t n = carry.size() / 2;
        std::vector<float> samples(n);
        for (size_t i = 0; i < n; ++i) {
          int16_t s = static_cast<int16_t>(static_cast<uint8_t>(carry[2 * i]) |
                                           (static_cast<uint8_t>(carry[2 * i + 1]) << 8));
          samples[i] = s / 32768.0f;
        }
        carry.erase(0, n * 2);
        for (const Utterance& u : seg.feed(samples.data(), samples.size())) emit(u, stream);
      }
      for (const Utterance& u : seg.flush()) emit(u, stream);
    } catch (const std::exception& e) {
      return grpc::Status(grpc::StatusCode::INTERNAL, e.what());
    }
    return grpc::Status::OK;
  }

 private:
  void emit(const Utterance& u,
            grpc::ServerReaderWriter<pb::TranscriptSegment, pb::AudioChunk>* stream) {
    for (const Segment& s : transcriber_.transcribe(u)) {
      pb::TranscriptSegment out;
      out.set_text(s.text);
      out.set_start_ms(s.start_ms);
      out.set_end_ms(s.end_ms);
      out.set_is_final(true);
      stream->Write(out);
    }
  }

  Transcriber& transcriber_;
};

static std::string env_or(const char* name, const char* fallback) {
  const char* v = std::getenv(name);
  return v && *v ? v : fallback;
}

int main() {
  std::string model = env_or("SPEECH_MODEL", "/models/ggml-tiny.en.bin");
  std::string addr = env_or("SPEECH_ADDR", "0.0.0.0:50051");
  try {
    Transcriber transcriber(model);
    SpeechService service(transcriber);
    grpc::ServerBuilder builder;
    builder.AddListeningPort(addr, grpc::InsecureServerCredentials());
    builder.RegisterService(&service);
    std::unique_ptr<grpc::Server> server = builder.BuildAndStart();
    if (!server) {
      std::cerr << "cannot listen on " << addr << std::endl;
      return 1;
    }
    std::cout << "speech service listening on " << addr << ", compute backend: "
              << transcriber.backend() << std::endl;
    server->Wait();
  } catch (const std::exception& e) {
    std::cerr << "fatal: " << e.what() << std::endl;
    return 1;
  }
  return 0;
}
