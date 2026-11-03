# CPU build by default. A GPU build is a different base and a flag, untested here:
#   docker build --build-arg BUILD_BASE=nvidia/cuda:12.4.1-devel-ubuntu22.04 \
#                --build-arg RUNTIME_BASE=nvidia/cuda:12.4.1-runtime-ubuntu22.04 \
#                --build-arg GGML_CUDA=ON speech
# and run it with a GPU (docker run --gpus all). The service reports "gpu:<name>" or "cpu" at start-up
# and in the x-compute-backend gRPC header.
ARG BUILD_BASE=ubuntu:24.04
ARG RUNTIME_BASE=ubuntu:24.04

FROM ${BUILD_BASE} AS build
ARG GGML_CUDA=OFF
ARG WHISPER_CPP_TAG=v1.7.6
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
      build-essential cmake pkg-config git ca-certificates \
      libgrpc++-dev libprotobuf-dev protobuf-compiler-grpc \
    && rm -rf /var/lib/apt/lists/*
RUN git clone --depth 1 --branch ${WHISPER_CPP_TAG} https://github.com/ggml-org/whisper.cpp /opt/whisper.cpp
WORKDIR /src
COPY CMakeLists.txt ./
COPY proto proto
COPY src src
COPY tests tests
RUN cmake -S . -B build -DWHISPER_DIR=/opt/whisper.cpp -DGGML_CUDA=${GGML_CUDA} \
    && cmake --build build -j"$(nproc)" \
    && ctest --test-dir build --output-on-failure

FROM ${BUILD_BASE} AS model
ARG MODEL_REVISION=5359861c739e955e79d9a303bcbc70fb988958b1
ARG MODEL_SHA256=921e4cf8686fdd993dcd081a5da5b6c365bfde1162e72b08d75ac75289920b1f
RUN apt-get update && apt-get install -y --no-install-recommends curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*
RUN mkdir /models && curl -fsSL -o /models/ggml-tiny.en.bin \
      https://huggingface.co/ggerganov/whisper.cpp/resolve/${MODEL_REVISION}/ggml-tiny.en.bin \
    && echo "${MODEL_SHA256}  /models/ggml-tiny.en.bin" | sha256sum -c -

FROM ${RUNTIME_BASE}
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
      libgrpc++1.51t64 libprotobuf32t64 libgomp1 \
    && rm -rf /var/lib/apt/lists/*
COPY --from=build /src/build/speech_server /usr/local/bin/speech_server
COPY --from=model /models /models
ENV SPEECH_MODEL=/models/ggml-tiny.en.bin SPEECH_ADDR=0.0.0.0:50051
EXPOSE 50051
CMD ["speech_server"]
