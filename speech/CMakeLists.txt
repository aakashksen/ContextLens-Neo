cmake_minimum_required(VERSION 3.16)
project(contextlens_speech CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

enable_testing()

# Audio ring buffer and voice activity gate: no third-party dependencies.
add_library(speech_core src/ring_buffer.cpp src/vad.cpp src/backend_log.cpp)
target_include_directories(speech_core PUBLIC src)

add_executable(test_core tests/test_core.cpp)
target_link_libraries(test_core speech_core)
add_test(NAME core COMMAND test_core)

# The service itself needs whisper.cpp (WHISPER_DIR) and gRPC.
if(WHISPER_DIR)
  find_package(PkgConfig REQUIRED)
  pkg_check_modules(GRPCPP REQUIRED grpc++ protobuf)
  find_program(PROTOC protoc REQUIRED)
  find_program(GRPC_CPP_PLUGIN grpc_cpp_plugin REQUIRED)

  set(GEN ${CMAKE_CURRENT_BINARY_DIR}/gen)
  file(MAKE_DIRECTORY ${GEN})
  add_custom_command(
    OUTPUT ${GEN}/speech.pb.cc ${GEN}/speech.pb.h ${GEN}/speech.grpc.pb.cc ${GEN}/speech.grpc.pb.h
    COMMAND ${PROTOC} -I ${CMAKE_CURRENT_SOURCE_DIR}/proto --cpp_out=${GEN} --grpc_out=${GEN}
            --plugin=protoc-gen-grpc=${GRPC_CPP_PLUGIN} ${CMAKE_CURRENT_SOURCE_DIR}/proto/speech.proto
    DEPENDS proto/speech.proto)

  set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
  set(WHISPER_BUILD_TESTS OFF)
  set(WHISPER_BUILD_EXAMPLES OFF)
  add_subdirectory(${WHISPER_DIR} whisper EXCLUDE_FROM_ALL)

  add_executable(speech_server src/main.cpp src/transcriber.cpp
                 ${GEN}/speech.pb.cc ${GEN}/speech.grpc.pb.cc)
  target_include_directories(speech_server PRIVATE ${GEN} ${GRPCPP_INCLUDE_DIRS}
                             ${WHISPER_DIR}/include ${WHISPER_DIR}/ggml/include)
  target_link_libraries(speech_server speech_core whisper ${GRPCPP_LIBRARIES})
  target_link_directories(speech_server PRIVATE ${GRPCPP_LIBRARY_DIRS})
  target_compile_options(speech_server PRIVATE ${GRPCPP_CFLAGS_OTHER})
endif()
