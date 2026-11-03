#include "backend_log.h"

#include <sstream>

std::string backend_from_log(const std::string& log) {
  const std::string using_prefix = "whisper_backend_init_gpu: using ";
  const std::string failed_prefix = "whisper_backend_init_gpu: failed to initialize ";
  const std::string suffix = " backend";
  std::string device;
  std::istringstream lines(log);
  std::string line;
  while (std::getline(lines, line)) {
    for (const auto& [prefix, ok] : {std::pair{using_prefix, true}, std::pair{failed_prefix, false}}) {
      size_t at = line.find(prefix);
      if (at == std::string::npos) continue;
      size_t end = line.rfind(suffix);
      if (end == std::string::npos || end < at + prefix.size()) continue;
      device = ok ? line.substr(at + prefix.size(), end - at - prefix.size()) : "";
    }
  }
  return device.empty() ? "cpu" : "gpu:" + device;
}
