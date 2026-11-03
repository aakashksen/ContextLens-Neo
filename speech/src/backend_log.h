#pragma once
#include <string>

// Reads whisper.cpp's start-up log and returns "gpu:<device>" if it reports initialising a GPU
// backend for the model, else "cpu". The device list alone is not enough: a registered GPU can
// fail to initialise and whisper then runs on the CPU.
std::string backend_from_log(const std::string& log);
