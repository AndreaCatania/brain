#pragma once

#include <string>

namespace brain {
std::string itos(int64_t p_number, int base = 10, bool capitalize_hex = false);
std::string rtos(double p_number, int p_decimals = -1);
} // namespace brain
