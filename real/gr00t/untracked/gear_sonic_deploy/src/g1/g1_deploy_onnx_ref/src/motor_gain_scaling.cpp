#include "motor_gain_scaling.hpp"

#include <array>
#include <cctype>
#include <charconv>
#include <optional>
#include <string>
#include <string_view>

namespace {

// Guards against magnitude typos (for example, 15 instead of 1.5) reaching
// the robot.
constexpr float kMaxScaleFactor = 4.0F;

std::optional<int> parse_motor_index(std::string_view value,
                                     std::string& error) {
  int index = 0;
  const auto result =
      std::from_chars(value.data(), value.data() + value.size(), index);
  if (result.ec == std::errc::invalid_argument ||
      result.ptr != value.data() + value.size()) {
    error = "invalid motor index '" + std::string(value) + "'";
    return std::nullopt;
  }
  if (result.ec != std::errc{} || index < 0 || index >= G1_NUM_MOTOR) {
    error = "motor index out of range [0, " +
            std::to_string(G1_NUM_MOTOR - 1) + "]: '" +
            std::string(value) + "'";
    return std::nullopt;
  }
  return index;
}

std::optional<float> parse_scale(std::string_view value, std::string& error) {
  // This target uses -ffast-math, which makes std::isfinite/std::signbit
  // unreliable. Reject signs and non-numeric spellings lexically first.
  if (value.empty() ||
      (!std::isdigit(static_cast<unsigned char>(value.front())) &&
       value.front() != '.')) {
    error = "invalid scale factor '" + std::string(value) + "'";
    return std::nullopt;
  }

  float scale = 0.0F;
  const auto result = std::from_chars(value.data(), value.data() + value.size(),
                                      scale, std::chars_format::general);
  if (result.ec != std::errc{} || result.ptr != value.data() + value.size()) {
    error = "invalid scale factor '" + std::string(value) + "'";
    return std::nullopt;
  }
  if (scale > kMaxScaleFactor) {
    error = "scale factor must be in [0, 4]: '" + std::string(value) + "'";
    return std::nullopt;
  }
  return scale;
}

bool select_motor(
    int index,
    std::span<const std::optional<float>, G1_NUM_MOTOR> scales,
    std::array<bool, G1_NUM_MOTOR>& selected, std::string& error) {
  if (selected[index] || scales[index]) {
    error = "motor index scaled more than once: " + std::to_string(index);
    return false;
  }
  selected[index] = true;
  return true;
}

bool select_motor_range(
    std::string_view selector,
    std::span<const std::optional<float>, G1_NUM_MOTOR> scales,
    std::array<bool, G1_NUM_MOTOR>& selected, std::string& error) {
  const auto separator = selector.find('-');
  if (separator == std::string_view::npos) {
    const auto index = parse_motor_index(selector, error);
    if (!index) {
      return false;
    }
    return select_motor(*index, scales, selected, error);
  }

  if (selector.find('-', separator + 1) != std::string_view::npos) {
    error = "invalid motor range '" + std::string(selector) + "'";
    return false;
  }

  const auto first = parse_motor_index(selector.substr(0, separator), error);
  const auto last = parse_motor_index(selector.substr(separator + 1), error);
  if (!first || !last) {
    return false;
  }
  if (*first > *last) {
    error = "motor range must be ascending: '" + std::string(selector) + "'";
    return false;
  }

  for (int index = *first; index <= *last; ++index) {
    if (!select_motor(index, scales, selected, error)) {
      return false;
    }
  }
  return true;
}

std::string format_scale(float scale) {
  std::array<char, 32> buffer;
  const auto result =
      std::to_chars(buffer.data(), buffer.data() + buffer.size(), scale);
  return std::string(buffer.data(), result.ptr);
}

}  // namespace

std::optional<std::string> add_motor_gain_scale(
    std::string_view specification,
    std::span<std::optional<float>, G1_NUM_MOTOR> scales) {
  const auto assignment = specification.find('=');
  if (assignment == std::string_view::npos ||
      specification.find('=', assignment + 1) != std::string_view::npos) {
    return "expected <motor-list>=<factor>, got '" +
           std::string(specification) + "'";
  }

  std::string error;
  const auto scale = parse_scale(specification.substr(assignment + 1), error);
  if (!scale) {
    return error;
  }

  const auto motor_list = specification.substr(0, assignment);
  std::array<bool, G1_NUM_MOTOR> selected{};
  std::size_t offset = 0;
  while (offset <= motor_list.size()) {
    const auto comma = motor_list.find(',', offset);
    const auto selector = motor_list.substr(
        offset, comma == std::string_view::npos ? motor_list.size() - offset
                                                : comma - offset);
    if (!select_motor_range(selector, scales, selected, error)) {
      return error;
    }
    if (comma == std::string_view::npos) {
      break;
    }
    offset = comma + 1;
  }

  for (int index = 0; index < G1_NUM_MOTOR; ++index) {
    if (selected[index]) {
      scales[index] = *scale;
    }
  }
  return std::nullopt;
}

std::string format_motor_gain_scales(
    std::span<const std::optional<float>, G1_NUM_MOTOR> scales) {
  std::string formatted;
  for (int index = 0; index < G1_NUM_MOTOR; ++index) {
    if (!scales[index]) {
      continue;
    }
    if (!formatted.empty()) {
      formatted += ',';
    }
    formatted += std::to_string(index) + '=' + format_scale(*scales[index]);
  }
  return formatted;
}

void apply_motor_gain_scales(const MotorGainScaleConfig& config,
                             MotorCommand& command) {
  for (int index = 0; index < G1_NUM_MOTOR; ++index) {
    if (config.kp[index]) {
      command.kp[index] *= *config.kp[index];
    }
    if (config.kd[index]) {
      command.kd[index] *= *config.kd[index];
    }
  }
}
