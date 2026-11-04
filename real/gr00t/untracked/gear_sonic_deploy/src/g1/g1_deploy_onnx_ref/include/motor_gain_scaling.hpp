#pragma once

#include <array>
#include <optional>
#include <span>
#include <string>
#include <string_view>

#include "robot_parameters.hpp"

struct MotorGainScaleConfig {
  std::array<std::optional<float>, G1_NUM_MOTOR> kp;
  std::array<std::optional<float>, G1_NUM_MOTOR> kd;
};

std::optional<std::string> add_motor_gain_scale(
    std::string_view specification,
    std::span<std::optional<float>, G1_NUM_MOTOR> scales);

std::string format_motor_gain_scales(
    std::span<const std::optional<float>, G1_NUM_MOTOR> scales);

void apply_motor_gain_scales(const MotorGainScaleConfig& config,
                             MotorCommand& command);
