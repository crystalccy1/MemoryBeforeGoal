#pragma once
#include "robot_parameters.hpp"
#include <bit>
#include <cstdint>

// Output selection only. Damping is the existing shutdown command, not a
// guarantee of balance or of delivery over a failed physical link.
inline MotorCommand DampingMotorCommand() {
  MotorCommand command;
  command.kd.fill(8.0f);
  return command;
}

inline MotorCommand SelectMotorCommand(const MotorCommand& active, bool stop) {
  return stop ? DampingMotorCommand() : active;
}

inline bool StateExpired(bool has_state, double age_seconds) {
  const auto exponent = std::bit_cast<uint64_t>(age_seconds) & UINT64_C(0x7ff0000000000000);
  return !has_state || exponent == UINT64_C(0x7ff0000000000000)
      || !(age_seconds >= 0.0 && age_seconds <= 0.5);
}

inline bool ValidMotorTelemetry(float position, float velocity) {
  const auto q_exp = std::bit_cast<uint32_t>(position) & UINT32_C(0x7f800000);
  const auto v_exp = std::bit_cast<uint32_t>(velocity) & UINT32_C(0x7f800000);
  return q_exp != UINT32_C(0x7f800000) && v_exp != UINT32_C(0x7f800000)
      && position > -100.0f && position < 100.0f && velocity >= -35.0f && velocity <= 35.0f;
}

// A pending planner start is inspected once per input tick. Never wait/sleep
// inside the input handler: new stop commands must be polled on the next tick.
class PlannerStartWait {
 public:
  enum class Result { Pending, Ready, Failed };
  void Begin(uint64_t epoch, double now) { epoch_ = epoch; deadline_ = now + 5.0; active_ = true; }
  void Cancel() { active_ = false; }
  bool Active() const { return active_; }
  Result Poll(bool stop, bool enabled, bool initialized, uint64_t ready_epoch,
              bool motion_installed, double now) {
    if (stop || !enabled || now >= deadline_) { active_ = false; return Result::Failed; }
    if (initialized && ready_epoch == epoch_ && motion_installed) {
      active_ = false;
      return Result::Ready;
    }
    return Result::Pending;
  }
 private:
  uint64_t epoch_ = 0;
  double deadline_ = 0;
  bool active_ = false;
};

inline bool PlannerCompletionCurrent(bool stop, bool enabled, uint64_t requested, uint64_t current) {
  return !stop && enabled && requested == current;
}
