#pragma once
#if defined(__FINITE_MATH_ONLY__) && __FINITE_MATH_ONLY__ > 0
#error "cart78 token guard requires -fno-finite-math-only so NaN/Inf checks cannot be optimized away"
#endif
// Dedicated cart78 token lease. Pure C++ standard-library code; no hardware side effects.
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>

namespace cart78 {
inline constexpr const char* kProtocol = "cart78.token_lease.v1";
inline constexpr const char* kBuildMarker = "CART78_TOKEN_LEASE_GUARD_V1";
inline int64_t now_ns() {
  return std::chrono::duration_cast<std::chrono::nanoseconds>(
      std::chrono::steady_clock::now().time_since_epoch()).count();
}
inline int64_t controller_instance() { static const int64_t value = now_ns(); return value; }
inline int timeout_ms() {
  static const int value = [] {
    const char* raw = std::getenv("CART78_LEASE_TIMEOUT_MS");
    const std::string s = raw ? raw : "250";
    size_t used = 0; int v = std::stoi(s, &used);
    if (used != s.size() || v < 100 || v > 1000) throw std::runtime_error("lease timeout must be 100..1000 ms");
    return v;
  }();
  return value;
}
inline std::string verified_hash_env(const char* key) {
  const char* raw = std::getenv(key); std::string s = raw ? raw : "";
  if (s.size() != 64 || s.find_first_not_of("0123456789abcdef") != std::string::npos)
    throw std::runtime_error(std::string("missing launcher-verified asset hash: ") + key);
  return s;
}
struct Packet {
  bool valid = false, enable = false, stop = false;
  int64_t instance = 0, session = 0, sequence = -1;
  std::string reason = "invalid_packet";
};

// Header/Buffer templates let the exact parser be exercised without Unitree/ZMQ libraries.
template<class Header, class Buffers>
Packet decode_packet(const Header& h, const Buffers& bufs) {
  Packet p;
  if (h.version != 4 || h.count != 1 || (h.endian != "le" && !h.endian.empty()) ||
      h.fields.size() != bufs.size() || h.fields.size() < 5 || h.fields.size() > 16) return p;
  const uint16_t endian_probe = 1;
  if (*reinterpret_cast<const uint8_t*>(&endian_probe) != 1) { p.reason="host_not_little_endian"; return p; }
  for (size_t i=0;i<h.fields.size();++i) for (size_t j=0;j<i;++j)
    if (h.fields[i].name == h.fields[j].name) { p.reason="duplicate_field"; return p; }
  auto find = [&](const char* name) -> int {
    for (size_t i=0;i<h.fields.size();++i) if (h.fields[i].name == name) return static_cast<int>(i);
    return -1;
  };
  auto scalar = [&](const char* name, const char* dtype, void* out, size_t bytes) -> bool {
    const int i=find(name); if(i<0) return false;
    const auto& f=h.fields[static_cast<size_t>(i)]; const auto& b=bufs[static_cast<size_t>(i)];
    if(f.dtype!=dtype || f.shape.size()!=1 || f.shape[0]!=1 || b.size!=bytes || !b.data) return false;
    std::memcpy(out,b.data,bytes); return true;
  };
  uint8_t en=0, st=0;
  if(!scalar("control_enable","u8",&en,1) || !scalar("control_stop","u8",&st,1) || en>1 || st>1 ||
     !scalar("controller_instance","i64",&p.instance,8) || !scalar("client_session","i64",&p.session,8) ||
     !scalar("action_seq","i64",&p.sequence,8) || p.instance<=0 || p.session<=0 || p.sequence<0) return p;
  p.enable=en; p.stop=st;
  if(p.stop || !p.enable) { p.valid=true; p.reason=p.stop?"explicit_stop":"control_disabled"; return p; }
  const std::array<float,7> lmin={-1.05f,-0.724f,0,-1.57f,-1.75f,-1.57f,-1.75f};
  const std::array<float,7> lmax={1.05f,1.05f,1.75f,0,0,0,0};
  const std::array<float,7> rmin={-1.05f,-1.05f,-1.75f,0,0,0,0};
  const std::array<float,7> rmax={1.05f,0.742f,0,1.57f,1.75f,1.57f,1.75f};
  for (const char* name : {"token_state","left_hand_joints","right_hand_joints"}) {
    const int i=find(name); if(i<0) return p;
    const bool token=std::string(name)=="token_state", left=std::string(name)=="left_hand_joints";
    const size_t width=token?64:7; const auto& f=h.fields[static_cast<size_t>(i)];const auto& b=bufs[static_cast<size_t>(i)];
    if(f.dtype!="f32" || f.shape.size()!=2 || f.shape[0]!=1 || f.shape[1]!=width || b.size!=4*width || !b.data) return p;
    for(size_t j=0;j<width;++j) {
      const auto* raw=static_cast<const uint8_t*>(b.data)+4*j;
      const uint32_t bits=uint32_t(raw[0])|(uint32_t(raw[1])<<8)|(uint32_t(raw[2])<<16)|(uint32_t(raw[3])<<24);
      // Reject wire NaN/Inf before any floating-point operation (-ffast-math safe).
      if((bits & 0x7f800000u)==0x7f800000u) {p.reason="nonfinite_action";return p;}
      float v; std::memcpy(&v,&bits,4);
      if(token) {
        if(v < -0.625001f || v > 0.625001f || std::fabs(v*16.0f-std::round(v*16.0f))>1e-5f) {p.reason="invalid_fsq_token";return p;}
      } else {
        const auto& lo=left?lmin:rmin; const auto& hi=left?lmax:rmax;
        if(v<lo[j]-1e-5f || v>hi[j]+1e-5f) {p.reason="hand_out_of_controller_bounds";return p;}
      }
    }
  }
  p.valid=true;p.reason="valid_action";return p;
}

enum class Receive { Buffer, Ignore, Fault };
class TokenLease {
 public:
  TokenLease(int64_t instance=controller_instance(), int timeout=timeout_ms())
      : instance_(instance), ttl_ns_(static_cast<int64_t>(timeout)*1000000) {
    if(instance<=0 || timeout<100 || timeout>1000) throw std::invalid_argument("bad lease configuration");
  }
  Receive observe(const Packet& p,int64_t now) {
    std::lock_guard<std::mutex> lock(mu_);
    if(armed_ && now-last_rx_>ttl_ns_) { trip_locked("token_lease_expired"); return Receive::Fault; }
    if(!p.valid) { trip_locked(p.reason); return Receive::Fault; }
    if(p.instance!=instance_) return Receive::Ignore; // Never renew another controller's lease.
    if(p.stop || !p.enable) {trip_locked(p.stop?"explicit_stop":"control_disabled");return Receive::Fault;}
    if(armed_ && p.session!=active_session_) {trip_locked("client_session_changed");return Receive::Fault;}
    if(p.session==offered_session_ && p.sequence<=offered_seq_) return Receive::Ignore;
    offered_session_=p.session;offered_seq_=p.sequence;last_rx_=now;candidate_=true;
    return Receive::Buffer;
  }
  bool arm(int64_t now) {
    std::lock_guard<std::mutex> lock(mu_);
    if(!candidate_ || now<last_rx_ || now-last_rx_>ttl_ns_) return false;
    active_session_=offered_session_;armed_=true;fault_pending_=false;reason_.clear();return true;
  }
  void disarm() {std::lock_guard<std::mutex> lock(mu_);armed_=false;candidate_=false;fault_pending_=false;reason_.clear();}
  bool usable(int64_t now) const {
    std::lock_guard<std::mutex> lock(mu_);
    return armed_ && candidate_ && now>=last_rx_ && now-last_rx_<=ttl_ns_;
  }
  bool poll_fault(int64_t now,std::string& reason) {
    std::lock_guard<std::mutex> lock(mu_);
    if(armed_ && (now<last_rx_ || now-last_rx_>ttl_ns_))trip_locked("token_lease_expired");
    if(!fault_pending_)return false;
    reason=reason_;fault_pending_=false;return true;
  }
  int64_t last_receive_ns() const {std::lock_guard<std::mutex> lock(mu_);return last_rx_;}
 private:
  void trip_locked(const std::string& why){armed_=false;candidate_=false;fault_pending_=true;reason_=why;}
  const int64_t instance_,ttl_ns_;
  mutable std::mutex mu_;
  bool armed_=false,candidate_=false,fault_pending_=false;
  int64_t active_session_=0,offered_session_=0,offered_seq_=-1,last_rx_=0;
  std::string reason_;
};
} // namespace cart78
