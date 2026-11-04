#include "../include/cart78_token_lease.hpp"
#include <cassert>
#include <iostream>
#include <limits>
struct Field {std::string name,dtype;std::vector<size_t> shape;};
struct Header {int version=4,count=1;std::string endian="le";std::vector<Field> fields;};
struct Buffer {const void* data;size_t size;};
struct Frame {
 Header h;std::vector<Buffer>b;uint8_t enable=1,stop=0;int64_t instance=123,session=456,seq=0;
 std::array<float,64>token{};std::array<float,7>left{},right{};
 Frame(bool controls_only=false){
  h.fields={{"control_enable","u8",{1}},{"control_stop","u8",{1}},{"controller_instance","i64",{1}},{"client_session","i64",{1}},{"action_seq","i64",{1}}};
  b={{&enable,1},{&stop,1},{&instance,8},{&session,8},{&seq,8}};
  if(!controls_only){h.fields.insert(h.fields.end(),{{"token_state","f32",{1,64}},{"left_hand_joints","f32",{1,7}},{"right_hand_joints","f32",{1,7}}});b.insert(b.end(),{{token.data(),256},{left.data(),28},{right.data(),28}});}
 }
 cart78::Packet parse(){return cart78::decode_packet(h,b);}
};
int main(){
 constexpr int64_t ms=1000000;std::string why;
 Frame f;assert(f.parse().valid);
 cart78::TokenLease g(123,250);
 assert(g.observe(f.parse(),0)==cart78::Receive::Buffer);assert(!g.usable(0));assert(g.arm(0));assert(g.usable(250*ms));
 assert(g.observe(f.parse(),200*ms)==cart78::Receive::Ignore);assert(g.last_receive_ns()==0);assert(!g.usable(251*ms));
 assert(g.poll_fault(251*ms,why)&&why=="token_lease_expired");assert(!g.arm(251*ms));
 f.seq=1;assert(g.observe(f.parse(),260*ms)==cart78::Receive::Buffer);assert(!g.usable(260*ms));assert(g.arm(260*ms));
 f.seq=2;assert(g.observe(f.parse(),300*ms)==cart78::Receive::Buffer);assert(g.last_receive_ns()==300*ms);
 f.seq=1;assert(g.observe(f.parse(),500*ms)==cart78::Receive::Ignore);assert(g.last_receive_ns()==300*ms);
 f.seq=3;assert(g.observe(f.parse(),551*ms)==cart78::Receive::Fault);assert(!g.usable(551*ms));
 g.poll_fault(551*ms,why);f.seq=4;g.observe(f.parse(),560*ms);assert(g.arm(560*ms));
 f.session=999;f.seq=0;assert(g.observe(f.parse(),570*ms)==cart78::Receive::Fault);assert(g.poll_fault(570*ms,why)&&why=="client_session_changed");
 f.session=456;f.seq=5;g.observe(f.parse(),580*ms);assert(g.arm(580*ms));
 Frame stop(true);stop.stop=1;stop.seq=0;assert(stop.parse().valid);assert(g.observe(stop.parse(),581*ms)==cart78::Receive::Fault);assert(g.poll_fault(581*ms,why)&&why=="explicit_stop");
 f.seq=6;g.observe(f.parse(),600*ms);assert(g.arm(600*ms));
 Frame other;other.instance=777;other.seq=99;assert(g.observe(other.parse(),700*ms)==cart78::Receive::Ignore);assert(g.last_receive_ns()==600*ms);
 f.token[3]=std::numeric_limits<float>::quiet_NaN();assert(!f.parse().valid);assert(g.observe(f.parse(),710*ms)==cart78::Receive::Fault);assert(!g.usable(710*ms));
 f.token[3]=std::numeric_limits<float>::infinity();assert(!f.parse().valid);
 f.token[3]=0.6875f;assert(!f.parse().valid);
 f.token[3]=0.1f;assert(!f.parse().valid);f.token[3]=0.625f;assert(f.parse().valid);
 f.left[1]=-1.5f;assert(!f.parse().valid);f.left={0,.7f,.7f,-1,-1.5f,-1,-1.5f};assert(f.parse().valid);
 Frame missing;missing.h.fields.erase(missing.h.fields.begin());missing.b.erase(missing.b.begin());assert(!missing.parse().valid);
 Frame malformed;malformed.h.fields[5].shape={64};assert(!malformed.parse().valid);
 Frame duplicate;duplicate.h.fields[7].name="left_hand_joints";assert(!duplicate.parse().valid);
 Frame disable(true);disable.enable=0;assert(disable.parse().valid);
 f.token[3]=0;f.seq=7;g.observe(f.parse(),800*ms);assert(g.arm(800*ms));g.disarm();assert(!g.usable(800*ms));assert(!g.arm(800*ms));
 std::cout<<"PASS parser, explicit arm, monotonic sequence, stale timeout, stop, wrong instance, takeover, NaN/range/grid rejection, no automatic rearm\n";
}
