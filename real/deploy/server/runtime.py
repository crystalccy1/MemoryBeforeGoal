"""Fail-closed30Hz cart78.v1 transport around the actual a7 inference methods.
No upstream controller thread or os._exit path is started.
"""
from __future__ import annotations
import asyncio,collections,dataclasses,json,math,time
from typing import Any
import numpy as np
from fastapi import WebSocket
try:
    from .bundle import PROTOCOL
    from .backend import BackendError,state_vector
except ImportError:
    from bundle import PROTOCOL
    from backend import BackendError,state_vector

class SafetyFault(RuntimeError):
    def __init__(self,code,message):self.code=code;super().__init__(message)

@dataclasses.dataclass(frozen=True)
class Observation:
    seq:int
    sent_ns:int
    received_ns:int
    payload:dict

@dataclasses.dataclass(frozen=True)
class Origin:
    observation:Observation
    generation_ns:int
    origin_chunk_id:int

@dataclasses.dataclass
class Chunk:
    normalized:np.ndarray
    native:np.ndarray
    origins:list[Origin]
    chunk_id:int
    model_observation_seq:int
    index:int=0

@dataclasses.dataclass
class Job:
    task:asyncio.Task
    observation:Observation
    started_ns:int
    action_count:int
    previous_origins:list[Origin] | None
    delay:int
    deadline_reported:bool=False

class DeploymentRuntime:
    def __init__(self,backend,allow_execution=False,rtc_delay=7,min_exec_horizon=15,observation_timeout_ms=250,action_ttl_ms=100):
        self.backend=backend;self.allow_execution=bool(allow_execution and not backend.mock)
        self.delay=int(rtc_delay);self.shift=int(min_exec_horizon);self.obs_timeout_ms=float(observation_timeout_ms);self.ttl_ms=int(action_ttl_ms)
        if not 1<=self.delay<8:raise ValueError('rtc_delay must be1..7; never silently clamped')
        if not 1<=self.shift<30 or self.shift+self.delay>30:raise ValueError('Require1<=exec_horizon<30 and exec_horizon+rtc_delay<=30')
        if not 0<self.obs_timeout_ms<=1000 or not 1<=self.ttl_ms<=1000:raise ValueError('Invalid freshness limits')
        self.period=1/30;self.budget_seconds=self.delay/30;self.samples=collections.deque(maxlen=32);self.fault=None
        self.warmed=False;self.active=False;self.seen_sessions=collections.deque(maxlen=256);self.model_loaded=True
        self.session_count=0;self.last_prefix_drift=None
        self.app=backend.app;backend.attach_runtime(self)
        @self.app.get('/deployment-info')
        async def deployment_info():return self.info()
        @self.app.get('/health')
        async def health():
            info=self.info()
            return {'status':'ready' if info['ready'] else 'not_ready','model_loaded':info['model_loaded'],'ready':info['ready'],'latency_ok':info['latency_ok'],'dry_run':info['dry_run'],'mock':info['mock'],'fault':info['fault']}

    @property
    def latency_ok(self):return len(self.samples)>=3 and max(list(self.samples)[-3:])<=self.budget_seconds

    def info(self):
        extra={'memory':self.backend.memory_status()} if hasattr(self.backend,'memory_status') else {}
        return {**extra,**self.backend.contract,'protocol':PROTOCOL,'checkpoint_sha256':self.backend.checkpoint_sha256,'checkpoint_step':getattr(self.backend,'manifest',{}).get('checkpoint',{}).get('step'),'contract_sha256':self.backend.contract_sha256,'inference_steps':self.backend.inference_steps,'attention_backend':self.backend.attention_backend,'actual_attention':self.backend.actual_attention,'runtime_versions':self.backend.runtime_versions,'ready':bool(self.model_loaded and self.latency_ok and self.allow_execution and self.fault is None and not self.backend.mock),'latency_ok':bool(self.latency_ok),'dry_run':not self.allow_execution,'mock':bool(self.backend.mock),'model_loaded':self.model_loaded,'active_session':self.active,'inference_busy':self.backend.busy,'fault':self.fault,'rtc_delay':self.delay,'min_exec_horizon':self.shift,'inference_budget_ms':self.budget_seconds*1000,'observation_timeout_ms':self.obs_timeout_ms,'action_ttl_ms':self.ttl_ms,'recent_inference_ms':[s*1000 for s in self.samples],'rtc_prefix_raw_max_error_last':self.last_prefix_drift,'adapters':['isolated_local_config_assets','state_shape_boundary_only','final_normalized_rtc_prefix_restore','safe_session_scheduler']}

    def record(self,seconds,result):
        self.samples.append(float(seconds));self.last_prefix_drift=result.get('prefix_raw_max_error')

    async def handle(self,ws:WebSocket):
        await ws.accept();session_id=None;mode=None;receiver=None;control=None;inflight=set();write_lock=asyncio.Lock();latest=None;latest_event=asyncio.Event();last_seq=-1;last_sent=-1;instruction=None
        async def send(data):
            async with write_lock:
                await asyncio.wait_for(ws.send_json(data),timeout=max(.25,self.ttl_ms/1000))
        async def profile(reason,**extra):
            if mode=='execute':return  # Execution packets carry telemetry; profile frames mean dry-run only.
            await send({'type':'profile','protocol':PROTOCOL,'session_id':session_id,'reason':reason,'deployment':self.info(),**extra})
        async def fail(code,message):
            if claimed or not self.active:
                self.fault={'code':code,'message':str(message),'timestamp_ns':time.time_ns()}
            try:await send({'type':'error','protocol':PROTOCOL,'session_id':session_id,'code':code,'message':str(message),'deployment':self.info()})
            except Exception:pass
        def spawn_predict(obs,prev=None,delay=0):
            t=asyncio.create_task(asyncio.to_thread(self.backend.predict,obs,prev,delay));inflight.add(t)
            t.add_done_callback(inflight.discard);return t
        async def get_fresh(after_ns=-1):
            while True:
                obs=latest
                if obs is not None and obs.received_ns>after_ns and (time.monotonic_ns()-obs.received_ns)/1e6<=self.obs_timeout_ms:return obs
                latest_event.clear()
                # Re-check after clear so a new frame cannot be lost around await.
                if latest is not None and latest is not obs:continue
                try:await asyncio.wait_for(latest_event.wait(),timeout=self.obs_timeout_ms/1000)
                except asyncio.TimeoutError:raise SafetyFault('observation_stale','No fresh observation arrived within the configured deadline')
        async def receive():
            nonlocal latest,last_seq,last_sent,instruction
            while True:
                raw=await ws.receive_text()
                if len(raw)>2_100_000:raise SafetyFault('payload_too_large','Observation envelope too large')
                p=json.loads(raw)
                if p.get('type')!='observation' or p.get('protocol')!=PROTOCOL or p.get('session_id')!=session_id:raise SafetyFault('session_mismatch','Unexpected message protocol/session')
                seq=p.get('observation_seq');sent=p.get('client_sent_monotonic_ns')
                if type(seq) is not int or seq<0 or seq<=last_seq or type(sent) is not int or sent<=last_sent:raise SafetyFault('observation_order','Observation sequence and client timestamp must strictly increase')
                state_vector(p.get('state_model43'))
                text=p.get('instruction')
                if not isinstance(text,str) or not text.strip() or len(text)>1024:raise SafetyFault('invalid_instruction','Instruction must be nonempty and at most1024characters')
                if instruction is not None and text!=instruction:raise SafetyFault('instruction_changed','Reconnect with a new session to change the instruction')
                instruction=text
                image=p.get('image_jpeg_b64')
                if not isinstance(image,str) or len(image)>2_000_000:raise SafetyFault('invalid_image','JPEG envelope is missing or too large')
                latest=Observation(seq,sent,time.monotonic_ns(),p);last_seq=seq;last_sent=sent;latest_event.set()
        async def run_control():
            current=None;job=None;action_count=0;next_chunk_id=0;last_input_ns=-1;next_tick=time.monotonic()
            initial=await get_fresh()
            memory=hasattr(self.backend,'begin_session')
            if not self.warmed:
                await profile('warming_up_no_actions')
                if memory:await asyncio.shield(asyncio.to_thread(self.backend.begin_session,initial))
                warm=await asyncio.shield(spawn_predict(initial))
                prev=np.concatenate([warm['normalized'][self.shift:],np.zeros((self.shift,78),np.float32)])
                for _ in range(2):
                    began=time.monotonic();result=await asyncio.shield(spawn_predict(initial,prev,self.delay));self.record(time.monotonic()-began,result)
                self.warmed=True
            warm_end=time.monotonic_ns()
            await profile('warmup_complete_waiting_for_new_observation')
            # Never use the observation captured before model warmup for active control.
            first=await get_fresh(after_ns=warm_end)
            if memory:
                # SAM3-v2: this session's memory starts at this frame (raw frame 0), tracked before any action.
                await profile('priming_memory_no_actions')
                await asyncio.shield(asyncio.to_thread(self.backend.begin_session,first))
                await get_fresh(after_ns=time.monotonic_ns())
            while True:
                now_ns=time.monotonic_ns();now=now_ns/1e9
                if latest is None or (now_ns-latest.received_ns)/1e6>self.obs_timeout_ms:raise SafetyFault('observation_stale','Observation stream stopped; no cached action will be replayed')
                if job is not None and not job.task.done() and (now_ns-job.started_ns)/1e9>self.budget_seconds:
                    if mode=='execute':raise SafetyFault('latency_budget_exceeded','Inference exceeded the frozen-prefix delay budget; action output stopped')
                    current=None
                    if not job.deadline_reported:
                        job.deadline_reported=True;await profile('latency_budget_exceeded_no_actions',measured_lower_bound_ms=(now_ns-job.started_ns)/1e6)
                if job is not None and job.task.done():
                    result=job.task.result();elapsed=(time.monotonic_ns()-job.started_ns)/1e9;self.record(elapsed,result)
                    ticks=math.ceil(elapsed*30-1e-9)
                    if ticks>=8 or ticks>self.delay:
                        current=None;await profile('latency_budget_exceeded_no_actions',latency_ms=elapsed*1000,delay_ticks=ticks)
                        if mode=='execute':raise SafetyFault('latency_budget_exceeded','Replan exceeded the trained delay range')
                    else:
                        next_chunk_id+=1;origin=Origin(job.observation,result['generation_timestamp_ns'],next_chunk_id)
                        origins=[origin]*30
                        if job.previous_origins is not None:
                            origins[:job.delay]=job.previous_origins[:job.delay]
                        spent=action_count-job.action_count if job.previous_origins is not None else 0
                        if spent>job.delay:raise SafetyFault('commit_horizon_exceeded','More actions were committed than the frozen prefix covers')
                        current=Chunk(result['normalized'],result['native'],origins,next_chunk_id,job.observation.seq,spent)
                        await profile('inference_complete',latency_ms=elapsed*1000,delay_ticks=ticks,chunk_id=next_chunk_id)
                        if job.previous_origins is None:next_tick=time.monotonic()
                    last_input_ns=job.observation.received_ns;job=None
                if current is None and job is None:
                    obs=await get_fresh(after_ns=last_input_ns)
                    job=Job(spawn_predict(obs),obs,time.monotonic_ns(),action_count,None,0)
                    await asyncio.sleep(min(self.period,.01));continue
                if current is not None and job is None and current.index>=self.shift:
                    obs=await get_fresh(after_ns=last_input_ns)
                    s=current.index
                    if s+self.delay>30:raise SafetyFault('chunk_exhausted','Not enough uncommitted horizon remains to replan')
                    prev=np.concatenate([current.normalized[s:],np.zeros((s,78),np.float32)])
                    origins=current.origins[s:]+[current.origins[-1]]*s
                    job=Job(spawn_predict(obs,prev,self.delay),obs,time.monotonic_ns(),action_count,origins,self.delay)
                if current is not None:
                    if mode=='execute' and not self.latency_ok:raise SafetyFault('latency_not_ready','Profile does not meet the execution latency budget')
                    now=time.monotonic()
                    if now>=next_tick:
                        if now-next_tick>self.period:
                            if mode=='execute':raise SafetyFault('control_tick_overrun','Control sender missed a full tick; refusing catch-up burst')
                            next_tick=now
                        if current.index>=30:raise SafetyFault('chunk_exhausted','Action chunk exhausted; last action is never repeated')
                        i=current.index;origin=current.origins[i];obs=origin.observation;issued_ns=time.time_ns()
                        server_elapsed=(time.monotonic_ns()-obs.received_ns)/1e6
                        if server_elapsed>(30+8)/30*1000:raise SafetyFault('source_observation_expired','Action derives from an expired observation')
                        action_count+=1
                        await send({'type':'action','protocol':PROTOCOL,'session_id':session_id,'action_seq':action_count,'source_observation_seq':obs.seq,'source_client_sent_monotonic_ns':obs.sent_ns,'action':current.native[i].tolist(),'actions_normalized':False,'checkpoint_sha256':self.backend.checkpoint_sha256,'stats_sha256':self.backend.contract['stats_sha256'],'contract_sha256':self.backend.contract_sha256,'chunk_id':current.chunk_id,'chunk_index':i,'origin_chunk_id':origin.origin_chunk_id,'model_observation_seq':current.model_observation_seq,'generation_timestamp_ns':origin.generation_ns,'issued_timestamp_ns':issued_ns,'ttl_ms':self.ttl_ms,'dry_run':mode!='execute','latency_ms':self.samples[-1]*1000,'latency_ok':bool(self.latency_ok),'server_elapsed_since_source_observation_received_ms':server_elapsed})
                        current.index+=1;next_tick+=self.period
                await asyncio.sleep(min(self.period/3,.01))
        claimed=False
        try:
            hello=await asyncio.wait_for(ws.receive_json(),timeout=5)
            if hello.get('type')!='hello' or hello.get('protocol')!=PROTOCOL:raise SafetyFault('protocol_mismatch','First frame must be cart78.v1 hello')
            session_id=hello.get('session_id');mode=hello.get('mode')
            if not isinstance(session_id,str) or not 8<=len(session_id)<=128 or session_id in self.seen_sessions:raise SafetyFault('invalid_session','Use a new session_id on every connection')
            if mode not in {'dry_run','execute'}:raise SafetyFault('invalid_mode','Mode must be dry_run or execute')
            expected=hello.get('expected',{})
            for k,actual in [('checkpoint_sha256',self.backend.checkpoint_sha256),('stats_sha256',self.backend.contract['stats_sha256'])]:
                if expected.get(k)!=actual:raise SafetyFault('identity_mismatch',f'Expected {k} does not match')
            if expected.get('contract_sha256',self.backend.contract_sha256)!=self.backend.contract_sha256:raise SafetyFault('identity_mismatch','Contract SHA mismatch')
            if self.active or self.backend.busy:raise SafetyFault('server_busy','Another client/inference owns the server')
            if mode=='execute' and not self.info()['ready']:raise SafetyFault('execution_not_ready','Run dry-run profiling first; real backend, --allow-execution and latency_ok are required')
            self.active=True;claimed=True;self.seen_sessions.append(session_id);self.session_count+=1
            if mode=='dry_run':self.fault=None
            await send({'type':'hello_ack','protocol':PROTOCOL,'session_id':session_id,'deployment':self.info()})
            receiver=asyncio.create_task(receive());control=asyncio.create_task(run_control())
            done,pending=await asyncio.wait({receiver,control},return_when=asyncio.FIRST_COMPLETED)
            for task in done:task.result()
        except Exception as exc:
            from starlette.websockets import WebSocketDisconnect
            if not isinstance(exc,WebSocketDisconnect):
                await fail(exc.code if isinstance(exc,SafetyFault) else 'server_error',str(exc))
        finally:
            for task in [receiver,control]:
                if task is not None and not task.done():task.cancel()
            await asyncio.gather(*[t for t in [receiver,control] if t is not None],return_exceptions=True)
            # A cancelled await cannot cancel CUDA work in its worker thread.
            # Keep ownership until every original inference is done; no reconnect race.
            if inflight:await asyncio.gather(*[asyncio.shield(t) for t in list(inflight)],return_exceptions=True)
            if claimed and hasattr(self.backend,'end_session'):
                try:await asyncio.to_thread(self.backend.end_session)
                except Exception:pass
            if claimed:self.active=False
            try:await ws.close(code=1000)
            except Exception:pass
