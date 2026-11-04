#!/usr/bin/env python3
"""Serve a verified SAM3-v2 (psi0-objmem-cart78-v1) bundle over cart78.v1 on loopback by default; never connects to a robot."""
from pathlib import Path
import argparse,json,sys
try:
    from .bundle import verify_bundle
except ImportError:
    from bundle import verify_bundle

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bundle',type=Path)
    p.add_argument('--expected-checkpoint-sha256')
    p.add_argument('--host',default='127.0.0.1');p.add_argument('--port',type=int,default=8000)
    p.add_argument('--device',default='cuda:0');p.add_argument('--inference-steps',type=int,default=8)
    p.add_argument('--attention-backend',choices=['sdpa','flash_attention_2','auto'],default='sdpa')
    p.add_argument('--rtc-delay',type=int,default=7);p.add_argument('--min-exec-horizon',type=int,default=15)
    p.add_argument('--observation-timeout-ms',type=float,default=250);p.add_argument('--action-ttl-ms',type=int,default=100)
    p.add_argument('--allow-execution',action='store_true',help='Allow execute clients only after successful dry-run profiling; default is dry-run only')
    p.add_argument('--verify-only',action='store_true');p.add_argument('--import-check',action='store_true',help='Verify and import the isolated source/config, without model weights/GPU')
    p.add_argument('--mock',action='store_true',help='CPU protocol tests only; execution is always prohibited')
    p.add_argument('--mock-delay-ms',type=float,default=5)
    p.add_argument('--sam3-root',type=Path,required=False,help='SAM3 vendor checkout and weights (pinned by the bundle memory spec)')
    p.add_argument('--memory-stride',type=int,default=24,help='raw 30 Hz frames between SAM3-v2 memory frames')
    p.add_argument('--memory-log',type=Path,help='JSONL log of memory slices and policy calls (default: private runtime dir)')
    p.add_argument('--sam3-share-backbone',action=argparse.BooleanOptionalAction,default=True,help='reuse the SAM3 image backbone of a frame across its prompt sessions (exact: every object matched the offline tracker on A100)')
    a=p.parse_args()
    if not 1<=a.inference_steps<=100:p.error('--inference-steps must be1..100')
    if a.mock and (a.bundle or a.allow_execution or a.verify_only or a.import_check):p.error('--mock cannot be combined with a bundle, execution, or bundle checks')
    if not a.mock and not a.bundle:p.error('--bundle is required for a real service')
    manifest=None
    if not a.mock:
        manifest=verify_bundle(a.bundle,a.expected_checkpoint_sha256)
        print(json.dumps({'bundle_verified':True,'step':manifest['checkpoint']['step'],'checkpoint_sha256':manifest['checkpoint']['sha256']}),flush=True)
        if a.verify_only:return
    if __package__:
        from .backend import MockBackend
        from .objmem_backend import ObjMemBackend,import_objmem
    else:
        from backend import MockBackend
        from objmem_backend import ObjMemBackend,import_objmem
    if a.import_check:
        native,model,objmem,producer,config=import_objmem(a.bundle.resolve())
        print(json.dumps({'import_check':True,'action_dim':config.model.action_dim,'state_dim':config.model.odim,'gpu_model_loaded':False}));return
    if __package__:
        from .runtime import DeploymentRuntime
    else:
        from runtime import DeploymentRuntime
    backend=MockBackend(a.mock_delay_ms/1000) if a.mock else ObjMemBackend(a.bundle,manifest,a.device,a.inference_steps,a.attention_backend,sam3_root=a.sam3_root,stride=a.memory_stride,exec_horizon=a.min_exec_horizon,memory_log=a.memory_log,share_backbone=a.sam3_share_backbone)
    app=DeploymentRuntime(backend,a.allow_execution,a.rtc_delay,a.min_exec_horizon,a.observation_timeout_ms,a.action_ttl_ms).app
    import uvicorn
    uvicorn.run(app,host=a.host,port=a.port,ws_max_size=2_100_000,ws_ping_interval=10,ws_ping_timeout=10,log_level='info')
if __name__=='__main__':main()
