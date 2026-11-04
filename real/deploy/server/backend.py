"""Isolated a7 model/server backend and a CPU-only protocol-test backend."""
from __future__ import annotations
import base64,copy,hashlib,io,json,os,sys,threading,time
from pathlib import Path
import numpy as np
from PIL import Image
from fastapi import WebSocket,FastAPI
try:
    from .bundle import PROTOCOL,json_sha
except ImportError:
    from bundle import PROTOCOL,json_sha

class BackendError(RuntimeError):pass

def state_vector(value):
    a=np.asarray(value)
    if a.dtype.kind not in 'iuf':raise BackendError('State entries must be JSON numbers, not strings or booleans')
    a=a.astype(np.float32)
    if a.shape==(1,43):a=a[0]
    if a.shape!=(43,) or not np.isfinite(a).all():raise BackendError('state_model43 must be finite shape(43,) or(1,43)')
    return a.copy()

def image_array(encoded):
    if not isinstance(encoded,str) or len(encoded)>2_000_000:raise BackendError('JPEG payload size/type invalid')
    try:
        b=base64.b64decode(encoded,validate=True)
        image=Image.open(io.BytesIO(b))
        if image.format!='JPEG' or image.size!=(640,360):raise BackendError('Expected a640x360 JPEG image')
        image.load();return np.asarray(image.convert('RGB'))
    except BackendError:raise
    except Exception as e:raise BackendError('Invalid JPEG payload') from e

def restore_prefix(raw,previous,delay,max_delay=8):
    raw=np.asarray(raw,dtype=np.float32);previous=np.asarray(previous,dtype=np.float32)
    if raw.shape!=(30,78) or previous.shape!=(30,78):raise BackendError('RTC chunks must have shape30x78')
    if not 1<=delay<min(30,max_delay):raise BackendError('RTC delay exceeds the trained prefix contract; no clamping')
    if not(np.isfinite(raw).all() and np.isfinite(previous).all()):raise BackendError('Nonfinite RTC chunk')
    drift=float(np.max(np.abs(raw[:delay]-previous[:delay])))
    result=raw.copy();result[:delay]=previous[:delay]
    if not np.array_equal(result[:delay],previous[:delay]) or not np.array_equal(result[delay:],raw[delay:]):raise BackendError('Prefix adapter invariant failed')
    return result,drift

def portable_config(bundle):
    cfg=json.loads((bundle/'run/run_config.json').read_text())
    cfg['model']['model_name_or_path']=str(bundle/'assets/qwen')
    cfg['model']['pretrained_action_header_path']=None
    cfg['data']['root_dir']=str(bundle/'__no_dataset_needed__')
    cfg['data']['transform']['field']['stat_path']=str(bundle/'__embedded_normalization_only__')
    cfg['train']['resume_from_checkpoint']=None
    return cfg

def import_native(bundle):
    source=(bundle/'source/src').resolve()
    if any(k=='psi' or k.startswith('psi.') for k in sys.modules):raise BackendError('A psi module was already imported; use a fresh server process')
    sys.dont_write_bytecode=True;sys.path.insert(0,str(source))
    os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
    from psi.config.train.finetune_real_psi0_config import DynamicLaunchConfig
    import psi.deploy.serve_psi0_sonic as native
    import psi.models.psi0 as model_module
    cfg=portable_config(bundle)
    original_defaults=native.apply_legacy_model_config_defaults
    # Server still runs its actual a7 constructor/model loader. Only config-path
    # resolution is isolated from the original training machine and argv paths.
    native.parse_args_to_tyro_config=lambda ignored:DynamicLaunchConfig.model_validate(copy.deepcopy(cfg))
    native.apply_legacy_model_config_defaults=lambda conf:original_defaults(copy.deepcopy(cfg))
    model_module.QWEN3VL_VARIANT=str(bundle/'assets/qwen')
    for name,module in list(sys.modules.items()):
        if name=='psi' or name.startswith('psi.'):
            file=getattr(module,'__file__',None)
            if file and not Path(file).resolve().is_relative_to(source):raise BackendError(f'Non-bundled psi import: {name}')
    return native,model_module,DynamicLaunchConfig.model_validate(cfg)

class NativeBackend:
    mock=False
    def __init__(self,bundle,manifest,device='cuda:0',inference_steps=8,attention_backend='sdpa'):
        import torch
        self.bundle=Path(bundle).resolve();self.manifest=manifest;self.contract=manifest['contract'];self.inference_steps=inference_steps
        self.checkpoint_sha256=manifest['checkpoint']['sha256'];self.contract_sha256=manifest['contract_sha256']
        self.lock=threading.Lock();self.busy=False
        native,model_module,cfg=import_native(self.bundle)
        self.native_module=native
        import transformers.utils as transform_utils
        if not torch.cuda.is_available():raise BackendError('Native backend requires allocated CUDA hardware')
        self.device=torch.device(device);torch.cuda.set_device(self.device)
        original_flash_check=transform_utils.is_flash_attn_2_available
        selected='sdpa' if attention_backend in {'sdpa','auto'} else 'flash_attention_2'
        if selected=='flash_attention_2' and not original_flash_check():raise BackendError('flash_attention_2 explicitly requested but unavailable')
        transform_utils.is_flash_attn_2_available=lambda:selected=='flash_attention_2'
        try:
            self.server=native.Server(policy='psi0',run_dir=self.bundle/'run',ckpt_step=manifest['checkpoint']['step'],device=device,enable_rtc=True,action_exec_horizon=15,dashboard=False)
        finally:transform_utils.is_flash_attn_2_available=original_flash_check
        self.server.model.requires_grad_(False)
        self.attention_backend=selected
        self.actual_attention={k:str(getattr(v,'_attn_implementation',None)) for k,v in [('vlm',self.server.model.vlm_model.config),('text',self.server.model.vlm_model.config.text_config),('vision',self.server.model.vlm_model.config.vision_config)]}
        if any(v!=selected for v in self.actual_attention.values()):raise BackendError(f'Unexpected actual attention backend: {self.actual_attention}')
        self.runtime_versions={}
        from importlib.metadata import version
        for name in ['torch','transformers','diffusers','accelerate','safetensors']:
            self.runtime_versions[name]=version(name)
        self.runtime_versions['cuda_device']=torch.cuda.get_device_name(self.device)
        # The real a7 constructor is used only for its loader/parser/model.
        # Its app is never served, so no legacy routes or scheduler can bypass cart78.
        if self.server.controller is not None or self.server._control_loop_started:
            raise BackendError('Upstream unexpectedly started a controller thread')
        self.app=FastAPI(docs_url=None,redoc_url=None,openapi_url=None)

    def attach_runtime(self,runtime):self.app.add_api_websocket_route('/ws',runtime.handle)

    def predict(self,observation,previous=None,delay=0):
        import torch
        with self.lock:
            self.busy=True
            try:
                state=state_vector(observation.payload['state_model43']);image=image_array(observation.payload['image_jpeg_b64'])
                instruction=observation.payload['instruction']
                image_key=self.server.launch_cfg.data.transform.repack.image_keys[0]
                request=self.native_module.RequestMessage(image={image_key:image},instruction=instruction,history={},state={'states':state},condition={},gt_action=None,dataset_name='r07',timestamp=str(observation.sent_ns))
                # Flat state deliberately enters the real parser, preventing its
                # historical(1,43)->(1,1,1,43) expansion bug. No joint reordering here.
                parsed=self.server._parse_obs_payload(request.serialize())
                if np.asarray(parsed['obs']).shape!=(1,1,43):raise BackendError('Native parser state shape mismatch')
                common={'observations':parsed['imgs'],'states':torch.from_numpy(parsed['obs']).to(self.device),'instructions':parsed['text_instructions'],'num_inference_steps':self.inference_steps,'traj2ds':None,'pooled_projections':parsed.get('pooled_projections')}
                torch.cuda.synchronize(self.device)
                with torch.inference_mode():
                    if previous is None:out=self.server.model.predict_action(**common)
                    else:
                        if not 1<=delay<8:raise BackendError('RTC delay must stay in1..7; no model-side clamping allowed')
                        out=self.server.model.predict_action_with_training_rtc_flow(**common,prev_actions=torch.from_numpy(previous[None]).to(self.device),inference_delay=delay,max_delay=8)
                torch.cuda.synchronize(self.device)
                normalized=out[0].float().detach().cpu().numpy()
                if normalized.shape!=(30,78) or not np.isfinite(normalized).all():raise BackendError('Invalid native model output')
                drift=0.0
                if previous is not None:normalized,drift=restore_prefix(normalized,previous,delay)
                native=np.asarray(self.server._postprocess_action(normalized),dtype=np.float32)
                if native.shape!=(30,78) or not np.isfinite(native).all():raise BackendError('Invalid denormalized model output')
                return {'normalized':normalized,'native':native,'prefix_raw_max_error':drift,'prefix_restored_max_error':0.0,'normalized_outside_pm1_fraction':float((np.abs(normalized)>1).mean()),'generation_timestamp_ns':time.time_ns()}
            finally:self.busy=False

class MockBackend:
    """Protocol tests only. Its identity and mock=true cannot pass execution gates."""
    mock=True
    def __init__(self,delay_seconds=0.005):
        from fastapi import FastAPI
        self.app=FastAPI(docs_url=None,redoc_url=None,openapi_url=None);self.delay_seconds=delay_seconds;self.inference_steps=8;self.attention_backend='mock';self.actual_attention={};self.runtime_versions={};self.busy=False
        self.checkpoint_sha256='1'*64
        self.contract={'protocol':PROTOCOL,'state_dim':43,'action_dim':78,'horizon':30,'action_tick_hz':30,'state_order':'model_urdf','action_order':'token64+left_model7+right_model7','actions_normalized':False,'max_delay':8,'stats_sha256':'2'*64,'controller_encoder_sha256':'013ab0287236aa2721e13f1e936d699db982302d0de0bfcdae76d5c3245362d3','controller_decoder_sha256':'c7241a123eaa36b5d64bad19540efde93cac1ad443bd4572fd12ca99898118ed','controller_observation_config_sha256':'466d05947c78af6c76388adfb86e3a2a77b2a1d921a64883ed3d085ebf58de1b'}
        self.contract_sha256=json_sha(self.contract)
    def attach_runtime(self,runtime):self.app.add_api_websocket_route('/ws',runtime.handle)
    def predict(self,observation,previous=None,delay=0):
        self.busy=True
        try:
            state_vector(observation.payload['state_model43']);image_array(observation.payload['image_jpeg_b64'])
            time.sleep(self.delay_seconds)
            a=np.full((30,78),(observation.seq%30)/1000,dtype=np.float32);drift=0.0
            if previous is not None:a,drift=restore_prefix(a,previous,delay)
            native=a.copy();native[:,64:]=0.0
            return {'normalized':a,'native':native,'prefix_raw_max_error':drift,'prefix_restored_max_error':0.0,'normalized_outside_pm1_fraction':0.0,'generation_timestamp_ns':time.time_ns()}
        finally:self.busy=False
