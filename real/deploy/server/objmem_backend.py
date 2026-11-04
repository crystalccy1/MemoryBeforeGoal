"""SAM3-v2 backend for psi0-objmem-cart78-v1 bundles.

Model path: the same a7 Server loader, parser, predict_action / predict_action_with_training_rtc_flow and
post-processing as the cart78 NativeBackend. The only differences:
- the loader builds Psi0ObjMemModel (the bundled SAM3-v2 model class), and every action-header call receives the
  memory of model.memory_context();
- the policy's own frozen Qwen3-VL is also the native keyframe encoder. Its 315 vision tensors are bit-identical
  to the m0 weights that built the training bank.

Memory path (online_memory.SlicedObjMem, same computation as the offline bank):
- Every session starts with a priming frame (raw frame 0), which is tracked synchronously before any action.
- Then one frame every `stride` raw frames (default 24 = 0.8 s) is taken from the observation of a policy call.
- Its SAM3 tracking runs in slices (push, one SAM3 step per prompt, finish) on a worker thread. A slice starts only
  when it is expected to end before the next policy call, which the runtime issues every exec-horizon ticks
  (12 = 0.4 s). One GPU lock serializes slices and policy calls, so SAM3 never runs concurrently with the policy.
- A policy call uses the memory of the latest finished frame and never waits for tracking.
  With stride 24 a frame is usually finished within two policy periods, so memory lags by 24-36 raw frames.
- SAM3 runs with its adapter's deterministic kernel flags (as in the parity tests); policy calls run with the
  process's original flags.
"""
from __future__ import annotations
import collections, copy, hashlib, json, os, sys, threading, time
from pathlib import Path
import numpy as np
from fastapi import FastAPI
try:
    from .backend import BackendError, state_vector, image_array, restore_prefix, portable_config
except ImportError:
    from backend import BackendError, state_vector, image_array, restore_prefix, portable_config

SELECTOR = {'max_frames': 7, 'event_fraction': 0.75, 'event_min_gap': 5, 'score_threshold': 0.35}


def import_objmem(bundle):
    source = (bundle / 'source/src').resolve()
    if any(k == 'psi' or k.startswith('psi.') for k in sys.modules):
        raise BackendError('A psi module was already imported; use a fresh server process')
    sys.dont_write_bytecode = True; sys.path.insert(0, str(source))
    os.environ['HF_HUB_OFFLINE'] = '1'; os.environ['TRANSFORMERS_OFFLINE'] = '1'
    from psi.config.train.finetune_real_psi0_objmem_config import DynamicLaunchConfig
    import psi.deploy.serve_psi0_sonic as native
    import psi.models.psi0 as model_module
    import psi.models.psi0_objmem as objmem_module
    from psi.models.keyframe_native_qwen import QwenVisionMergerNativeFrameProducer
    cfg = portable_config(bundle)
    original_defaults = native.apply_legacy_model_config_defaults
    native.parse_args_to_tyro_config = lambda ignored: DynamicLaunchConfig.model_validate(copy.deepcopy(cfg))
    native.apply_legacy_model_config_defaults = lambda conf: original_defaults(copy.deepcopy(cfg))
    model_module.QWEN3VL_VARIANT = str(bundle / 'assets/qwen')
    for name, module in list(sys.modules.items()):
        if name == 'psi' or name.startswith('psi.'):
            file = getattr(module, '__file__', None)
            if file and not Path(file).resolve().is_relative_to(source):
                raise BackendError(f'Non-bundled psi import: {name}')
    return native, model_module, objmem_module, QwenVisionMergerNativeFrameProducer, DynamicLaunchConfig.model_validate(cfg)


def torch_flags():
    import torch
    return {'cudnn_benchmark': bool(torch.backends.cudnn.benchmark),
            'cudnn_deterministic': bool(torch.backends.cudnn.deterministic),
            'deterministic': bool(torch.are_deterministic_algorithms_enabled()),
            'deterministic_warn_only': bool(torch.is_deterministic_algorithms_warn_only_enabled()),
            'matmul_tf32': bool(torch.backends.cuda.matmul.allow_tf32), 'cudnn_tf32': bool(torch.backends.cudnn.allow_tf32),
            'float32_matmul_precision': torch.get_float32_matmul_precision(),
            'sdp': [bool(torch.backends.cuda.flash_sdp_enabled()), bool(torch.backends.cuda.mem_efficient_sdp_enabled()),
                    bool(torch.backends.cuda.math_sdp_enabled()), bool(torch.backends.cuda.cudnn_sdp_enabled())]}


def set_torch_flags(flags):
    import torch
    torch.backends.cudnn.benchmark = flags['cudnn_benchmark']
    torch.backends.cudnn.deterministic = flags['cudnn_deterministic']
    torch.use_deterministic_algorithms(flags['deterministic'], warn_only=flags['deterministic_warn_only'])
    torch.backends.cuda.matmul.allow_tf32 = flags['matmul_tf32']
    torch.backends.cudnn.allow_tf32 = flags['cudnn_tf32']
    torch.set_float32_matmul_precision(flags['float32_matmul_precision'])
    flash, efficient, math_sdp, cudnn_sdp = flags['sdp']
    torch.backends.cuda.enable_flash_sdp(flash); torch.backends.cuda.enable_mem_efficient_sdp(efficient)
    torch.backends.cuda.enable_math_sdp(math_sdp); torch.backends.cuda.enable_cudnn_sdp(cudnn_sdp)


def torch_alloc_stats():
    import torch
    s = torch.cuda.memory_stats()
    return {'alloc_retries': s.get('num_alloc_retries', 0), 'device_alloc': s.get('num_device_alloc', 0),
            'device_free': s.get('num_device_free', 0), 'reserved_mb': s.get('reserved_bytes.all.current', 0) // 2**20}


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


class ObjMemBackend:
    mock = False

    def __init__(self, bundle, manifest, device='cuda:0', inference_steps=8, attention_backend='sdpa', *, sam3_root,
                 stride=24, exec_horizon=12, memory_log=None, margin_ms=15.0, lock_wait_ms=30.0,
                 share_backbone=False):
        import torch
        self.bundle = Path(bundle).resolve(); self.manifest = manifest; self.contract = manifest['contract']
        self.inference_steps = inference_steps
        self.checkpoint_sha256 = manifest['checkpoint']['sha256']; self.contract_sha256 = manifest['contract_sha256']
        self.lock = threading.Lock(); self.busy = False
        self.gpu = threading.Lock(); self.cv = threading.Condition()
        self.stride = int(stride); self.period = int(exec_horizon) / 30; self.margin = margin_ms / 1000
        self.lock_wait = lock_wait_ms / 1000
        self.share_backbone = bool(share_backbone)
        if not 12 <= self.stride <= 60:
            raise BackendError('memory stride must be 12..60 raw frames')
        self.tolerance = 2  # a policy observation within 2 raw frames of the next memory target is taken
        spec = json.loads((self.bundle / 'memory/spec.json').read_text())
        self.spec = spec
        self.prompts = list(spec['objects']['prompts'])
        selector = {k: spec['selector'][k] for k in SELECTOR}
        if selector != SELECTOR or spec['native_keyframes']['max_frames'] != 7 or spec['objects']['object_slots'] != len(self.prompts):
            raise BackendError(f'Unsupported SAM3-v2 memory spec: {selector}')
        native, model_module, objmem_module, Producer, cfg = import_objmem(self.bundle)
        if cfg.model.object_slots != len(self.prompts):
            raise BackendError('object_slots differs from the prompt count')
        self.native_module = native; self.objmem_module = objmem_module
        import transformers.utils as transform_utils
        if not torch.cuda.is_available():
            raise BackendError('Native backend requires allocated CUDA hardware')
        self.device = torch.device(device); torch.cuda.set_device(self.device)
        original_flash_check = transform_utils.is_flash_attn_2_available
        selected = 'sdpa' if attention_backend in {'sdpa', 'auto'} else 'flash_attention_2'
        if selected == 'flash_attention_2' and not original_flash_check():
            raise BackendError('flash_attention_2 explicitly requested but unavailable')
        transform_utils.is_flash_attn_2_available = lambda: selected == 'flash_attention_2'
        # Psi0Model.from_pretrained builds `Psi0Model(...)` by its module-global name; point that name at the
        # bundled SAM3-v2 class while the a7 Server loads, so its strict action-header load covers the memory weights.
        original_class = model_module.Psi0Model
        model_module.Psi0Model = objmem_module.Psi0ObjMemModel
        try:
            self.server = native.Server(policy='psi0', run_dir=self.bundle / 'run', ckpt_step=manifest['checkpoint']['step'],
                                        device=device, enable_rtc=True, action_exec_horizon=15, dashboard=False)
        finally:
            model_module.Psi0Model = original_class
            transform_utils.is_flash_attn_2_available = original_flash_check
        model = self.server.model
        if type(model) is not objmem_module.Psi0ObjMemModel:
            raise BackendError(f'Loaded {type(model).__name__}, not Psi0ObjMemModel')
        if tuple(model.action_header.objmem_contract_version.tolist()) != objmem_module.OBJMEM_CONTRACT:
            raise BackendError('Object-memory contract version mismatch')
        model.requires_grad_(False)
        self.attention_backend = selected
        self.actual_attention = {k: str(getattr(v, '_attn_implementation', None)) for k, v in [
            ('vlm', model.vlm_model.config), ('text', model.vlm_model.config.text_config),
            ('vision', model.vlm_model.config.vision_config)]}
        if any(v != selected for v in self.actual_attention.values()):
            raise BackendError(f'Unexpected actual attention backend: {self.actual_attention}')
        if self.server.controller is not None or self.server._control_loop_started:
            raise BackendError('Upstream unexpectedly started a controller thread')
        self.fields = tuple(objmem_module.MEMORY_FIELDS)
        self._install_memory_injection(model)
        self.producer = Producer(model=model.vlm_model, processor=model.vlm_processor, device=self.device)
        self.runtime_versions = {}
        from importlib.metadata import version
        for name in ['torch', 'transformers', 'diffusers', 'accelerate', 'safetensors']:
            self.runtime_versions[name] = version(name)
        self.runtime_versions['cuda_device'] = torch.cuda.get_device_name(self.device)
        # SAM3: the bundle's pinned pipeline code, the vendor checkout and weights from --sam3-root.
        sam_src = (self.bundle / 'memory/pipeline/sam3_source').resolve()
        sys.path.insert(0, str(sam_src))
        import sam_adapter, selector as selector_module, run as run_module
        for mod in (sam_adapter, selector_module, run_module):
            if not Path(mod.__file__).resolve().is_relative_to(sam_src):
                raise BackendError(f'Non-bundled SAM3 pipeline import: {mod.__name__} from {mod.__file__}')
        pin = spec['sam3']
        weights = Path(sam3_root) / 'weights/sam3' / pin['file']
        vendor = Path(sam3_root) / f"vendor/sam3-{pin['vendor_commit'][:8]}"
        if weights.stat().st_size != pin['bytes'] or file_sha256(weights) != pin['sha256']:
            raise BackendError('SAM3 weights do not match the bundle pin')
        self.policy_flags = torch_flags()
        self.predictor = sam_adapter.create_predictor(weights, vendor)
        self.sam3_flags = torch_flags()
        # The SAM3 tracking predictor enters a bf16 autocast context for the rest of the thread that builds it
        # (sam3_tracking_predictor.py: bf16_context.__enter__()), so the offline tracker and the memory parity
        # test ran SAM3 and the native encoder under it. Memory slices run on other threads and re-enter it.
        # Policy calls run on runtime worker threads without autocast, as in the Psi0 server.
        self.sam3_autocast = bool(torch.is_autocast_enabled('cuda'))
        self.sam3_autocast_dtype = torch.get_autocast_dtype('cuda')
        set_torch_flags(self.policy_flags)
        torch.set_grad_enabled(True)  # create_predictor disabled grad on this thread; policy calls use inference_mode
        # Session state (guarded by self.cv).
        self.session = None; self.session_index = 0; self.t0_ns = None; self.instruction = None
        self.queue = collections.deque(); self.next_target = None
        self.memory_cpu = None; self.memory_info = None
        self.predicting = False; self.policy_waiting = False; self.last_predict_start = None
        self.durations = {k: collections.deque(maxlen=20) for k in ('begin', 'prompt', 'finish')}
        self.warm_durations = {k: [] for k in ('begin', 'prompt', 'finish')}
        self.estimate = {'begin': 0.03, 'prompt': 0.13, 'finish': 0.08}
        self.counters = collections.Counter(); self.last_lag = None; self.lags = collections.deque(maxlen=200)
        self.worker_error = None
        self.log_path = Path(memory_log) if memory_log else Path(os.environ.get('CART78_RUNTIME_DIR', '.')) / 'sam3v2_memory.jsonl'
        self.log_file = open(self.log_path, 'a', buffering=1)
        import gc
        self._gc_start = {}

        def gc_timer(phase, info):
            key = threading.get_ident()
            if phase == 'start':
                self._gc_start[key] = time.perf_counter()
            elif key in self._gc_start:
                ms = (time.perf_counter() - self._gc_start.pop(key)) * 1e3
                if ms >= 10:
                    self._log({'event': 'gc', 'generation': info.get('generation'), 'ms': ms, 'collected': info.get('collected'),
                               'thread': threading.current_thread().name})
        gc.callbacks.append(gc_timer)
        self._warm_memory()
        gc.collect(); gc.freeze()
        self._log({'event': 'gc_freeze', 'frozen_objects': gc.get_freeze_count()})
        self.worker = threading.Thread(target=self._work, name='sam3v2-memory', daemon=True); self.worker.start()
        self.app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)

    # -- model --------------------------------------------------------------------------------------------------
    def _install_memory_injection(self, model):
        header = model.action_header; forward = header.forward; fields = self.fields

        def forward_with_memory(*args, **kwargs):
            if not any(f in kwargs for f in fields):
                active = model._active_memory
                if active is None:
                    raise BackendError('SAM3-v2 action header called outside memory_context')
                kwargs.update(active)
            began = time.perf_counter()
            try:
                return forward(*args, **kwargs)
            finally:
                if self._header_times is not None:
                    self._header_times.append(time.perf_counter() - began)
        header.forward = forward_with_memory
        self._header_times = None
        # The native and object memory encoders see the same memory tensors at every flow step of one policy call,
        # so their outputs are computed once per call and reused (same inputs, same ops: identical values). This
        # also drops the encoder's per-step GPU->CPU validation syncs.
        self._encoder_memo = None
        for name in ('memory_encoder', 'object_memory_encoder'):
            module = getattr(header, name); original = module.forward

            def memo_forward(*args, _name=name, _original=original, **kwargs):
                memo = self._encoder_memo
                if memo is None:
                    return _original(*args, **kwargs)
                key = (_name, tuple(id(v) for v in kwargs.values()), tuple(id(v) for v in args))
                if key not in memo:
                    memo[key] = _original(*args, **kwargs)
                return memo[key]
            module.forward = memo_forward

    def attach_runtime(self, runtime): self.app.add_api_websocket_route('/ws', runtime.handle)

    def _log(self, record):
        record = {'t': time.time(), **record}
        try:
            self.log_file.write(json.dumps(record, default=float) + '\n')
        except Exception:
            pass

    # -- memory sessions ------------------------------------------------------------------------------------------
    def _new_memory(self, instruction):
        from online_memory import SlicedObjMem
        return SlicedObjMem(self.predictor, self.producer, instruction, self.prompts, self.stride,
                            share_backbone=self.share_backbone)

    def _sam3_call(self, fn):
        import torch
        set_torch_flags(self.sam3_flags)
        try:
            with torch.no_grad(), torch.autocast(device_type='cuda', dtype=self.sam3_autocast_dtype, enabled=self.sam3_autocast):
                out = fn()
            torch.cuda.synchronize(self.device)
            return out
        finally:
            set_torch_flags(self.policy_flags)

    def _warm_new_objects(self, max_objects=8):
        """SAM3 pays ~0.9 s the first time a given number of objects appears in one frame (N=1..6 measured on the
        5090; after that the same event takes ~0.1 s). A live session would pay it inside a slice and delay a policy
        call, so every count up to max_objects is paid here, on synthetic red circles appearing at once and one by one."""
        import cv2
        from sam3_stream import StreamingTracker
        spots = [(80 + 160 * (i % 4), 100 + 160 * (i // 4)) for i in range(max_objects)]

        def circles(n):
            img = np.full((360, 640, 3), 235, np.uint8)
            for x, y in spots[:n]:
                cv2.circle(img, (x, y), 45, (220, 30, 30), -1)
            return img
        seen = {}
        sequences = [[circles(0), circles(0), circles(n), circles(n)] for n in range(1, max_objects + 1)]
        sequences.append([circles(0)] + [circles(n) for n in range(1, max_objects + 1)])
        for seq in sequences:
            tracker = StreamingTracker(self.predictor, ['red circle'], namespace='warm')
            try:
                for frame in seq:
                    t = time.monotonic()
                    self._sam3_call(lambda: (tracker.push(frame), tracker.step_prompt(0)))
                    n = len(tracker.objects[0]); tracker.collect()
                    seen[n] = max(seen.get(n, 0), round((time.monotonic() - t) * 1e3))
            finally:
                tracker.close()
        return seen

    def _warm_memory(self):
        """Compile/allocate SAM3 and the native encoder once, on synthetic frames, before any client connects."""
        rng = np.random.default_rng(0)
        frame = (rng.integers(0, 255, (360, 640, 3))).astype(np.uint8)
        with self.gpu:
            began = time.monotonic()
            new_objects = self._warm_new_objects()
            self._log({'event': 'new_object_warmup', 's': time.monotonic() - began, 'max_ms_by_count': new_objects})
            mem = self._new_memory('warm up')
            try:
                began = time.monotonic()
                self._sam3_call(lambda: mem.process(frame, 0))
                first = time.monotonic() - began
                for raw in (self.stride, 2 * self.stride, 3 * self.stride):
                    t = time.monotonic(); self._sam3_call(lambda: mem.begin(frame, raw)); self._record('begin', time.monotonic() - t, True)
                    for k in range(len(self.prompts)):
                        t = time.monotonic(); self._sam3_call(lambda: mem.step_prompt(k)); self._record('prompt', time.monotonic() - t, True)
                    t = time.monotonic(); self._sam3_call(lambda: (mem.finish(), mem.memory())); self._record('finish', time.monotonic() - t, True)
            finally:
                mem.close()
        self._log({'event': 'memory_warmup', 'first_frame_s': first, 'estimate_s': dict(self.estimate),
                   'sam3_flags': self.sam3_flags, 'policy_flags': self.policy_flags,
                   'sam3_autocast': [self.sam3_autocast, str(self.sam3_autocast_dtype)], 'prompts': self.prompts,
                   'stride': self.stride, 'exec_period_s': self.period, 'share_backbone': self.share_backbone})

    def _record(self, kind, seconds, warm=False):
        (self.warm_durations[kind] if warm else self.durations[kind]).append(seconds)
        # Live slices: the largest of the last 20, with 10% headroom. Warm-up (first compiles) only seeds it.
        base = self.durations[kind] or self.warm_durations[kind]
        self.estimate[kind] = max(base) * 1.1

    @staticmethod
    def _batched(memory):
        return {k: v[None].contiguous() for k, v in memory.items()}

    def begin_session(self, observation):
        """Reset the memory and track the priming frame (raw frame 0) synchronously. No action is produced."""
        image = image_array(observation.payload['image_jpeg_b64'])
        instruction = observation.payload['instruction']
        with self.cv:
            old, self.session = self.session, None
            self.session_index += 1; self.queue.clear(); self.memory_cpu = None; self.memory_info = None
            self.last_predict_start = None; self.lags.clear(); self.last_lag = None
            self.counters = collections.Counter()
        began = time.monotonic()
        with self.gpu:  # waits for an in-flight slice of the old session
            if old is not None:
                old.close()
            mem = self._new_memory(instruction)
            info = self._sam3_call(lambda: mem.process(image, 0))
            memory = self._batched(mem.memory())
        with self.cv:
            # Raw frame 0 is the session's first policy observation: priming takes ~2.5 s with no action, so
            # counting from the priming observation would shift every memory time index by that wait.
            self.session = mem; self.instruction = instruction; self.t0_ns = None
            self.next_target = self.stride - self.tolerance
            self.memory_cpu = memory; self.memory_info = {'frames': 1, 'latest_raw': 0, 'selected_raw': info['selected_raw_frames'],
                                                          'valid_slots': info['valid_slots']}
            self.cv.notify_all()
        self._log({'event': 'session_start', 'session': self.session_index, 'observation_seq': observation.seq,
                   'priming_s': time.monotonic() - began, **info})
        return info

    def end_session(self):
        with self.cv:
            old, self.session = self.session, None
            self.queue.clear(); self.memory_cpu = None; self.memory_info = None; self.last_predict_start = None
        with self.gpu:
            if old is not None:
                old.close()
        self._log({'event': 'session_end', 'session': self.session_index, 'counters': dict(self.counters)})

    # -- worker ---------------------------------------------------------------------------------------------------
    def _plan(self):
        s = self.session
        if s is None:
            return None
        nxt = s.next_slice()
        if nxt is None:
            return (s, 'begin', None) if self.queue else None
        return (s, nxt[0], nxt[1])

    def _fits(self, kind):
        if self.predicting or self.policy_waiting:
            return False
        if self.last_predict_start is None:
            return True
        return time.monotonic() + self.estimate[kind] + self.margin <= self.last_predict_start + self.period

    def _work(self):
        while True:
            with self.cv:
                while True:
                    plan = self._plan()
                    if plan is not None and self._fits(plan[1]):
                        break
                    self.cv.wait(timeout=0.05 if plan is None or self.predicting or self.policy_waiting else 0.01)
            session, kind, k = plan
            with self.gpu:
                with self.cv:
                    if self.session is not session or self.predicting or self.policy_waiting:
                        continue
                    item = self.queue.popleft() if kind == 'begin' else None
                stats0 = torch_alloc_stats()
                began = time.monotonic()
                try:
                    if kind == 'begin':
                        self._sam3_call(lambda: session.begin(item[0], item[1]))
                        result = None
                    elif kind == 'prompt':
                        self._sam3_call(lambda: session.step_prompt(k))
                        result = None
                    else:
                        result = self._sam3_call(lambda: (session.finish(), self._batched(session.memory())))
                except Exception as exc:  # the policy refuses to act on memory that stopped updating
                    with self.cv:
                        self.worker_error = f'{type(exc).__name__}: {exc}'
                    self._log({'event': 'memory_error', 'kind': kind, 'error': self.worker_error})
                    continue
                seconds = time.monotonic() - began
            deadline_slack = None
            with self.cv:
                self._record(kind, seconds)
                if self.last_predict_start is not None:
                    deadline_slack = self.last_predict_start + self.period - time.monotonic()
                if result is not None and self.session is session:
                    info, memory = result
                    self.memory_cpu = memory
                    self.memory_info = {'frames': info['frame_index'] + 1, 'latest_raw': info['raw_frame'],
                                        'selected_raw': info['selected_raw_frames'], 'valid_slots': info['valid_slots']}
                    self.counters['frames'] += 1
                self.counters[f'{kind}_slices'] += 1
                if deadline_slack is not None and deadline_slack < 0:
                    self.counters['slices_past_deadline'] += 1
                self.cv.notify_all()
            stats1 = torch_alloc_stats()
            alloc = {name: stats1[name] - stats0[name] for name in stats0 if stats1[name] != stats0[name]}
            objects = len(session.tracker.objects[k] or []) if kind == 'prompt' and session.tracker.objects else None
            self._log({'event': 'slice', 'kind': kind, 'k': k, 'ms': seconds * 1e3,
                       'slack_ms': None if deadline_slack is None else deadline_slack * 1e3,
                       **({'alloc': alloc} if alloc else {}), **({'objects': objects} if objects is not None else {}),
                       **({'frame': result[0]} if result is not None else {})})

    # -- policy ---------------------------------------------------------------------------------------------------
    def predict(self, observation, previous=None, delay=0):
        import torch
        entered = time.monotonic()
        with self.lock:
            self.busy = True
            try:
                if self.worker_error:
                    raise BackendError(f'SAM3-v2 memory runtime failed: {self.worker_error}')
                state = state_vector(observation.payload['state_model43']); image = image_array(observation.payload['image_jpeg_b64'])
                instruction = observation.payload['instruction']
                with self.cv:
                    if self.session is None or self.memory_cpu is None:
                        raise BackendError('SAM3-v2 memory session was not primed')
                    if instruction != self.instruction:
                        raise BackendError('Instruction differs from the primed memory session')
                    if self.t0_ns is None:
                        self.t0_ns = observation.sent_ns
                    raw = int(round((observation.sent_ns - self.t0_ns) * 30 / 1e9))
                    memory, minfo = self.memory_cpu, dict(self.memory_info)
                    enqueued = False
                    if raw >= self.next_target:
                        if not self.queue:
                            self.queue.append((image, raw)); self.next_target = raw + self.stride - self.tolerance; enqueued = True
                            self.cv.notify_all()
                        else:
                            self.counters['backlog_waits'] += 1
                    lag = raw - minfo['latest_raw']; self.last_lag = lag; self.lags.append(lag)
                    self.policy_waiting = True
                image_key = self.server.launch_cfg.data.transform.repack.image_keys[0]
                request = self.native_module.RequestMessage(image={image_key: image}, instruction=instruction, history={},
                                                            state={'states': state}, condition={}, gt_action=None,
                                                            dataset_name='r07', timestamp=str(observation.sent_ns))
                parsed = self.server._parse_obs_payload(request.serialize())
                if np.asarray(parsed['obs']).shape != (1, 1, 43):
                    raise BackendError('Native parser state shape mismatch')
                common = {'observations': parsed['imgs'], 'states': torch.from_numpy(parsed['obs']).to(self.device),
                          'instructions': parsed['text_instructions'], 'num_inference_steps': self.inference_steps,
                          'traj2ds': None, 'pooled_projections': parsed.get('pooled_projections')}
                waited = time.monotonic()
                # A slice normally ends before the policy call (it only starts if it fits); if one overruns, the
                # policy does not wait for it beyond lock_wait and runs on the same stream concurrently.
                owned = self.gpu.acquire(timeout=self.lock_wait)
                try:
                    with self.cv:
                        self.policy_waiting = False; self.predicting = True; self.last_predict_start = entered
                        if not owned:
                            self.counters['policy_overlapped_slice'] += 1
                    gpu_wait = time.monotonic() - waited
                    if not owned:
                        set_torch_flags(self.policy_flags)
                    try:
                        memory_device = {k: v.to(self.device) for k, v in memory.items()}
                        torch.cuda.synchronize(self.device)
                        self._encoder_memo = {}; self._header_times = []
                        with torch.inference_mode(), self.server.model.memory_context(memory_device):
                            if previous is None:
                                out = self.server.model.predict_action(**common)
                            else:
                                if not 1 <= delay < 8:
                                    raise BackendError('RTC delay must stay in1..7; no model-side clamping allowed')
                                out = self.server.model.predict_action_with_training_rtc_flow(
                                    **common, prev_actions=torch.from_numpy(previous[None]).to(self.device),
                                    inference_delay=delay, max_delay=8)
                        torch.cuda.synchronize(self.device)
                    finally:
                        memo_entries = len(self._encoder_memo or {}); self._encoder_memo = None
                        header_cpu = [round(x * 1e3, 1) for x in (self._header_times or [])]; self._header_times = None
                        with self.cv:
                            self.predicting = False; self.cv.notify_all()
                finally:
                    if owned:
                        self.gpu.release()
                normalized = out[0].float().detach().cpu().numpy()
                if normalized.shape != (30, 78) or not np.isfinite(normalized).all():
                    raise BackendError('Invalid native model output')
                drift = 0.0
                if previous is not None:
                    normalized, drift = restore_prefix(normalized, previous, delay)
                native = np.asarray(self.server._postprocess_action(normalized), dtype=np.float32)
                if native.shape != (30, 78) or not np.isfinite(native).all():
                    raise BackendError('Invalid denormalized model output')
                total = time.monotonic() - entered
                self._log({'event': 'predict', 'seq': observation.seq, 'raw': raw, 'rtc': previous is not None,
                           'memory_frames': minfo['frames'], 'memory_latest_raw': minfo['latest_raw'], 'lag': lag,
                           'selected_raw': minfo['selected_raw'], 'valid_slots': minfo['valid_slots'],
                           'enqueued': enqueued, 'gpu_wait_ms': gpu_wait * 1e3, 'overlap': not owned, 'encoder_memo': memo_entries, 'header_cpu_ms': header_cpu, 'latency_ms': total * 1e3})
                return {'normalized': normalized, 'native': native, 'prefix_raw_max_error': drift, 'prefix_restored_max_error': 0.0,
                        'normalized_outside_pm1_fraction': float((np.abs(normalized) > 1).mean()),
                        'generation_timestamp_ns': time.time_ns()}
            finally:
                with self.cv:
                    self.policy_waiting = False
                self.busy = False

    def memory_status(self):
        with self.cv:
            lags = list(self.lags)
            return {'policy_memory': 'sam3v2', 'prompts': self.prompts, 'stride': self.stride,
                    'share_backbone': self.share_backbone,
                    'exec_period_s': self.period, 'session_active': self.session is not None,
                    'frames': None if self.memory_info is None else self.memory_info['frames'],
                    'latest_raw': None if self.memory_info is None else self.memory_info['latest_raw'],
                    'last_lag': self.last_lag, 'max_lag': max(lags) if lags else None,
                    'slice_estimate_ms': {k: round(v * 1e3, 1) for k, v in self.estimate.items()},
                    'counters': dict(self.counters), 'worker_error': self.worker_error}
