"""Portable, content-checked inference bundle. Standard library only."""
from __future__ import annotations
import hashlib,json,math,os,re,stat,struct
from pathlib import Path,PurePosixPath

SOURCE_COMMIT='a7db4fbb9e3ed55bb46978e890224fa3ff723ad9'
FORMAT='psi0-objmem-cart78-v1'
MEMORY_TENSOR=re.compile(
    r'action_header\.(memory_encoder\.(memory_proj|position_proj)\.(weight|bias)'
    r'|object_memory_encoder\.((geometry_proj|memory_proj|position_proj)\.(weight|bias)|slot_embedding)'
    r'|objmem_contract_version'
    r'|transformer_blocks\.[0-5]\.memory_attention\.((k_proj|v_proj|q_proj|out_proj)\.weight|rms\.weight'
    r'|object_gate|object_attention\.((k_proj|v_proj|q_proj|out_proj)\.weight|rms\.weight))'
    r'|transformer_blocks\.[0-5]\.memory_to_scale_shift\.(weight|bias))$')
MEMORY_PIPELINE=('memory/spec.json','memory/tensor_inventory.json','memory/pipeline/sam3_source/sam_adapter.py',
                 'memory/pipeline/sam3_source/selector.py','memory/pipeline/sam3_source/run.py','memory/pipeline/sam3_source/common.py')
TENSOR_SCHEMA_SHA256='a459c86279b60371388dd21ddfa77461272a9870f3549da89bf5c032ef9fb466'
PROTOCOL='cart78.v1'
EXPECTED_MODEL_SHAPES={
 'action_header.action_proj_in.dec_pos':[30,1536],
 'action_header.action_proj_in.ac_proj.0.weight':[78,78],
 'action_header.action_proj_in.ac_proj.2.weight':[1536,78],
 'action_header.action_proj_out.linear.weight':[78,1536],
 'action_header.obs_proj._obs_proc.1.weight':[1536,43],
 'action_header.obs_proj.views_proj.weight':[1536,2048],
}

def canonical_bytes(value):
    return (json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False)+'\n').encode()

def sha256(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):digest.update(block)
    return digest.hexdigest()

def json_sha(value):return hashlib.sha256(canonical_bytes(value)).hexdigest()

def write_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('xb') as f:f.write(canonical_bytes(value));f.flush();os.fsync(f.fileno())

def safe_relative(path):
    p=PurePosixPath(path)
    if p.is_absolute() or not p.parts or any(x in {'.','..',''} for x in p.parts):raise ValueError(f'Unsafe bundle path: {path!r}')
    return p

def regular_file(root,relative):
    rel=safe_relative(relative);p=Path(root).joinpath(*rel.parts)
    # No symlink anywhere in the bundle path, including directories.
    current=Path(root)
    for part in rel.parts:
        current=current/part
        if current.is_symlink():raise ValueError(f'Bundle contains a symlink: {relative}')
    if not stat.S_ISREG(p.stat().st_mode):raise ValueError(f'Not a regular file: {relative}')
    return p

def read_header(path):
    p=Path(path)
    with p.open('rb') as f:
        first=f.read(8)
        if len(first)!=8:raise ValueError('Truncated safetensors length')
        length=struct.unpack('<Q',first)[0]
        if not 2<=length<=16*1024*1024:raise ValueError('Unexpected safetensors header length')
        data=f.read(length)
        if len(data)!=length:raise ValueError('Truncated safetensors header')
        header=json.loads(data)
    tensors={k:v for k,v in header.items() if k!='__metadata__'}
    if not tensors:raise ValueError('Empty model')
    schema_path=Path(__file__).with_name('a7_cart78_tensor_schema.json')
    if sha256(schema_path)!=TENSOR_SCHEMA_SHA256:raise ValueError('Package tensor schema integrity failure')
    expected_tensors=json.loads(schema_path.read_text())['tensors']
    extra=set(tensors)-set(expected_tensors)
    if not set(expected_tensors)<=set(tensors) or len(extra)!=90 or not all(MEMORY_TENSOR.fullmatch(k) for k in extra):raise ValueError('Checkpoint tensor inventory is not the audited Psi0 policy plus the 90 SAM3-v2 memory tensors')
    for key,value in expected_tensors.items():
        if tensors[key]['shape']!=value['shape'] or tensors[key]['dtype']!=value['dtype']:raise ValueError(f'Checkpoint architecture/precision mismatch: {key}')
    offsets=sorted((v['data_offsets'][0],v['data_offsets'][1]) for v in tensors.values())
    end=0
    for start,stop in offsets:
        if start!=end or stop<start:raise ValueError('Non-contiguous/overlapping safetensors storage')
        end=stop
    expected=8+length+end
    if p.stat().st_size!=expected:raise ValueError(f'Incomplete safetensors: expected {expected}, got {p.stat().st_size}')
    for key,shape in EXPECTED_MODEL_SHAPES.items():
        if tensors.get(key,{}).get('shape')!=shape:raise ValueError(f'Wrong model interface at {key}')
    blocks={int(k.split('.')[2]) for k in tensors if k.startswith('action_header.transformer_blocks.')}
    if blocks!=set(range(6)):raise ValueError('Checkpoint does not contain exactly six action blocks')
    if not any(k.startswith('vlm_model.') for k in tensors):raise ValueError('Full VLM weights are missing')
    return {'header_bytes':length,'model_bytes':expected,'tensor_count':len(tensors),
            'vlm_tensors':sum(k.startswith('vlm_model.') for k in tensors),
            'action_tensors':sum(k.startswith('action_header.') for k in tensors),'memory_tensors':len(extra),
            'embedding_shape':tensors.get('vlm_model.model.language_model.embed_tokens.weight',{}).get('shape')}

def normalization_from_config(cfg):
    m=cfg['model'];field=cfg['data']['transform']['field']
    if (m['action_dim'],m['odim'],m['action_chunk_size'],m['num_blocks'])!=(78,43,30,6):raise ValueError('Expected 78/43/30/6 model')
    if m.get('noise_scheduler')!='flow' or m.get('rtc') is not True or m.get('max_delay')!=8:raise ValueError('Expected trained RTC flow model with max_delay8')
    if m.get('combined_temb') or m.get('pooled_text_encoder'):raise ValueError('This bundle supports the audited no-pooled-encoder architecture only')
    if m.get('resnet_store_path') is not None or m.get('use_dit') is not False or m.get('vlm_layer_indices') is not None:raise ValueError('External encoders or a different action architecture are not supported')
    if field.get('action_norm_type')!='bounds' or field.get('normalize_state') is not True or field.get('use_norm_mask') is not False:raise ValueError('Unsupported normalization')
    out={'action_norm_type':'bounds','normalize_state':True,'use_norm_mask':False}
    for key,n in [('action_min',78),('action_max',78),('state_min',43),('state_max',43)]:
        value=field.get(key)
        if not isinstance(value,list) or len(value)!=n or not all(isinstance(x,(int,float)) and math.isfinite(x) for x in value):raise ValueError(f'Invalid {key}')
        out[key]=value
    for lo,hi in [('action_min','action_max'),('state_min','state_max')]:
        if any(x>y for x,y in zip(out[lo],out[hi])):raise ValueError('Inverted normalization bounds')
    model_transform=cfg['data']['transform']['model']
    if model_transform['resize']['size']!=[180,320] or model_transform['center_crop']['size']!=[180,320]:raise ValueError('Expected180x320 transform')
    return out

def verify_bundle(root,expected_checkpoint_sha256=None):
    root=Path(root).resolve()
    manifest_path=regular_file(root,'bundle.json');ready_path=regular_file(root,'BUNDLE_READY.json')
    manifest=json.loads(manifest_path.read_text());ready=json.loads(ready_path.read_text())
    if manifest.get('format')!=FORMAT or manifest.get('source_commit')!=SOURCE_COMMIT:raise ValueError('Unsupported bundle format/source')
    if ready.get('bundle_sha256')!=sha256(manifest_path):raise ValueError('Bundle manifest does not match completion receipt')
    contract=manifest['contract']
    if manifest['contract_sha256']!=json_sha(contract):raise ValueError('Contract hash mismatch')
    for k,v in {'state_dim':43,'action_dim':78,'horizon':30,'action_tick_hz':30,'max_delay':8,'state_order':'model_urdf','action_order':'token64+left_model7+right_model7','actions_normalized':False}.items():
        if contract.get(k)!=v:raise ValueError(f'Unsupported contract field {k}')
    ckpt=manifest['checkpoint']
    if not isinstance(ckpt['step'],int) or ckpt['step']<=0:raise ValueError('Explicit completed checkpoint step required')
    if expected_checkpoint_sha256 and ckpt['sha256']!=expected_checkpoint_sha256:raise ValueError('Unexpected checkpoint identity')
    files=manifest.get('files',{})
    required={f"run/checkpoints/ckpt_{ckpt['step']}/model.safetensors",'run/run_config.json','run/argv.txt','normalization.json','assets/qwen/config.json','assets/qwen/tokenizer.json','assets/qwen/tokenizer_config.json','assets/qwen/preprocessor_config.json','source/src/psi/models/psi0.py','source/src/psi/deploy/serve_psi0_sonic.py'}
    if not required.issubset(files):raise ValueError('Required bundle files are not in the manifest')
    actual_files=set()
    for candidate in root.rglob('*'):
        if candidate.is_symlink():raise ValueError('Symlinks are not allowed inside a bundle')
        if candidate.is_file():actual_files.add(candidate.relative_to(root).as_posix())
    if actual_files!=set(files)|{'bundle.json','BUNDLE_READY.json'}:raise ValueError('Unlisted or missing bundle files')
    for relative,entry in files.items():
        p=regular_file(root,relative)
        if p.stat().st_size!=entry['bytes'] or sha256(p)!=entry['sha256']:raise ValueError(f'Integrity check failed: {relative}')
    model_rel=f"run/checkpoints/ckpt_{ckpt['step']}/model.safetensors"
    if files[model_rel]['sha256']!=ckpt['sha256']:raise ValueError('Model identity mismatch')
    info=read_header(root/model_rel)
    cfg=json.loads((root/'run/run_config.json').read_text());norm=normalization_from_config(cfg)
    if canonical_bytes(norm)!=(root/'normalization.json').read_bytes():raise ValueError('Runtime normalization differs from exported stats')
    if sha256(root/'normalization.json')!=contract['stats_sha256']:raise ValueError('Stats identity mismatch')
    base=json.loads((root/'assets/qwen/config.json').read_text())
    if base.get('model_type')!='qwen3_vl' or base.get('text_config',{}).get('hidden_size')!=2048:raise ValueError('Wrong local Qwen architecture')
    if info['embedding_shape']!=[base['text_config']['vocab_size'],2048]:raise ValueError('Local Qwen vocabulary does not match weights')
    if not set(MEMORY_PIPELINE)<=set(files):raise ValueError('SAM3-v2 memory spec/pipeline files are missing')
    spec_path=regular_file(root,'memory/spec.json')
    if contract.get('policy_memory')!='sam3v2' or contract.get('memory_spec_sha256')!=sha256(spec_path):raise ValueError('SAM3-v2 memory spec identity mismatch')
    if info['memory_tensors']!=ckpt.get('memory_tensors'):raise ValueError('Memory tensor count differs from the manifest')
    spec=json.loads(spec_path.read_text());m=cfg['model']
    if m.get('objmem_enabled') is not True or not m.get('object_slots')==spec['objects']['object_slots']==len(spec['objects']['prompts']):raise ValueError('SAM3-v2 model/memory spec mismatch')
    if cfg['train']['name']!='sonic_objmem':raise ValueError('Not a sonic_objmem run')
    return manifest
