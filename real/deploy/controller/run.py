#!/usr/bin/env python3
"""Default: check only. --execute launches the verified guarded native binary."""
from pathlib import Path
import argparse,fcntl,hashlib,json,os,platform,socket,subprocess,sys,time,uuid
ROOT=Path(__file__).resolve().parent
PROTOCOL='cart78.token_lease.v1'

def require(condition,message):
 if not condition:raise ValueError(message)

def sha(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
 return h.hexdigest()

def conflicts():
 hits=[]
 if not Path('/proc').is_dir():return hits
 for p in Path('/proc').iterdir():
  if not p.name.isdigit() or int(p.name)==os.getpid():continue
  try:
   comm=(p/'comm').read_text().strip()
   try:exe=os.readlink(p/'exe')
   except OSError:exe=''
   if 'g1_deploy' in comm or 'g1_deploy' in Path(exe.removesuffix(' (deleted)')).name:
    hits.append({'pid':int(p.name),'comm':comm})
  except OSError:pass
 return hits

def output_port_free(port):
 try:
  with socket.socket(socket.AF_INET,socket.SOCK_STREAM) as s:s.bind(('0.0.0.0',port))
  return True
 except OSError:return False

def resolve_nic(value):
 if value and value!='auto':
  if not (Path('/sys/class/net')/value).exists():raise ValueError(f'Network interface missing: {value}')
  return value
 proc=subprocess.run(['ip','-j','-4','addr','show'],text=True,capture_output=True,check=True)
 candidates={x['ifname'] for x in json.loads(proc.stdout) if any(a.get('local','').startswith('192.168.123.') for a in x.get('addr_info',[]))}
 if len(candidates)!=1:raise ValueError('auto requires exactly one NIC on 192.168.123.0/24; specify network_interface explicitly')
 return candidates.pop()

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--config',required=True);m=ap.add_mutually_exclusive_group();m.add_argument('--execute',action='store_true');m.add_argument('--check',action='store_true');args=ap.parse_args()
 cfg0=json.loads(Path(args.config).expanduser().read_text());cfg=cfg0.get('controller',cfg0)
 os.umask(0o077)
 source=Path(cfg['source_root']).expanduser().resolve();receipt=json.loads((source/'cart78_build_receipt.json').read_text())
 package=json.loads((ROOT/'patch_manifest.json').read_text());require(receipt['patch_sha256']==package['patch_sha256'],'Build is from another patch version')
 binary=Path(cfg.get('binary') or receipt['binary']).expanduser().resolve()
 require(sha(binary)==receipt['binary_sha256'],'Native binary SHA differs from build receipt')
 for e in package['files']:require(sha(source/e['path'])==e['patched_sha256'],'Patched source mismatch: '+e['path'])
 env=os.environ.copy();env['LD_LIBRARY_PATH']=':'.join(receipt['runtime_library_dirs']+[env.get('LD_LIBRARY_PATH','')])
 marker=json.loads(subprocess.check_output([str(binary),'--psi0-lease-info'],text=True,env=env,timeout=20))
 require(marker['protocol']==PROTOCOL and marker['build_marker']=='CART78_TOKEN_LEASE_GUARD_V1' and marker['hardware_initialized'] is False,'Wrong native guard marker')
 assets=json.loads((ROOT/'asset_contract.json').read_text());paths={};actual_hashes={}
 for key in ['encoder','decoder','observation_config']:
  paths[key]=Path(cfg[key]).expanduser().resolve();actual_hashes[key]=sha(paths[key]);require(actual_hashes[key]==assets[key],f'{key} is not the dataset-matched SONIC v1.0 release asset')
 reference=Path(cfg['reference_dir']).expanduser().resolve();name=cfg.get('default_motion')
 candidates=sorted(p.name for p in reference.iterdir() if p.is_dir())
 if not name or name not in candidates:raise ValueError('Choose the verified paused reference default_motion from: '+', '.join(candidates))
 ref_contract=json.loads((ROOT/'reference_contract.json').read_text());require(name==ref_contract['name'],'Only the observed existing paused-reference contract is packaged')
 for filename,want in ref_contract['files'].items():require(sha(reference/name/filename)==want,'Reference differs from captured WS files: '+filename)
 timeout=int(cfg.get('lease_timeout_ms',250));require(100<=timeout<=1000,'lease_timeout_ms must be 100..1000')
 iface=resolve_nic(cfg.get('network_interface','auto'));out_port=int(cfg.get('out_port',5557));require(1<=out_port<=65535,'invalid output port')
 require(1<=int(cfg.get('zmq_port',5556))<=65535,'invalid input port')
 require(cfg.get('zmq_topic','pose')=='pose','Only the dedicated pose topic is supported')
 require(0.2<=float(cfg.get('max_close_ratio',1.0))<=1.0,'max_close_ratio must be 0.2..1.0')
 # SONIC v1.0 was trained and recorded at nominal PD gains; the x1.5 ankle scaling is v1.1-only.
 require('kp_scale' not in cfg and 'kd_scale' not in cfg,'SONIC v1.0 runs at nominal gains; remove kp_scale/kd_scale from the controller config')
 if cfg.get('planner'):require(Path(cfg['planner']).expanduser().is_file(),'Planner asset missing')
 pids=conflicts();port_free=output_port_free(out_port)
 lock_dir=Path.home()/'.local/state/leases';lock_dir.mkdir(mode=0o700,parents=True,exist_ok=True)
 lock_path=lock_dir/'p.lock';lock_fd=os.open(lock_path,os.O_RDWR|os.O_CREAT,0o600);lock_acquired=False
 try:fcntl.flock(lock_fd,fcntl.LOCK_EX|fcntl.LOCK_NB);lock_acquired=True
 except BlockingIOError:pass
 ready=not pids and port_free and lock_acquired and platform.system()=='Linux'
 plan={'check_only':not args.execute,'protocol':PROTOCOL,'patched_binary_verified':True,'sonic_model':'v1.0 (policy/release)','motor_gains':'nominal','assets_verified':actual_hashes,'reference':name,'reference_behavior':'frame0 paused; no automatic trajectory playback','network_interface':iface,'existing_controllers':pids,'output_port_free':port_free,'launcher_lock_available':lock_acquired,'execute_allowed':ready,'no_motor_commands_sent_by_check':True}
 print(json.dumps(plan,indent=2),flush=True)
 if not args.execute:
  os.close(lock_fd);return 0 if ready else 2
 if not ready:os.close(lock_fd);raise SystemExit('Execute refused: existing controller, occupied output port, active launcher lock, or unsupported platform. Exit old controller through its normal safe procedure; nothing is killed automatically.')
 # Recheck immediately before exec. Never stop or signal another process.
 if conflicts() or not output_port_free(out_port):os.close(lock_fd);raise SystemExit('Controller conflict appeared during preflight')
 run_root=Path.home()/'.local/state'/('r-'+uuid.uuid4().hex[:8]);run_root.mkdir(mode=0o700)
 alias=run_root/'r';alias.symlink_to(binary);private_args=run_root/'c.json';logdir=run_root/'l';logdir.mkdir(mode=0o700)
 native=[iface,str(paths['decoder']),str(reference),'--encoder-file',str(paths['encoder']),'--obs-config',str(paths['observation_config']),'--input-type','zmq','--output-type','zmq','--zmq-host',str(cfg.get('zmq_host','127.0.0.1')),'--zmq-port',str(int(cfg.get('zmq_port',5556))),'--zmq-topic',str(cfg.get('zmq_topic','pose')),'--zmq-out-port',str(out_port),'--zmq-out-topic',str(cfg.get('out_topic','g1_debug')),'--default-motion',name,'--max-close-ratio',str(float(cfg.get('max_close_ratio',1.0))),'--logs-dir',str(logdir)]
 if cfg.get('planner'):native+=['--planner-file',str(Path(cfg['planner']).expanduser().resolve()),'--planner-precision','32']
 native+=['--policy-precision','32']
 private_args.write_text(json.dumps(native)+'\n');os.chmod(private_args,0o600)
 env.update(CART78_LEASE_TIMEOUT_MS=str(timeout),CART78_ENCODER_SHA256=actual_hashes['encoder'],CART78_DECODER_SHA256=actual_hashes['decoder'],CART78_OBSERVATION_SHA256=actual_hashes['observation_config'])
 launch={'pid':os.getpid(),'binary_sha256':receipt['binary_sha256'],'guard_protocol':PROTOCOL,'started_unix':time.time(),'private_args':str(private_args),'assets':actual_hashes,'reference':name}
 (run_root/'receipt.json').write_text(json.dumps(launch,indent=2)+'\n');os.ftruncate(lock_fd,0);os.write(lock_fd,json.dumps(launch).encode());os.fsync(lock_fd);os.set_inheritable(lock_fd,True)
 os.chdir(source/'gear_sonic_deploy')
 print('Native guarded controller starts only now. Physical control remains under its existing keyboard gates; P/N/T/R reference playback is disabled.',flush=True)
 os.execve(str(alias),[str(alias),'--args-file',str(private_args)],env)
if __name__=='__main__':
 try:raise SystemExit(main())
 except (AssertionError,ValueError,KeyError,FileNotFoundError,subprocess.CalledProcessError) as e:raise SystemExit('Controller preflight failed: '+str(e))
