#!/usr/bin/env python3
"""Compile only in the private prepared tree; never starts a controller or GPU test."""
from pathlib import Path
import argparse,datetime,hashlib,json,os,subprocess
ROOT=Path(__file__).resolve().parent

def require(condition,message):
 if not condition:raise ValueError(message)

def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
 return h.hexdigest()

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--source',required=True);ap.add_argument('--tensorrt-root',required=True);ap.add_argument('--cuda-root',default='/usr/local/cuda-12.8');ap.add_argument('--googletest-source');ap.add_argument('--jobs',type=int,default=2);ap.add_argument('--check',action='store_true');a=ap.parse_args()
 if not 1<=a.jobs<=2:raise SystemExit('Build jobs must be 1 or 2 to avoid interfering with existing control workloads')
 src=Path(a.source).expanduser().resolve();prep=json.loads((src/'cart78_prepare_receipt.json').read_text());manifest=json.loads((ROOT/'patch_manifest.json').read_text())
 require(prep['source']!=str(src),'Must not compile in the original live checkout')
 for e in manifest['files']:require(sha(src/e['path'])==e['patched_sha256'],e['path'])
 trt=Path(a.tensorrt_root).expanduser().resolve();cuda=Path(a.cuda_root).expanduser().resolve();require((trt/'include').is_dir() and (trt/'lib').is_dir(),'TensorRT include/lib missing')
 deploy=src/'gear_sonic_deploy';build=deploy/'build-cart78';binary=deploy/'target/release/g1_deploy_onnx_ref'
 env=dict(os.environ,HAS_ROS2='0',TensorRT_ROOT=str(trt),CUDAToolkit_ROOT=str(cuda),CUDA_HOME=str(cuda))
 libpaths=[str(trt/'lib'),str(cuda/'lib64'),str(deploy/'thirdparty/onnxruntime-linux-x64-1.22.0/lib')]
 env['LD_LIBRARY_PATH']=':'.join(libpaths+[env.get('LD_LIBRARY_PATH','')])
 configure=['cmake','-S',str(deploy),'-B',str(build),'-DCMAKE_BUILD_TYPE=Release','-DCMAKE_CXX_FLAGS_RELEASE=-O3 -DNDEBUG -fno-finite-math-only','-DCMAKE_EXPORT_COMPILE_COMMANDS=ON',f'-DTensorRT_ROOT={trt}']
 if a.googletest_source:configure.append('-DFETCHCONTENT_SOURCE_DIR_GOOGLETEST='+str(Path(a.googletest_source).resolve()))
 compile_cmd=['cmake','--build',str(build),'--target','g1_deploy_onnx_ref','-j',str(a.jobs)]
 if a.check:print(json.dumps({'configure':configure,'build':compile_cmd,'hardware_started':False},indent=2));return
 os.umask(0o077)
 try:os.nice(10)
 except OSError:pass
 log=src/'cart78_build.log'
 with log.open('w') as output:
  subprocess.run(configure,env=env,stdout=output,stderr=subprocess.STDOUT,check=True)
  subprocess.run(compile_cmd,env=env,stdout=output,stderr=subprocess.STDOUT,check=True)
 require(binary.is_file(),'Native binary missing')
 marker=json.loads(subprocess.check_output([str(binary),'--psi0-lease-info'],env=env,text=True,timeout=20))
 require(marker['protocol']=='cart78.token_lease.v1' and marker['hardware_initialized'] is False,'Invalid native marker')
 receipt={'source_root':str(src),'binary':str(binary),'binary_sha256':sha(binary),'guard_info':marker,'source_files':manifest['files'],'patch_sha256':manifest['patch_sha256'],'runtime_library_dirs':libpaths,'compiled_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'log':str(log),'build_jobs':a.jobs,'hardware_started':False,'cuda_benchmark_executed':False}
 (src/'cart78_build_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt,indent=2))
if __name__=='__main__':main()
