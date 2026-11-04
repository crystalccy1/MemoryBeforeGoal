#!/usr/bin/env python3
"""Exercise only main's early --psi0-lease-info branch; never constructs hardware."""
import argparse,hashlib,json,os,subprocess,tempfile
from pathlib import Path
ap=argparse.ArgumentParser();ap.add_argument('--source',required=True);a=ap.parse_args()
root=Path(a.source);r=json.loads((root/'cart78_build_receipt.json').read_text());binary=r['binary']
env=os.environ.copy();env['LD_LIBRARY_PATH']=':'.join(r['runtime_library_dirs']+[env.get('LD_LIBRARY_PATH','')])
results=[]
with tempfile.TemporaryDirectory(prefix='r-') as folder:
 os.chmod(folder,0o700);p=Path(folder)/'c.json'
 for name,value,mode,want in [('private-marker',['--psi0-lease-info'],0o600,0),('reject-public-marker',['--psi0-lease-info'],0o644,2),('reject-non-string',[42],0o600,2),('reject-recursive',['--args-file'],0o600,2)]:
  p.write_text(json.dumps(value));os.chmod(p,mode)
  result=subprocess.run([binary,'--args-file',str(p)],env=env,text=True,capture_output=True,timeout=20)
  if result.returncode!=want:raise RuntimeError((name,result.returncode,result.stdout,result.stderr))
  if want==0:
   data=json.loads(result.stdout)
   if data['hardware_initialized'] is not False or data['protocol']!='cart78.token_lease.v1':raise RuntimeError(data)
  results.append({'case':name,'returncode':result.returncode,'passed':True})
print(json.dumps({'binary_sha256':r['binary_sha256'],'tests':results,'hardware_controller_started':False,'cuda_benchmark':False},indent=2))
