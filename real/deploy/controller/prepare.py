#!/usr/bin/env python3
"""Copy a verified controller source into a NEW private checkout and apply the guard."""
from pathlib import Path
import argparse,datetime,hashlib,json,os,shutil,subprocess
ROOT=Path(__file__).resolve().parent

def require(condition,message):
 if not condition:raise ValueError(message)

def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
 return h.hexdigest()

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--source',required=True);ap.add_argument('--destination',required=True);ap.add_argument('--check',action='store_true');a=ap.parse_args()
 src=Path(a.source).expanduser().resolve();dst=Path(a.destination).expanduser().resolve()
 manifest=json.loads((ROOT/'patch_manifest.json').read_text());patch=ROOT/'cart78_controller.patch'
 require(sha(patch)==manifest['patch_sha256'],'Package patch hash mismatch')
 if dst==src or src in dst.parents:raise SystemExit('Destination must be a new directory outside the live source checkout')
 if dst.exists():raise SystemExit('Refusing to overwrite an existing destination')
 for entry in manifest['files']:
  if entry['base_sha256']:
   p=src/entry['path'];require(p.is_file() and sha(p)==entry['base_sha256'],f'Wrong source snapshot: {p}')
 plan={'source':str(src),'destination':str(dst),'protocol':manifest['protocol'],'check_only':a.check}
 if a.check:print(json.dumps(plan,indent=2));return
 os.umask(0o077);dst.mkdir(parents=True,mode=0o700)
 deploy=dst/'gear_sonic_deploy';deploy.mkdir(mode=0o700)
 # Do not reuse/mutate a live build tree, caches, logs, or checkpoint outputs.
 cmd=['rsync','-a','--links','--exclude=.git','--exclude=build','--exclude=target','--exclude=logs','--exclude=logs_*','--exclude=__pycache__',str(src/'gear_sonic_deploy')+'/',str(deploy)+'/']
 subprocess.run(cmd,check=True)
 subprocess.run(['git','apply','--check',str(patch)],cwd=dst,check=True)
 subprocess.run(['git','apply',str(patch)],cwd=dst,check=True)
 for entry in manifest['files']:require(sha(dst/entry['path'])==entry['patched_sha256'],entry['path'])
 plan.update({'prepared_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'source_files':manifest['files'],'patch_sha256':manifest['patch_sha256'],'hardware_started':False})
 (dst/'cart78_prepare_receipt.json').write_text(json.dumps(plan,indent=2)+'\n')
 os.chmod(dst,0o700);print(json.dumps(plan,indent=2))
if __name__=='__main__':main()
