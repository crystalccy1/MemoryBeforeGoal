#!/usr/bin/env python3
"""Real CUDA bundle + actual localhost WebSocket latency test. No robot ports.
Run only in a separately allocated GPU job. Requires pyarrow+av for test fixtures.
"""
from __future__ import annotations
import argparse,asyncio,base64,datetime,io,json,os,socket,subprocess,sys,time,urllib.request,uuid
from pathlib import Path
import numpy as np
from PIL import Image
import websockets



def private_server_command(argv,runtime_base,python=None):
    """Same server args, private files and opaque process argv; preserve venv lookup."""
    runtime_base=Path(runtime_base).expanduser().resolve()
    runtime_base.mkdir(parents=True,exist_ok=True)
    run=runtime_base/('r-'+uuid.uuid4().hex[:12]);run.mkdir(mode=0o700)
    py=Path(python or sys.executable).absolute()
    if py.parent.name=='bin' and (py.parent.parent/'pyvenv.cfg').is_file():
        (run/'e').symlink_to(py.parent.parent,target_is_directory=True)
        executable=run/'e/bin'/py.name
    else:
        (run/'p').symlink_to(py);executable=run/'p'
    private_argv=list(argv)
    if '--bundle' in private_argv:
        i=private_argv.index('--bundle')+1;private_argv[i]=str(Path(private_argv[i]).expanduser().resolve())
    config={'target':str(Path(__file__).with_name('serve.py').resolve()),'argv':private_argv}
    script="import json,os,runpy,sys\nfrom pathlib import Path\nr=Path(__file__).resolve().parent\nos.umask(0o077)\nc=json.loads((r/'c.json').read_text())\nos.environ.setdefault('PYTHONDONTWRITEBYTECODE','1')\nos.environ['CART78_RUNTIME_DIR']=str(r)\nsys.path.insert(0,str(Path(c['target']).parent))\nsys.path.insert(0,str(Path(c['target']).parent.parent))\nsys.argv=[c['target'],*c['argv']]\nos.chdir(r)\nrunpy.run_path(c['target'],run_name='__main__')\n"
    for name,content in [('c.json',json.dumps(config)),('r.py',script)]:
        fd=os.open(run/name,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
        with os.fdopen(fd,'w') as f:f.write(content)
    return [str(executable),str(run/'r.py')],run


def fixtures(dataset,count=180):
    import pyarrow.parquet as pq
    import av
    info=json.loads((dataset/'meta/info.json').read_text())
    table=pq.read_table(dataset/'data/chunk-000/episode_000000.parquet',columns=['states'])
    states=np.asarray(table['states'].to_pylist(),dtype=np.float32)
    tasks=[json.loads(x) for x in (dataset/'meta/tasks.jsonl').read_text().splitlines() if x]
    task=tasks[0].get('task',tasks[0].get('description','Move the dining cart.'))
    video=dataset/'videos/chunk-000/observation.images.egocentric/episode_000000.mp4'
    frames=[]
    with av.open(str(video)) as c:
        for i,frame in enumerate(c.decode(video=0)):
            if i>=min(count,len(states)):break
            image=frame.to_image().convert('RGB')
            if image.size!=(640,360):raise RuntimeError('Real fixture is not640x360')
            b=io.BytesIO();image.save(b,format='JPEG',quality=85)
            frames.append({'state_model43':states[i].tolist(),'image_jpeg_b64':base64.b64encode(b.getvalue()).decode(),'instruction':str(task)})
    if len(frames)<30 or states.shape[1]!=43:raise RuntimeError('Insufficient real fixture data')
    return frames,{'dataset':str(dataset),'converted_episode':0,'real_frames_loaded':len(frames),'source_fps':info['fps'],'scope':'Sequential real validation frames, repeated for transport/latency testing; not closed-loop quality evidence'}


def get_info(port):return json.load(urllib.request.urlopen(f'http://127.0.0.1:{port}/deployment-info',timeout=2))

async def session(port,info,frames,seconds,mode='dry_run'):
    sid=uuid.uuid4().hex;ledger={};messages=[];actions=[];last_action=None;warm_done=False
    h={'type':'hello','protocol':'cart78.v1','session_id':sid,'mode':mode,'expected':{k:info[k] for k in ['checkpoint_sha256','stats_sha256','contract_sha256']}}
    async with websockets.connect(f'ws://127.0.0.1:{port}/ws',max_size=2_100_000,open_timeout=10) as ws:
        await ws.send(json.dumps(h));ack=json.loads(await asyncio.wait_for(ws.recv(),10))
        if ack.get('type')!='hello_ack':return {'mode':mode,'handshake':ack,'actions':0,'error':'handshake_rejected'}
        async def produce():
            seq=0;next_tick=time.monotonic()
            while True:
                sent=time.monotonic_ns();ledger[seq]=sent
                payload={**frames[seq%len(frames)],'type':'observation','protocol':'cart78.v1','session_id':sid,'observation_seq':seq,'client_sent_monotonic_ns':sent}
                await ws.send(json.dumps(payload));seq+=1;next_tick+=1/30
                await asyncio.sleep(max(0,next_tick-time.monotonic()))
        producer=asyncio.create_task(produce());started=time.monotonic();deadline=started+max(seconds,180 if mode=='dry_run' else seconds);active_deadline=None
        try:
            while time.monotonic()<deadline:
                if active_deadline is not None and time.monotonic()>=active_deadline:break
                message=json.loads(await asyncio.wait_for(ws.recv(),max(2,deadline-time.monotonic())))
                if message.get('session_id')!=sid or message.get('protocol')!='cart78.v1':raise AssertionError('Wrong response session')
                if message.get('type')=='profile':
                    messages.append(message)
                    if message.get('reason')=='warmup_complete_waiting_for_new_observation':warm_done=True;active_deadline=time.monotonic()+seconds
                elif message.get('type')=='action':
                    a=np.asarray(message['action'],dtype=np.float32)
                    assert a.shape==(78,) and np.isfinite(a).all()
                    assert message['checkpoint_sha256']==info['checkpoint_sha256'] and message['stats_sha256']==info['stats_sha256']
                    source=message['source_observation_seq'];assert ledger[source]==message['source_client_sent_monotonic_ns']
                    if last_action is not None:
                        assert message['action_seq']>last_action['action_seq'] and source>=last_action['source_observation_seq']
                        if message['chunk_id']==last_action['chunk_id']:assert message['chunk_index']>last_action['chunk_index']
                    if mode=='execute':assert message['dry_run'] is False and message['latency_ok'] is True
                    last_action=message
                    actions.append({k:message[k] for k in ['action_seq','source_observation_seq','chunk_id','chunk_index','latency_ms','latency_ok','dry_run']})
                    if active_deadline is None:active_deadline=time.monotonic()+seconds
                elif message.get('type')=='error':
                    messages.append(message);break
                else:raise AssertionError('Unexpected packet type')
        finally:
            producer.cancel();await asyncio.gather(producer,return_exceptions=True)
        latency=[float(x['latency_ms']) for x in messages if x.get('type')=='profile' and 'latency_ms' in x]
        return {'mode':mode,'handshake_ok':True,'warmup_completed':warm_done,'action_count':len(actions),'actions':actions,'profiles':messages,'completed_inference_ms':latency,'duration_seconds':time.monotonic()-started,'all_action_shapes_finite_78':True if actions else None,'identity_and_provenance_checks_passed':True if actions else None,'error_frames':[x for x in messages if x.get('type')=='error'],'robot_ports_used':False}


def summary(values):
    if not values:return {'n':0}
    a=np.asarray(values,dtype=float)
    return {'n':len(a),'min_ms':float(a.min()),'p50_ms':float(np.percentile(a,50)),'p95_ms':float(np.percentile(a,95)),'max_ms':float(a.max())}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bundle',type=Path,required=True);p.add_argument('--dataset',type=Path,required=True)
    p.add_argument('--output-dir',type=Path,required=True);p.add_argument('--inference-steps',type=int,nargs='+',default=[8,6,4])
    p.add_argument('--seconds',type=float,default=8);p.add_argument('--attention-backend',choices=['sdpa','flash_attention_2'],default='sdpa')
    a=p.parse_args();a.output_dir.mkdir(parents=True,exist_ok=True)
    report={'started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'attention_backend':a.attention_backend,'bundle':str(a.bundle),'slurm_job_id':os.getenv('SLURM_JOB_ID'),'cuda_visible_devices':os.getenv('CUDA_VISIBLE_DEVICES'),'cases':[],'robot_ports_used':False,'training_modified':False}
    def save():
        tmp=a.output_dir/'result.json.tmp';tmp.write_text(json.dumps(report,indent=2)+'\n');tmp.replace(a.output_dir/'result.json')
    frames,evidence=fixtures(a.dataset);report['fixture']=evidence;save()
    for n in a.inference_steps:
        sock=socket.socket();sock.bind(('127.0.0.1',0));port=sock.getsockname()[1];sock.close()
        log=a.output_dir/f'server_{n}.log';case={'inference_steps':n,'port':port,'status':'starting','server_log':str(log)};report['cases'].append(case);save()
        env=os.environ.copy();env['PYTHONUNBUFFERED']='1';env['OMP_NUM_THREADS']='8';env['TOKENIZERS_PARALLELISM']='false'
        command,private_run=private_server_command(['--bundle',str(a.bundle),'--host','127.0.0.1','--port',str(port),'--attention-backend',a.attention_backend,'--inference-steps',str(n),'--allow-execution'],a.output_dir.parent)
        case['private_runtime']=str(private_run)
        began=time.monotonic()
        with log.open('w') as out:
            proc=subprocess.Popen(command,stdout=out,stderr=subprocess.STDOUT,env=env)
            try:
                deadline=time.monotonic()+300
                while True:
                    if proc.poll() is not None:raise RuntimeError(f'Server exited{proc.returncode}; see{log}')
                    try:info=get_info(port);break
                    except Exception:
                        if time.monotonic()>deadline:raise TimeoutError('Real service failed to start in300s')
                        time.sleep(.25)
                assert info['mock'] is False and info['state_dim']==43 and info['action_dim']==78 and info['horizon']==30
                assert info['attention_backend']==a.attention_backend and info['inference_steps']==n
                case.update(status='profiling',startup_seconds=time.monotonic()-began,initial_info=info);save()
                case['dry_run']=asyncio.run(session(port,info,frames,a.seconds))
                for _ in range(100):
                    after=get_info(port)
                    if not after['active_session'] and not after['inference_busy']:break
                    time.sleep(.1)
                case['post_profile_info']=after
                case['steady_latency']=summary(case['dry_run']['completed_inference_ms'])
                if after['ready'] and after['latency_ok']:
                    # Onlylocalhost WS actions are consumed; no ZMQ/DDS/robot publisher exists.
                    case['execute_protocol_only']=asyncio.run(session(port,after,frames,2,'execute'))
                    case['status']='latency_gate_passed' if case['execute_protocol_only'].get('action_count',0)>0 and not case['execute_protocol_only'].get('error_frames') else 'execute_protocol_failed'
                else:case['status']='latency_blocked_or_fault'
                case['final_info']=get_info(port)
                if case['status']=='latency_gate_passed' and case['final_info'].get('fault') is not None:case['status']='execute_protocol_failed'
                save()
            except BaseException as exc:
                case.update(status='failed',error=type(exc).__name__+': '+str(exc));save()
            finally:
                proc.terminate()
                try:proc.wait(timeout=30)
                except subprocess.TimeoutExpired:proc.kill();proc.wait()
                case['server_exit_after_test']=proc.returncode;save()
    report['finished_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat()
    report['completed']=True
    report['scope']='Real exported policy, actual localhost uvicorn/WebSocket, real replay observations. No robot/control ports or training process changes. Latency gate pass is not real-robot task acceptance.'
    save();print(json.dumps({'output':str(a.output_dir/'result.json'),'cases':[{k:c.get(k) for k in ['inference_steps','status','steady_latency','error']} for c in report['cases']]},indent=2))
    if any(c['status']=='failed' for c in report['cases']):raise SystemExit(1)
if __name__=='__main__':main()
