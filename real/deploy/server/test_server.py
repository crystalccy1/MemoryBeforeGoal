#!/usr/bin/env python3
"""CPU unit tests and actual uvicorn/WebSocket protocol tests. No robot/GPU access."""
import asyncio,base64,io,json,math,os,socket,struct,subprocess,sys,tempfile,time,uuid
from pathlib import Path
import numpy as np
from PIL import Image
import websockets
import urllib.request
from backend import restore_prefix,state_vector,BackendError
from bundle import read_header,safe_relative
ROOT=Path(__file__).resolve().parent

def units():
    rng=np.random.default_rng(1);raw=rng.normal(size=(30,78)).astype('f4');prev=rng.normal(size=(30,78)).astype('f4')
    for d in [1,3,7]:
        out,err=restore_prefix(raw,prev,d)
        assert np.array_equal(out[:d],prev[:d]);assert np.array_equal(out[d:],raw[d:]);assert err>0
    try:restore_prefix(raw,prev,8)
    except BackendError:pass
    else:raise AssertionError('Delay8 was clamped instead of rejected')
    assert state_vector(np.zeros((1,43))).shape==(43,)
    for bad in [np.zeros(44),np.zeros((1,1,43)),[float('nan')]*43]:
        try:state_vector(bad)
        except BackendError:pass
        else:raise AssertionError('Invalid state accepted')
    for path in ['../outside','/absolute','a/../../outside']:
        try:safe_relative(path)
        except ValueError:pass
        else:raise AssertionError('Unsafe path accepted')
    with tempfile.TemporaryDirectory() as td:
        base=Path(td)
        schema=json.loads((ROOT/'a7_cart78_tensor_schema.json').read_text())['tensors'];header={};offset=0
        # the 90 memory tensors of a real WholeBodyMem checkpoint (read_header rejects a header without them)
        schema.update(json.loads((ROOT.parents[1]/'memory/TrashBag/tensor_inventory.json').read_text()))
        sizes={'F32':4,'BF16':2,'I64':8}
        for k,v in schema.items():
            n=math.prod(v['shape'])*sizes[v['dtype']];header[k]={**v,'data_offsets':[offset,offset+n]};offset+=n
        data=json.dumps(header).encode();p=base/'truncated.safetensors';p.write_bytes(struct.pack('<Q',len(data))+data)
        try:read_header(p)
        except ValueError as e:assert 'Incomplete' in str(e)
        else:raise AssertionError('Truncated full-schema weight file accepted')

class Process:
    def __init__(self,delay=5):
        s=socket.socket();s.bind(('127.0.0.1',0));self.port=s.getsockname()[1];s.close()
        self.tmp=tempfile.TemporaryDirectory();self.log=Path(self.tmp.name)/'server.log';self.handle=self.log.open('w')
        self.proc=None;self.closed=False
        try:
            self.proc=subprocess.Popen([sys.executable,str(ROOT/'serve.py'),'--mock','--mock-delay-ms',str(delay),'--port',str(self.port)],stdout=self.handle,stderr=subprocess.STDOUT)
            # Dependency imports can exceed ten seconds on a busy CPU host.
            # This bounds fixture startup only; action latency limits stay intact.
            deadline=time.monotonic()+45
            while time.monotonic()<deadline:
                try:self.info=self.get_info();break
                except Exception:
                    if self.proc.poll() is not None:raise RuntimeError('Mock server exited during startup')
                    time.sleep(.05)
            else:raise RuntimeError('Mock server startup exceeded 45 seconds')
        except BaseException as exc:
            diagnostic=self.close()
            raise RuntimeError(f'Mock server fixture failed: {exc}\n{diagnostic}') from exc
    def get_info(self):
        return json.load(urllib.request.urlopen(f'http://127.0.0.1:{self.port}/deployment-info',timeout=1))
    def close(self):
        if self.closed:return ''
        self.closed=True
        if self.proc is not None:
            if self.proc.poll() is None:self.proc.terminate()
            try:self.proc.wait(timeout=5)
            except subprocess.TimeoutExpired:self.proc.kill();self.proc.wait()
        self.handle.close();text=self.log.read_text();self.tmp.cleanup();return text

def jpeg():
    b=io.BytesIO();Image.new('RGB',(640,360),(20,70,120)).save(b,format='JPEG');return base64.b64encode(b.getvalue()).decode()
IMAGE=jpeg()
def hello(info,sid,mode='dry_run'):
    return {'type':'hello','protocol':'cart78.v1','session_id':sid,'mode':mode,'expected':{k:info[k] for k in ['checkpoint_sha256','stats_sha256','contract_sha256']}}

async def observations(ws,sid,ledger,state):
    while True:
        seq=state['seq'];state['seq']+=1;sent=time.monotonic_ns();ledger[seq]=sent
        await ws.send(json.dumps({'type':'observation','protocol':'cart78.v1','session_id':sid,'observation_seq':seq,'client_sent_monotonic_ns':sent,'state_model43':[0.0]*43,'image_jpeg_b64':IMAGE,'instruction':'Move the dining cart.'}))
        await asyncio.sleep(1/30)

async def fast_tests(p):
    info=p.info;uri=f'ws://127.0.0.1:{p.port}/ws';results={}
    assert info['mock'] and not info['ready'] and info['dry_run']
    sid=uuid.uuid4().hex
    async with websockets.connect(uri) as ws:
        await ws.send(json.dumps(hello(info,sid)));ack=json.loads(await ws.recv());assert ack['type']=='hello_ack'
        ledger={};state={'seq':0};producer=asyncio.create_task(observations(ws,sid,ledger,state));actions=[]
        try:
            deadline=time.monotonic()+5
            while len(actions)<45 and time.monotonic()<deadline:
                msg=json.loads(await asyncio.wait_for(ws.recv(),2))
                if msg['type']=='error':raise AssertionError(msg)
                if msg['type']=='action':
                    assert msg['session_id']==sid and msg['dry_run'] and len(msg['action'])==78
                    seq=msg['source_observation_seq'];assert ledger[seq]==msg['source_client_sent_monotonic_ns']
                    rtt=(time.monotonic_ns()-ledger[seq])/1e6-msg['server_elapsed_since_source_observation_received_ms'];assert -5<=rtt<100
                    assert abs(msg['action'][0]-(seq%30)/1000)<1e-7,'Source provenance does not match actual mock inference input'
                    if actions:
                        last=actions[-1];assert msg['action_seq']==last['action_seq']+1
                        assert seq>=last['source_observation_seq']
                        assert msg['chunk_id']>=last['chunk_id']
                        if msg['chunk_id']==last['chunk_id']:assert msg['chunk_index']>last['chunk_index']
                    actions.append(msg)
            assert len(actions)==45 and len({x['chunk_id'] for x in actions})>=2
            assert actions[0]['source_observation_seq']>0,'Warmup observation was executed'
            results['normal_stream_and_provenance']=True
        finally:producer.cancel();await asyncio.gather(producer,return_exceptions=True)
        # Stop observations: cached actions must terminate at the input deadline.
        began=time.monotonic();seen_error=False
        while time.monotonic()-began<1:
            try:msg=json.loads(await asyncio.wait_for(ws.recv(),1))
            except websockets.ConnectionClosed:break
            if msg['type']=='error':
                assert msg['code']=='observation_stale';seen_error=True
            elif seen_error:assert msg['type']!='action','Action emitted after the fault'
        assert seen_error;results['stale_observation_fail_closed']=True
    await asyncio.sleep(.05)
    # A new session starts with sequence1 and fresh observations, never old actions.
    sid2=uuid.uuid4().hex
    async with websockets.connect(uri) as ws:
        await ws.send(json.dumps(hello(info,sid2)));assert json.loads(await ws.recv())['type']=='hello_ack'
        ledger={};state={'seq':0};producer=asyncio.create_task(observations(ws,sid2,ledger,state))
        try:
            while True:
                msg=json.loads(await asyncio.wait_for(ws.recv(),2))
                if msg['type']=='error':raise AssertionError(msg)
                if msg['type']=='action':assert msg['session_id']==sid2 and msg['action_seq']==1 and msg['source_observation_seq']>0;break
        finally:producer.cancel();await asyncio.gather(producer,return_exceptions=True)
    results['reconnect_discards_previous_session']=True
    await asyncio.sleep(.1)
    async with websockets.connect(uri) as ws:
        await ws.send(json.dumps(hello(info,uuid.uuid4().hex,'execute')));msg=json.loads(await ws.recv());assert msg['type']=='error' and msg['code']=='execution_not_ready'
    results['mock_execution_rejected']=True
    async with websockets.connect(uri) as ws:
        h=hello(info,uuid.uuid4().hex);h['expected']['checkpoint_sha256']='9'*64
        await ws.send(json.dumps(h));msg=json.loads(await ws.recv());assert msg['type']=='error' and msg['code']=='identity_mismatch'
    results['wrong_checkpoint_rejected']=True
    return results

async def slow_tests(p):
    sid=uuid.uuid4().hex;ledger={};state={'seq':0}
    async with websockets.connect(f'ws://127.0.0.1:{p.port}/ws') as ws:
        await ws.send(json.dumps(hello(p.info,sid)));assert json.loads(await ws.recv())['type']=='hello_ack'
        producer=asyncio.create_task(observations(ws,sid,ledger,state));seen=False
        try:
            deadline=time.monotonic()+4
            while time.monotonic()<deadline:
                msg=json.loads(await asyncio.wait_for(ws.recv(),3))
                assert msg['type']!='action','Late model emitted an action'
                if msg['type']=='profile' and msg['reason']=='latency_budget_exceeded_no_actions':seen=True;break
            assert seen
        finally:producer.cancel();await asyncio.gather(producer,return_exceptions=True)
    return {'slow_inference_reported_without_action_or_clamp':True}

if __name__=='__main__':
    units();out={'unit_checks':True,'mock_only':True,'gpu_used':False}
    p=Process()
    try:out.update(asyncio.run(fast_tests(p)))
    except BaseException:
        print(p.log.read_text());raise
    finally:p.close()
    p=Process(delay=300)
    try:out.update(asyncio.run(slow_tests(p)))
    except BaseException:
        print(p.log.read_text());raise
    finally:p.close()
    (ROOT/'test_results.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
