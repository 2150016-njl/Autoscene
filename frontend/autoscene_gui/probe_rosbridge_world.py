#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, time
import websocket
TOPICS={
"/autoscene/experiment/status":"autoscene_msgs/ExperimentStatus",
"/autoscene/scenario/status":"autoscene_msgs/ScenarioStatus",
"/autoscene/simulation/status":"autoscene_msgs/SimulationStatus",
"/autoscene/world/state":"autoscene_msgs/WorldState",
"/autoscene/ego/state":"autoscene_msgs/VehicleState",
"/autoscene/agents/state":"autoscene_msgs/VehicleStateArray",
}
def summarize(topic,msg):
    if topic.endswith('/world/state'):
        ego=msg.get('ego') or {}; p=(ego.get('pose') or {}).get('position') or {}
        return {'sim_time':msg.get('sim_time'),'ego':[p.get('x'),p.get('y'),ego.get('heading')],'agents':len(msg.get('agents') or [])}
    if topic.endswith('/ego/state'):
        p=(msg.get('pose') or {}).get('position') or {}; return [p.get('x'),p.get('y'),msg.get('heading'),msg.get('valid')]
    if topic.endswith('/agents/state'):
        return {'vehicles':len(msg.get('vehicles') or msg.get('agents') or [])}
    if topic.endswith('/experiment/status'):
        return {'state':msg.get('state'),'mode':msg.get('mode'),'message':msg.get('message')}
    if topic.endswith('/scenario/status'):
        return {'loaded':msg.get('loaded'),'running':msg.get('running'),'step':msg.get('step'),'message':msg.get('message')}
    if topic.endswith('/simulation/status'):
        return {'sim_time':msg.get('sim_time'),'real_time_factor':msg.get('real_time_factor')}
    return msg

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--bridge',default='ws://127.0.0.1:9090'); ap.add_argument('--seconds',type=float,default=15.0); args=ap.parse_args()
    print('connecting:',args.bridge)
    ws=websocket.create_connection(args.bridge,timeout=3); ws.settimeout(1.0)
    for i,(topic,mtype) in enumerate(TOPICS.items()):
        ws.send(json.dumps({'op':'subscribe','id':'probe_%d'%i,'topic':topic,'type':mtype,'throttle_rate':0,'queue_length':1}))
        print('SUB',topic,mtype)
    counts={t:0 for t in TOPICS}; deadline=time.time()+args.seconds
    while time.time()<deadline:
        try: raw=ws.recv()
        except Exception as exc:
            if 'timed out' in str(exc).lower(): continue
            print('RECV ERROR:',repr(exc)); break
        if not raw: continue
        try: obj=json.loads(raw)
        except Exception: print('NON-JSON:',raw[:300]); continue
        op=obj.get('op')
        if op=='status': print('ROSBRIDGE STATUS:',json.dumps(obj,ensure_ascii=False)); continue
        if op!='publish': print('OTHER:',json.dumps(obj,ensure_ascii=False)[:800]); continue
        topic=obj.get('topic','')
        if topic not in counts: continue
        counts[topic]+=1
        if counts[topic]<=3 or counts[topic]%20==0:
            print('RX',topic,'#',counts[topic],json.dumps(summarize(topic,obj.get('msg') or {}),ensure_ascii=False))
    print('\nCOUNTS')
    for topic,count in counts.items(): print('%5d  %s'%(count,topic))
    ws.close()
if __name__=='__main__': main()
