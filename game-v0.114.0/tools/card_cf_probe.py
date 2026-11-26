"""单张牌的反事实对照（2026-11-01，决策记录 238，研究报告/2026-11-01_碎片整理出牌调查）。只评测，不碰训练。

题目：考卷 boss3_exam.json 里的 175 套真实第三幕牌组，各加一张指定的牌（--card），遭遇换成普通战、精英或原 Boss（--group），
开局种子不同（--reps 重复几遍）。模型贪心打这场战斗；走到这张牌在手里、能打、而模型选了别的动作的第一个决策点（--max-tests 个），
同一批随机世界（reseed，--nseed 个）里比较两条臂，之后都由模型贪心打完：
  A 现在打这张牌；
  B 模型原来的选择，并且禁止打这张牌（--ban combat：整场战斗；--ban turn：只到本回合结束回合为止）。
记胜率与剩余血量（占最大生命的比例）。A − B 为正说明这张牌被少打了。

    .venv/bin/python -I tools/card_cf_probe.py run --ckpt 检查点.pt --card CARD.DEFRAGMENT --group boss --reps 3 --shard 0/3 --out 结果_boss_0.jsonl
    .venv/bin/python -I tools/card_cf_probe.py report "结果_boss_*.jsonl"

检查点可以是评测留存的裸权重（best/vNNN.pt，配置取 B 的 config.json 的 model 项）或带配置的打包文件。
"""
import argparse, json, math, sys, time, copy, random
from pathlib import Path
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO/"python")); sys.path.insert(0, str(REPO/"tools"))
import numpy as np, torch
import mcts_eval as M
from mcts_eval import Task, outcome, enemy_hp, MAX_STEPS
from sts2env import VecEnv
from sts2env.env import EngineError
from sts2rl.actor import MODE_COMBAT
from sts2rl.features import Ragged, collate_ragged
from sts2rl.model import ModelConfig, PolicyValueNet, load_state

DEFRAG = None      # 要测的牌的词表 ID（main 里按 --card 查）
class BTask(Task):
    """B 臂：这回合剩下的时间不打碎片整理（选到结束回合后解禁）。"""
    ban = True

NORMAL = ["SCROLLS_OF_BITING_NORMAL","FROG_KNIGHT_NORMAL","AXEBOTS_NORMAL","GLOBE_HEAD_NORMAL","FABRICATOR_NORMAL"]
ELITE = ["KNIGHTS_ELITE","MECHA_KNIGHT_ELITE","SOUL_NEXUS_ELITE"]

def make_specs(exam, group, reps, seed0, card):
    items=[]
    base=exam["items"]
    exam={"items":[x for _ in range(reps) for x in base]}
    rnd=random.Random(seed0)
    for n,it in enumerate(exam["items"]):
        sp=copy.deepcopy(it["spec"]); snap=json.loads(sp["player_snapshot"])
        if not any(c["id"]==card for c in snap["deck"]):
            snap["deck"].append({"floor_added_to_deck":1,"id":card})
        sp["player_snapshot"]=json.dumps(snap)
        sp["seed"]="DF%05d"%(n*7+seed0)
        if group=="normal": sp["encounter"]=rnd.choice(NORMAL); sp["room"]="Monster"; sp.pop("next_encounter",None)
        elif group=="elite": sp["encounter"]=rnd.choice(ELITE); sp["room"]="Elite"; sp.pop("next_encounter",None)
        else: sp.pop("next_encounter",None); sp["room"]="Boss"
        items.append({"id":n,"group":group,"spec":sp,"encounter":sp["encounter"]})
    return items

def defrag_cand(o):
    for k in range(o.num_actions):
        t,a=int(o.cand_ints[k,0]),int(o.cand_ints[k,1])
        if t==2 and a>=0 and int(o.tok_ints[a,1])==DEFRAG: return k
    return None
def end_cand(o):
    for k in range(o.num_actions):
        if int(o.cand_ints[k,0])==1: return k
    return None

def probe_item(item, nseed, max_tests, base):
    spec, prefix = item["spec"], []
    tests=[]
    while True:
        (o,) = yield ("tasks",[Task(spec,prefix,"obs")])
        if isinstance(o,EngineError) or o.done or o.stuck or o.num_actions==0 or len(prefix)>MAX_STEPS:
            res=outcome(o) if not isinstance(o,EngineError) and len(prefix)<=MAX_STEPS else {"win":False,"hp":0,"max_hp":1}
            return {"final":res,"tests":tests,"decisions":len(prefix)}
        p,v = yield ("policy",o)
        act=int(np.argmax(p))
        dk=defrag_cand(o)
        if dk is not None and act!=dk and len(tests)<max_tests and o.num_actions>1:
            seeds=[base+7919*(len(prefix)+1)+j for j in range(nseed)]
            tasks=[(Task if c==dk else BTask)(spec,prefix,"rollout",seed=s,first=c,owner=(c,s)) for c in (dk,act) for s in seeds]
            if int(o.cand_ints[act,0])==1 and ARGS.ban!="combat":
                for t in tasks:
                    if isinstance(t,BTask): t.ban=False
            results=yield ("tasks",tasks)
            A=[];B=[]
            for t,r in zip(tasks,results):
                r=r if isinstance(r,dict) else {"win":False,"hp":0,"max_hp":1}
                (A if t.owner[0]==dk else B).append((int(r["win"]), r["hp"]/max(1,r["max_hp"]), r.get("enemy_left")))
            tests.append({"step":len(prefix),"act_end":int(o.cand_ints[act,0])==1,"pcard":float(p[dk]),"pact":float(p[act]),"v":float(v),
                          "hp":int(o.hp),"max_hp":int(o.max_hp),"A":A,"B":B})
        prefix.append(act)

def run(items, net, envs, live, out):
    ctrls={}; waiting={}; queue=[]; pol_req=[]; env_task={}; obs_of={}; free=None
    nxt=iter(items); n_live=0; cid_seq=[0]
    f=open(out,"a",encoding="utf-8")
    def advance(cid,value):
        nonlocal n_live
        gen=ctrls[cid][0]
        try: kind,payload=gen.send(value)
        except StopIteration as e:
            it=ctrls.pop(cid)[1]; f.write(json.dumps({"id":it["id"],"group":it["group"],"enc":it["encounter"],**e.value},ensure_ascii=False)+"\n"); f.flush()
            n_live-=1; start_more(); return
        if kind=="policy": pol_req.append((cid,payload))
        else:
            for t in payload: t.owner=(cid,t.owner)
            waiting[cid]=(payload,0); queue.extend(payload)
    def start_more():
        nonlocal n_live
        while n_live<live:
            try: it=next(nxt)
            except StopIteration: return
            n_live+=1; cid=cid_seq[0]; cid_seq[0]+=1
            ctrls[cid]=[probe_item(it,ARGS.nseed,ARGS.max_tests,100003*(it["id"]+1)),it]; advance(cid,None)
    def task_done(t,value):
        cid,owner=t.owner; t.owner=owner; t.result=value
        tasks,k=waiting[cid]; k+=1
        if k==len(tasks): del waiting[cid]; advance(cid,[x.result for x in tasks])
        else: waiting[cid]=(tasks,k)
    t0=time.time()
    with VecEnv(envs) as env:
        free=list(range(len(env)))
        def feed(i,o):
            t=env_task[i]
            if isinstance(o,EngineError): del env_task[i]; free.append(i); task_done(t,o); return
            if t.mode=="obs": del env_task[i]; free.append(i); task_done(t,o); return
            if o.done or o.stuck or o.num_actions==0 or t.steps>MAX_STEPS:
                del env_task[i]; free.append(i); task_done(t,outcome(o,t) if t.steps<=MAX_STEPS else {"win":False,"hp":0,"max_hp":1}); return
            obs_of[i]=o
        start_more()
        while ctrls:
            while free and queue:
                i=free.pop(); t=queue.pop(); env_task[i]=t
                req={"op":"reset","mode":"combat","spec":t.spec,"replay":t.prefix}
                if t.seed is not None:
                    req["reseed"]=int(t.seed)&0xFFFFFFFF
                    if t.first is not None: req["first"]=int(t.first)
                env.send(i,req)
            for i,o in env.recv_ready(timeout=0.5): feed(i,o)
            deadline=time.time()+0.002
            while env.pending and time.time()<deadline:
                for i,o in env.recv_ready(timeout=max(0.0,deadline-time.time())): feed(i,o)
            if not obs_of and not pol_req: continue
            ids=list(obs_of); reqs=list(pol_req); pol_req.clear()
            batch=[obs_of[i] for i in ids]+[o for _,o in reqs]
            with torch.no_grad():
                logits,values=net(collate_ragged(Ragged.from_obs(batch,np.array([MODE_COMBAT]*len(batch)))).to("cpu"))
            logits,values=logits.float().cpu(),values.float().cpu()
            for k,i in enumerate(ids):
                o=obs_of.pop(i); t=env_task[i]; t.steps+=1
                lg=logits[k,:o.num_actions].clone()
                if getattr(t,"ban",False) and t.seed is not None:
                    d=defrag_cand(o)
                    if d is not None and o.num_actions>1: lg[d]=-1e9
                a_=int(lg.argmax())
                if (ARGS.ban!="combat") and getattr(t,"ban",False) and int(o.cand_ints[a_,0])==1: t.ban=False
                env.send(i,{"op":"step","action":a_})
            for k,(cid,o) in enumerate(reqs):
                p=torch.softmax(logits[len(ids)+k,:o.num_actions],-1).numpy()
                advance(cid,(p,float(values[len(ids)+k])))
    print("done",time.time()-t0,flush=True)

def report(files):
    import glob, statistics as st
    rows=[json.loads(l) for pat in files for f in glob.glob(pat) for l in open(f) if l.strip()]
    T=[t for r in rows for t in r["tests"]]
    print(f"{len(rows)} 场战斗，{len(T)} 个测试点（A 现在打这张牌，B 模型原选择并禁止打它；A−B 为正说明被少打了）")
    if not T:
        return
    def ms(x):
        m=sum(x)/len(x); sd=st.pstdev(x) if len(x)>1 else 0.0
        return m, 1.96*sd/math.sqrt(len(x))
    for label,sel in (("全部",lambda t:True),("模型原选结束回合",lambda t:t["act_end"]),("模型原选别的牌",lambda t:not t["act_end"])):
        TT=[t for t in T if sel(t)]
        if len(TT)<5:
            continue
        dw=[sum(x[0] for x in t["A"])/len(t["A"])-sum(x[0] for x in t["B"])/len(t["B"]) for t in TT]
        dh=[sum(x[1] for x in t["A"])/len(t["A"])-sum(x[1] for x in t["B"])/len(t["B"]) for t in TT]
        (m,e),(m2,e2)=ms(dw),ms(dh)
        print(f"  {label}: n={len(TT)}  胜率差 {100*m:+.1f} ± {100*e:.1f} 点  剩余血量比差 {100*m2:+.2f} ± {100*e2:.2f} 点  模型给这张牌的概率中位 {st.median(t['pcard'] for t in TT):.3f}")


if __name__=="__main__":
    if len(sys.argv)>1 and sys.argv[1]=="report":
        report(sys.argv[2:]); sys.exit(0)
    if len(sys.argv)>1 and sys.argv[1]=="run":
        del sys.argv[1]
    ap=argparse.ArgumentParser()
    ap.add_argument("--ckpt",required=True); ap.add_argument("--group",required=True); ap.add_argument("--out",required=True)
    ap.add_argument("--shard",default="0/1"); ap.add_argument("--envs",type=int,default=6); ap.add_argument("--live",type=int,default=4)
    ap.add_argument("--nseed",type=int,default=12); ap.add_argument("--max-tests",type=int,default=2); ap.add_argument("--limit",type=int,default=0); ap.add_argument("--reps",type=int,default=1); ap.add_argument("--ban",choices=["combat","turn"],default="combat"); ap.add_argument("--card",default="CARD.DEFRAGMENT")
    ap.add_argument("--exam",default=str(Path.home()/".local/share/sls2/boss3_exam.json"))
    ARGS=ap.parse_args(); torch.set_num_threads(2)
    vocab=[l.strip() for l in open(REPO/"engine/catalog/vocab.txt",encoding="utf-8")]; DEFRAG=vocab.index(ARGS.card)
    blob=torch.load(ARGS.ckpt,map_location="cpu",weights_only=False)
    if "config" not in blob:
        mc=json.load(open(Path.home()/".local/share/sls2/runs/remote/defect-b/config.json"))["model"]; blob={"version":Path(ARGS.ckpt).stem,"config":{"model":mc},"state":blob}
    net=PolicyValueNet(ModelConfig(**blob["config"]["model"])).eval(); load_state(net,blob["state"])
    print("version",blob.get("version"),flush=True)
    exam=json.load(open(ARGS.exam)); items=make_specs(exam,ARGS.group,ARGS.reps,1,ARGS.card)
    k,n=map(int,ARGS.shard.split("/")); items=[x for j,x in enumerate(items) if j%n==k]
    if ARGS.limit: items=items[:ARGS.limit]
    run(items,net,ARGS.envs,ARGS.live,ARGS.out)
