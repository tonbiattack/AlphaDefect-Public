"""价值诊断接口：导出续局根状态、导入固定策略续局/反事实标签、离线资源基线。
所有命令仅产生诊断文件，不发布权重，不写训练配置。
"""
from __future__ import annotations
import argparse
from collections import defaultdict
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

REPO=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(REPO/'python'),str(REPO/'tools')]
import numpy as np
from sts2rl.value_probe import load_npz, moments, stats, digest
from value_probe_summary import cluster_interval, merge

LABEL_SCHEMA={'schema':1,'kind':'value or action','root_id':'exported root_id','world_id':'independent future identifier',
              'policy_id':'sha256 of frozen checkpoint plus action/temperature rule','reward_id':'exported reward_id',
              'info_id':'hash of fixed conditioning information','information':'public or privileged',
              'prediction':0.0,'return':0.0,'win':False,'action_id':'required for action rows',
              'role':'estimate or validate (action rows)','provenance':'engine/build and sampler description'}
LABEL_SCHEMA['optional']={'win_probability':'[0,1], distinct probability head; never use raw reward value',
                         'score_projection':'fixed projections of frozen-policy grad log pi for sampled root action',
                         'advantage_old':'old-critic advantage','advantage_new':'candidate-critic advantage'}


def atomic_json(path,obj):
    path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix('.tmp')
    tmp.write_text(json.dumps(obj,ensure_ascii=False,allow_nan=False,indent=2));os.replace(tmp,path)


def validate_labels(rows):
    seen=set();context=set();predictions={}
    for r in rows:
        if r.get('schema')!=1 or r.get('kind') not in ('value','action'):raise ValueError('schema/kind 无效')
        for k in ('root_id','world_id','policy_id','reward_id','info_id','information','provenance'):
            if not isinstance(r.get(k),str) or not r[k]:raise ValueError(f'缺少 {k}')
        if r['information'] not in ('public','privileged'):raise ValueError('information 必须明确为 public/privileged')
        for k in ('prediction','return'):
            if not isinstance(r.get(k),(int,float)) or not math.isfinite(r[k]):raise ValueError(f'{k} 必须有限')
        if 'win' in r and type(r['win']) is not bool:raise ValueError('win 必须为布尔值')
        if 'win_probability' in r and (not isinstance(r['win_probability'],(int,float)) or not 0<=r['win_probability']<=1 or 'win' not in r):
            raise ValueError('win_probability 必须在 [0,1] 且有 win 标签')
        if 'score_projection' in r:
            z=r['score_projection']
            if not isinstance(z,list) or not z or len(z)>1024 or not all(isinstance(x,(int,float)) and math.isfinite(x) for x in z):
                raise ValueError('score_projection 需要 1–1024 个有限值')
            for k in ('advantage_old','advantage_new'):
                if not isinstance(r.get(k),(int,float)) or not math.isfinite(r[k]):raise ValueError('梯度样本缺少有限优势')
        if r['kind']=='action' and (not isinstance(r.get('action_id'),str) or r.get('role') not in ('estimate','validate')):
            raise ValueError('动作标签需要 action_id 和独立的 estimate/validate 标记')
        key=(r['kind'],r['root_id'],r['world_id'],r.get('action_id'))
        if key in seen:raise ValueError('同一个根/世界/动作重复，不能当成独立续局')
        seen.add(key);context.add((r['policy_id'],r['reward_id'],r['information']))
        pk=(r['kind'],r['root_id'],r.get('action_id'))
        old=predictions.setdefault(pk,(r['prediction'],r['info_id']))
        if old!=(r['prediction'],r['info_id']):raise ValueError('同一根的固定信息或预测随世界变化')
    if len(context)!=1:raise ValueError('一次导入必须使用同一冻结策略、奖励和信息口径')
    return next(iter(context))


def continuation_summary(rows):
    policy,reward,information=validate_labels(rows);byroot=defaultdict(list)
    for r in rows:
        if r['kind']=='value':byroot[r['root_id']].append(r)
    repeated=[rr for rr in byroot.values() if len(rr)>=2]
    out={'schema':1,'time':time.time(),'policy_id':policy,'reward_id':reward,'information':information,
         'roots':len(byroot),'repeated_roots':len(repeated),'rollouts':sum(map(len,byroot.values())),
         'noise_ceiling':None,'single_outcome':None,'mean_outcome':None,'ranking':None,
         'note':'噪声上限是有限样本估计；按根聚类。少于 200 个独立根不作正式结论。'}
    if repeated:
        ys=[np.array([r['return'] for r in rr],float) for rr in repeated]
        mu=np.array([x.mean() for x in ys]);vv=np.array([x.var(ddof=1) for x in ys]);nn=np.array(list(map(len,ys)))
        within=float(vv.mean());between=float(mu.var(ddof=1)-(vv/nn).mean()) if len(mu)>1 else None
        if between is not None:
            signal=max(0.,between);out['noise_ceiling']={'between_raw':between,'between':signal,'within':within,
                'ev_max_estimate':signal/(signal+within) if signal+within>1e-10 else None,'roots':len(mu),'K_min':int(nn.min()),'K_max':int(nn.max())}
        pv=np.array([rr[0]['prediction'] for rr in repeated])
        # 每个根等权，再在该根的世界内等权；K 不同也不偏向多算了几次的根。
        clusters=[(np.asarray(moments(np.full(len(x),p),x))/len(x)).tolist() for p,x in zip(pv,ys)]
        out['single_outcome']=dict(stats(merge(clusters)),cluster_ci=cluster_interval(clusters),
                                   weighting='uniform_root_then_uniform_world',rollouts=int(nn.sum()))
        out['mean_outcome']=dict(stats(moments(pv,mu)),estimated_value_mse=float(((pv-mu)**2-vv/nn).mean()))
    actions=defaultdict(lambda:defaultdict(lambda:defaultdict(dict)))
    for r in rows:
        if r['kind']=='action':actions[r['root_id']][r['role']][r['action_id']][r['world_id']]=r
    ranking=[]
    for root,roles in actions.items():
        est,val=roles['estimate'],roles['validate']
        if len(est)<2:continue
        # 估计用与验证用世界完全分离，防止从同一噪声里挑赢家又验收。
        if {w for a in est.values() for w in a}&{w for a in val.values() for w in a}:raise ValueError('反事实 estimate/validate 世界重叠')
        pred={k:next(iter(a.values()))['prediction'] for k,a in est.items()}
        predicted=max(pred,key=pred.get);estimated=max(est,key=lambda k:np.mean([x['return'] for x in est[k].values()]))
        row={'root_id':root,'predicted_action':predicted,'estimated_action':estimated,'agree':predicted==estimated,
             'validation_worlds':0,'validated_delta':None,'ci95':None}
        if predicted in val and estimated in val:
            common=sorted(set(val[predicted])&set(val[estimated]))
            diff=np.array([val[predicted][w]['return']-val[estimated][w]['return'] for w in common])
            row['validation_worlds']=len(diff)
            if len(diff)>=2:
                mean=float(diff.mean());half=1.96*float(diff.std(ddof=1))/math.sqrt(len(diff))
                row.update(validated_delta=mean,ci95=[mean-half,mean+half])
        ranking.append(row)
    if ranking:
        out['ranking']={'roots':len(ranking),'agreement':sum(x['agree'] for x in ranking)/len(ranking),
                        'validated_roots':sum(x['validated_delta'] is not None for x in ranking),'details':ranking}
    wp=[r for r in rows if 'win_probability' in r]
    if wp:
        p=np.array([r['win_probability'] for r in wp]);y=np.array([r['win'] for r in wp],float);pc=np.clip(p,1e-7,1-1e-7)
        out['win_probability']={'n':len(wp),'brier':float(((p-y)**2).mean()),'log_loss':float(-(y*np.log(pc)+(1-y)*np.log1p(-pc)).mean())}
    gp=[r for r in rows if 'score_projection' in r]
    if len(gp)>1:
        if len({len(r['score_projection']) for r in gp})!=1:raise ValueError('投影维数不一致')
        z=np.array([r['score_projection'] for r in gp]);out['policy_gradient_projection']={'n':len(gp),'dimensions':z.shape[1]}
        for key in ('old','new'):
            g=z*np.array([r['advantage_'+key] for r in gp])[:,None]
            out['policy_gradient_projection'][key+'_variance_trace']=float(g.var(axis=0,ddof=1).sum())
        out['policy_gradient_projection']['note']='只对所提供的固定投影与采样分布成立，不冒充完整参数空间的梯度方差。'
    return out


def import_rollouts(root,path):
    rows=[json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    result=continuation_summary(rows)
    name=hashlib.sha256(path.read_bytes()).hexdigest()+'.jsonl';dest=root/'labels'/name;dest.parent.mkdir(parents=True,exist_ok=True)
    if not dest.exists():dest.write_bytes(path.read_bytes())
    # 保留每一批的结果；看板展示最近一次同口径批次，不把异质策略混起来。
    atomic_json(root/'labels'/name.replace('.jsonl','.summary.json'),result)
    atomic_json(root/'continuation_summary.json',result);return result


def archives(root,limit):
    return sorted(root.glob('*/episodes/*.npz'),key=lambda p:p.stat().st_mtime)[-limit:]


def export_roots(root,out,limit,split):
    rows=[];seen=set()
    for p in archives(root,max(2000,limit*20)):
        meta,a,obs=load_npz(p)
        if meta['split']!=split or not meta['mc_valid'] or meta['root_id'] in seen or obs is None:continue
        seen.add(meta['root_id']);j=int(meta['root_id'][:8],16)%len(obs);step=int(a['macro_index'][j])
        req=dict(meta['request']);req['replay']=list(map(int,a['actions'][:step]));req['privileged']=False
        raw=obs.select(np.array([j]));raw.tok_priv=np.zeros_like(raw.tok_priv)
        info=hashlib.sha256(b''.join(getattr(raw,k).tobytes() for k in raw.__dataclass_fields__ if getattr(raw,k) is not None)).hexdigest()
        rows.append({'schema':1,'root_id':meta['root_id']+f':{step}','episode_id':meta['episode_id'],'split':split,
                     'source_npz':str(p.resolve()),'macro_row':j,'primitive_step':step,'reward_id':meta['reward_id'],
                     'info_id':info,'information':'public','reset_replay_request':req,'source_version':int(a['versions'][step]),
                     'source_old_value':float(a['values'][step]),'source_old_value_is_frozen_prediction':False,
                     'required':'固定检查点与策略规则；验证重放观测一致后，对未知未来按合法条件分布独立采样。'})
        if len(rows)>=limit:break
    out.parent.mkdir(parents=True,exist_ok=True);out.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows))
    return {'roots':len(rows),'out':str(out),'split':split}


def fit_baseline(root,limit=1000):
    games=[]
    for p in archives(root,limit):
        meta,a,_=load_npz(p)
        if not meta['mc_valid'] or not len(a['macro_index']):continue
        games.append((meta,a))
    latest=games[-1][0] if games else {}
    games=[g for g in games if g[0]['reward_id']==latest['reward_id'] and g[0]['behavior_id']==latest['behavior_id']]
    counts={s:sum(m['split']==s for m,a in games) for s in ('train','validation','test')}
    out={'schema':1,'time':time.time(),'counts':counts,'ready':False,'reward_id':latest.get('reward_id'),
         'behavior_id':latest.get('behavior_id'),'note':'线性进度/血量/金币/牌数/遗物基线；按整局划分，不进入 PPO。标签为行为序列 MC。'}
    if counts['train']>=100 and counts['validation']>=20 and counts['test']>=20:
        xx=np.zeros((7,7));xy=np.zeros(7)
        for m,a in games:
            if m['split']!='train':continue
            x=a['baseline_features'].astype(float);y=a['mc_behavior_target'];w=1/len(y)
            xx+=w*x.T@x;xy+=w*x.T@y
        reg=np.eye(7)*1e-3;reg[0,0]=0;coef=np.linalg.solve(xx+reg,xy);out['coefficients']=coef.tolist();out['ready']=True
        for s in ('validation','test'):
            base=[];old=[]
            for m,a in games:
                if m['split']!=s:continue
                y=a['mc_behavior_target'];base.append(moments(a['baseline_features']@coef,y));old.append(moments(a['values'][a['macro_index']],y))
            out[s]={'baseline':stats(merge(base)),'actor_value':stats(merge(old)),'baseline_ci':cluster_interval(base),'actor_value_ci':cluster_interval(old)}
    atomic_json(root/'baseline_summary.json',out);return out


def main():
    ap=argparse.ArgumentParser();sub=ap.add_subparsers(dest='cmd',required=True)
    sub.add_parser('schema')
    ex=sub.add_parser('export-roots');ex.add_argument('--root',type=Path,required=True);ex.add_argument('--out',type=Path,required=True)
    ex.add_argument('--limit',type=int,default=256);ex.add_argument('--split',choices=['train','validation','test'],default='test')
    im=sub.add_parser('import-rollouts');im.add_argument('--root',type=Path,required=True);im.add_argument('--input',type=Path,required=True)
    bl=sub.add_parser('fit-baseline');bl.add_argument('--root',type=Path,required=True);bl.add_argument('--limit',type=int,default=1000)
    a=ap.parse_args()
    if a.cmd=='schema':result=LABEL_SCHEMA
    elif a.cmd=='export-roots':result=export_roots(a.root,a.out,a.limit,a.split)
    elif a.cmd=='import-rollouts':result=import_rollouts(a.root,a.input)
    else:result=fit_baseline(a.root,a.limit)
    print(json.dumps(result,ensure_ascii=False,allow_nan=False,indent=2))


if __name__=='__main__':main()
