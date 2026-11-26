"""增量汇总本机价值诊断数据；只用标准库，SQLite 仅作可重建索引。"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
import json
import math
import os
from pathlib import Path
import random
import sqlite3
import time


def metrics(m):
    n=m[0]
    if not n:return {'n':0,'ev':None,'mse':None,'bias':None,'target_var':None,'invalid':m[-1]}
    my,mv=m[1]/n,m[2]/n;vy=max(0.,m[3]/n-my*my);vv=max(0.,m[4]/n-mv*mv);bias=m[8]/n;ve=max(0.,m[7]/n-bias*bias)
    return {'n':n,'ev':1-ve/vy if n>1 and vy>1e-10 else None,'mse':m[7]/n,'rmse':math.sqrt(max(0,m[7]/n)),
            'mae':m[6]/n,'bias':bias,'target_var':vy,'residual_var':ve,'mean_target':my,'mean_value':mv,
            'calibration_slope':(m[5]/n-my*mv)/vv if vv>1e-10 else None,'invalid':m[-1]}


def merge(items):
    out=[0.]*10
    for a in items:
        for i,x in enumerate(a):out[i]+=x
    return out


def merge_groups(rows):
    groups=defaultdict(list)
    for row in rows:
        for k,v in row.items():groups[k].append(v)
    return {k:metrics(merge(v)) for k,v in groups.items()}


def cluster_interval(clusters, repeats=200):
    """按根局面/整局重采样；点估计仍为按决策数加权的 pooled 指标。"""
    if len(clusters)<20:return {'clusters':len(clusters),'ev':None,'mse':None,'reason':'至少 20 个整局后才显示粗略区间；正式判断仍需更大样本'}
    rng=random.Random(731);ev=[];mse=[]
    for _ in range(repeats):
        s=metrics(merge(rng.choices(clusters,k=len(clusters))))
        if s['ev'] is not None:ev.append(s['ev'])
        mse.append(s['mse'])
    def interval(x):
        x=sorted(x);return [x[int((len(x)-1)*.025)],x[int((len(x)-1)*.975)]] if x else None
    return {'clusters':len(clusters),'repeats':repeats,'ev':interval(ev),'mse':interval(mse)}


def ingest(db,root):
    db.execute('CREATE TABLE IF NOT EXISTS offsets (path TEXT PRIMARY KEY, pos INTEGER, size INTEGER)')
    db.execute('CREATE TABLE IF NOT EXISTS events (id TEXT PRIMARY KEY, kind TEXT, t REAL, body TEXT)')
    db.execute('CREATE INDEX IF NOT EXISTS events_kind_time ON events(kind,t)')
    errors=0
    for p in sorted(root.glob('*/metrics/*.jsonl')):
        key=str(p.relative_to(root));row=db.execute('SELECT pos,size FROM offsets WHERE path=?',(key,)).fetchone();size=p.stat().st_size
        pos=row[0] if row and size>=row[0] else 0
        if pos==size:continue
        with p.open('rb') as f:
            f.seek(pos)
            while True:
                start=f.tell();line=f.readline()
                if not line or not line.endswith(b'\n'):break
                pos=f.tell()
                try:
                    rec=json.loads(line)
                    if rec.get('schema')!=1:raise ValueError('unknown schema')
                    kind=rec['type'];tm=float(rec['time']);body=json.dumps(rec,ensure_ascii=False)
                    db.execute('INSERT OR IGNORE INTO events VALUES (?,?,?,?)',(key+':'+str(start),kind,tm,body))
                except (ValueError,KeyError,TypeError):errors+=1
        db.execute('INSERT OR REPLACE INTO offsets VALUES (?,?,?)',(key,pos,size))
    db.commit();return errors


def capture_existing(db,log_path,root,host):
    """留存现有健康统计作上线前基线；不伪造成新探针的矩或独立评估。"""
    if not log_path.exists():return
    db.execute('CREATE TABLE IF NOT EXISTS legacy_offsets (path TEXT PRIMARY KEY, pos INTEGER)')
    row=db.execute('SELECT pos FROM legacy_offsets WHERE path=?',(str(log_path),)).fetchone();size=log_path.stat().st_size
    pos=row[0] if row and row[0]<=size else max(0,size-(4<<20))
    dest=root/host/'metrics';dest.mkdir(parents=True,exist_ok=True)
    with log_path.open('rb') as f:
        f.seek(pos)
        if not row and pos:f.readline();pos=f.tell()
        while True:
            line=f.readline()
            if not line or not line.endswith(b'\n'):break
            pos=f.tell()
            try:x=json.loads(line)
            except ValueError:continue
            if not x.get('health'):continue
            out={k:x.get(k) for k in ('time','version','total_steps','health','macro_frac','loss_v_macro','run_episodes','sps','approx_kl','stale_steps')}
            out.update(schema=1,type='legacy',provenance='existing_training_log_not_new_value_probe')
            p=dest/('legacy_'+time.strftime('%Y%m%d',time.gmtime(x['time']))+'.jsonl')
            with p.open('a',encoding='utf-8') as h:h.write(json.dumps(out,ensure_ascii=False,separators=(',',':'))+'\n')
    db.execute('INSERT OR REPLACE INTO legacy_offsets VALUES (?,?)',(str(log_path),pos));db.commit()


def build_summary(db,root,cfg_path=None):
    counts=dict(db.execute('SELECT kind,count(*) FROM events GROUP BY kind'))
    def recent(kind,n):return [json.loads(r[0]) for r in db.execute('SELECT body FROM events WHERE kind=? ORDER BY t DESC LIMIT ?',(kind,n))][::-1]
    batches=recent('batch',200);episodes=recent('episode',2000);errors=recent('error',5);legacy=recent('legacy',200)
    latest=batches[-1] if batches else None
    reward_id=latest.get('reward_id') if latest else (episodes[-1].get('reward_id') if episodes else None)
    behavior_id=latest.get('behavior_id') if latest else (episodes[-1].get('behavior_id') if episodes else None)
    batches=[x for x in batches if x.get('reward_id')==reward_id and x.get('behavior_id')==behavior_id]
    episodes=[x for x in episodes if x.get('reward_id')==reward_id and x.get('behavior_id')==behavior_id]
    # 构筑价值势能（决策记录 253）上线时奖励指纹没跟着变：按主线配置的 bv_start 把势能前后分开，爬坡期另记
    era=None
    if cfg_path and cfg_path.is_file():
        try:
            c=json.loads(cfg_path.read_text(encoding='utf-8'))
            if (c.get('bv_shaping') or 0)>0 and c.get('bv_start') is not None:
                era={'bv_start':int(c['bv_start']),'bv_ramp_end':int(c['bv_start'])+int(c.get('bv_ramp') or 0)}
        except (ValueError,OSError):pass
    if era and latest and latest['version']>=era['bv_start']:
        batches=[x for x in batches if x['version']>=era['bv_start']]
        episodes=[x for x in episodes if x.get('version_start',0)>=era['bv_start']]
        era['ramp_episodes']=sum(1 for x in episodes if x.get('version_start',0)<era['bv_ramp_end'])
    good=[x for x in episodes if x.get('mc_valid')]
    # 同一 root 的多个观察合并，bootstrap 不把它们当独立游戏。
    roots=defaultdict(list)
    for e in good:
        if e['groups']['mc_behavior']['macro'][0]:roots[e['root_id']].append(e['groups']['mc_behavior']['macro'])
    clusters=[merge(v) for v in roots.values()]
    statuses=[]
    for p in root.glob('*/*_status.json'):
        try:statuses.append(dict(json.loads(p.read_text()),host=p.parent.name))
        except (ValueError,OSError):pass
    archive_files=[p for p in root.glob('*/*/*.npz')]
    episode_stats={target:merge_groups([e['groups'][target] for e in good if target in e['groups']])
                   for target in ('segment','full_lambda','full_lambda_no_cut','mc_behavior')}
    splits={split:merge_groups([e['groups']['mc_behavior'] for e in good if e['split']==split])
            for split in ('train','validation','test')}
    paired={key:merge_groups([b['paired'][key] for b in batches]) for key in ('old','current','public')}
    gradients=[b for b in batches if b.get('gradients')]
    hist=[{'time':b['time'],'version':b['version'],**{key:metrics(b['old_groups'][key]) for key in ('macro','combat')},
           'seconds':b['seconds']} for b in batches]
    result={'schema':1,'generated_at':time.time(),'latest_time':max([x['time'] for x in batches+episodes],default=None),
            'latest_version':latest['version'] if latest else None,'reward_id':reward_id,'behavior_id':behavior_id,'era':era,
            'all_counts':counts,'window_batches':len(batches),'window_episodes':len(episodes),'valid_mc_episodes':len(good),
            'unique_mc_roots':len(roots),'split_counts':dict(Counter(e['split'] for e in episodes)),
            'archive_files':len(archive_files),'archive_bytes':sum(p.stat().st_size for p in archive_files),
            'old_groups':merge_groups([b['old_groups'] for b in batches]),'paired':paired,
            'episode_targets':episode_stats,'episode_splits':splits,'mc_cluster_ci':cluster_interval(clusters),
            'gradients':gradients[-1]['gradients'] if gradients else None,
            'gradient_version':gradients[-1]['version'] if gradients else None,'history':hist,'statuses':statuses,'errors':errors,
            'probe_seconds_mean':sum(b['seconds'] for b in batches)/len(batches) if batches else None,
            'advantage_sign':{'macro_n':sum(b['paired']['current']['macro'][0] for b in batches),
                               'old_current_changes':sum(b['advantage_sign_changes'] for b in batches)},
            'privileged_policy_max_delta':max([b['privileged_policy_max_delta'] for b in batches],default=None),
            'calibration':{'current':merge_calibration([b['current_calibration'] for b in batches]),
                           'mc_behavior':merge_calibration([e['calibration_mc'] for e in good])},
            'target_difference':{key:metrics(merge([e[key] for e in good])) for key in ('segment_vs_mc','full_vs_mc')},
            'horizon':{'macro_steps':sum(e['macro_steps'] for e in episodes),'segments':sum(e['segments'] for e in episodes),
                       'mean_to_boundary':sum(e['horizon'].get('mean',0)*e['macro_steps'] for e in episodes)/max(1,sum(e['macro_steps'] for e in episodes))},
            'external':None,'baseline':None,
            'notes':['批次指标按矩合并，不平均 EV。','MC 是已记录的行为策略序列回报，不是固定策略独立续局。',
                     'train/validation/test 是未来离线诊断划分，不改变 PPO 入训样本。','已卡住的局不提供真实终局 MC 标签。']}
    result['legacy_latest']=legacy[-1] if legacy else None
    result['legacy_history']=legacy
    result['activation_pending']=(root/'activation_pending.json').exists()
    for name,key in [('continuation_summary.json','external'),('baseline_summary.json','baseline')]:
        p=root/name
        if p.exists():
            try:result[key]=json.loads(p.read_text())
            except ValueError:pass
    p=root/'sync.json'
    if p.exists():
        try:result['sync']=json.loads(p.read_text())
        except ValueError:pass
    return result


def merge_calibration(rows):
    rows=[r for r in rows if r]
    if not rows:return None
    edges=rows[0]['edges'];sums=[[0.,0.,0.,0.] for _ in rows[0]['bins']]
    for r in rows:
        if r['edges']!=edges:continue
        for a,b in zip(sums,r['bins']):
            for i in range(4):a[i]+=b[i]
    return {'edges':edges,'bins':[{'n':a[0],'predicted':a[1]/a[0] if a[0] else None,'target':a[2]/a[0] if a[0] else None,
                                 'mse':a[3]/a[0] if a[0] else None} for a in sums]}


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);ap.add_argument('--index',type=Path,required=True)
    ap.add_argument('--legacy-log',type=Path);ap.add_argument('--host',default='b')
    ap.add_argument('--config',type=Path,help='主线的训练配置（读 bv_start 分开势能前后）；不给时用 --legacy-log 同目录的 config.json')
    a=ap.parse_args();a.root.mkdir(parents=True,exist_ok=True);a.index.parent.mkdir(parents=True,exist_ok=True)
    db=sqlite3.connect(a.index,timeout=30)
    if a.legacy_log:capture_existing(db,a.legacy_log,a.root,a.host)
    bad=ingest(db,a.root);s=build_summary(db,a.root,a.config or (a.legacy_log.parent/'config.json' if a.legacy_log else None));s['invalid_records_this_scan']=bad;db.close()
    p=a.root/'summary.json';tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(s,ensure_ascii=False,allow_nan=False,indent=1));os.replace(tmp,p)
    print(json.dumps({k:s[k] for k in ('latest_version','all_counts','window_batches','valid_mc_episodes','archive_files')},ensure_ascii=False))


if __name__=='__main__':main()
