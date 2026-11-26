"""只读价值诊断。独立抽样、后台落盘；输出从不参与动作、GAE、loss 或优化器更新。

schema=1: metrics 为可合并矩，episodes/batches 为 allow_pickle=False 的 NPZ。
完整局保存原始时间轴、局外原始观测及分段边界，支持离线重算不同目标。
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import os
from pathlib import Path
import queue
import threading
import time
import uuid

import numpy as np

from sts2env.protocol import TokenKind

from .features import Ragged, collate_ragged, macro_rows

SCHEMA = 1
MOMENT_FIELDS = ('n', 'sum_y', 'sum_v', 'sum_y2', 'sum_v2', 'sum_yv', 'sum_abs_error', 'sum_error2', 'sum_error', 'invalid')
REWARD_KEYS = ('gamma', 'lam', 'macro_lam', 'win_reward', 'lose_reward', 'hp_coef', 'potion_coef', 'floor_reward',
               'act1_boss_reward', 'act2_boss_reward', 'act3_boss_reward', 'act3_boss2_reward', 'victory_reward',
               'run_hp_shaping', 'hp_concavity', 'potion_discard_penalty', 'deck_over_coef', 'upgrade_coef',
               'rare_coef', 'relic_coef', 'combat_dmg_coef', 'run_potion_coef', 'relic_skip_penalty',
               'gold_skip_penalty', 'deck_shaping', 'wp_shaping', 'prog_floor', 'prog_boss', 'boss_dmg_coef', 'boss_win_bonus')
BEHAVIOR_KEYS = ('explore_probs', 'explore_temper', 'explore_until', 'explore_offpolicy', 'combat_explore',
                 'combat_explore_gate', 'combat_explore_until', 'combat_explore_cut', 'run_segment_steps')


def digest(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(',', ':')).encode()).hexdigest()


def context(cfg):
    rewards = {k: getattr(cfg, k, None) for k in REWARD_KEYS}
    # 构筑价值势能（决策记录 253）开着时，系数、爬坡与模型本身都进奖励指纹；关着时不加，旧数据的 reward_id 不变。
    # 爬坡期间系数随版本变，指纹里只有配置，分析时另按版本把 bv_start 到 bv_start + bv_ramp 单列
    if (getattr(cfg, 'bv_shaping', 0.0) or 0.0) > 0:
        bvm = Path(cfg.bv_model) if getattr(cfg, 'bv_model', '') else None
        rewards.update({k: getattr(cfg, k, None) for k in ('bv_shaping', 'bv_model', 'bv_start', 'bv_ramp')},
                       bv_model_sha256=hashlib.sha256(bvm.read_bytes()).hexdigest() if bvm and bvm.is_file() else None)
    behavior = {k: getattr(cfg, k, None) for k in BEHAVIOR_KEYS}
    repo=Path(__file__).resolve().parents[2]
    paths=['python/sts2rl/actor.py','python/sts2rl/train.py','python/sts2rl/model.py','python/sts2rl/value_probe.py',
           'engine/catalog/vocab.txt','engine/Sts2Headless/bin/Release/net9.0/Sts2Headless.dll']
    hashes={p:hashlib.sha256((repo/p).read_bytes()).hexdigest() for p in paths if (repo/p).is_file()}
    return {'reward': rewards, 'reward_id': digest(rewards), 'behavior': behavior, 'behavior_id': digest(behavior),
            'policy': 'sample_temperature_1_current_weights', 'model': dataclasses.asdict(cfg.model), 'code_hashes':hashes,
            'engine':getattr(cfg,'engine',None)}


def moments(v, y):
    v, y = np.asarray(v, np.float64), np.asarray(y, np.float64)
    ok = np.isfinite(v) & np.isfinite(y)
    bad = int((~ok).sum()); v, y = v[ok], y[ok]; e = v - y
    return [len(y), float(y.sum()), float(v.sum()), float(y @ y), float(v @ v), float(y @ v),
            float(np.abs(e).sum()), float(e @ e), float(e.sum()), bad]


def stats(m):
    n = m[0]
    if not n:
        return {'n': 0, 'ev': None, 'mse': None, 'bias': None, 'target_var': None, 'invalid': m[-1]}
    my, mv = m[1] / n, m[2] / n
    vy, vv = max(0., m[3] / n - my * my), max(0., m[4] / n - mv * mv)
    bias = m[8] / n; ve = max(0., m[7] / n - bias * bias)
    return {'n': n, 'ev': 1 - ve / vy if n > 1 and vy > 1e-10 else None, 'target_var': vy,
            'residual_var': ve, 'mse': m[7] / n, 'rmse': math.sqrt(max(0., m[7] / n)), 'mae': m[6] / n,
            'bias': bias, 'mean_target': my, 'mean_value': mv,
            'calibration_slope': (m[5] / n - my * mv) / vv if vv > 1e-10 else None, 'invalid': m[-1]}


def add_moments(items):
    return [sum(m[j] for m in items) for j in range(len(MOMENT_FIELDS))]


def grouped(v, y, *, act, kind, multi, macro, mode=None, source=None, lag=None, horizon=None, terminal=None, split=None, ancient=None):
    """各维度独立分组，外加幕×决策类型；各组重叠，不可跨不同维度相加。"""
    v, y = np.asarray(v), np.asarray(y); macro = np.asarray(macro, bool)
    out = {'all': moments(v, y), 'macro': moments(v[macro], y[macro]), 'combat': moments(v[~macro], y[~macro])}
    columns = {'act': act, 'kind': kind, 'multi': np.asarray(multi, int)}
    for k, a in (('mode', mode), ('source', source), ('lag', lag), ('horizon', horizon), ('terminal', terminal), ('split', split), ('ancient',ancient)):
        if a is not None: columns[k] = a
    for name, col in columns.items():
        col = np.asarray(col)
        for val in np.unique(col[macro]):
            m = macro & (col == val)
            out[f'macro/{name}/{val}'] = moments(v[m], y[m])
    for a in np.unique(np.asarray(act)[macro]):
        for k in np.unique(np.asarray(kind)[macro & (np.asarray(act) == a)]):
            m = macro & (np.asarray(act) == a) & (np.asarray(kind) == k)
            out[f'macro/act_kind/{a}/{k}'] = moments(v[m], y[m])
    if mode is not None:
        for a in np.unique(np.asarray(mode)[~macro]):
            m = ~macro & (np.asarray(mode) == a)
            out[f'combat/mode/{a}'] = moments(v[m], y[m])
    return out


def calibration(v, y):
    """固定奖励单位分箱；首尾含溢出。返回 [n, sum_v, sum_y, sum_error2]。"""
    v, y = np.asarray(v, np.float64), np.asarray(y, np.float64)
    ok = np.isfinite(v) & np.isfinite(y); v, y = v[ok], y[ok]
    edges = [-2., -1., 0., 1., 2., 3., 4., 5., 6., 7., 8., 10.]
    bins = np.searchsorted(edges, v)
    return {'edges': edges, 'bins': [[int((m := bins == j).sum()), float(v[m].sum()), float(y[m].sum()),
                                     float(((v[m] - y[m]) ** 2).sum())] for j in range(len(edges) + 1)]}


def gae_copy(v, r, lam, cut=None, boot=0., gamma=1.):
    """诊断副本；不被训练调用。与 train.gae 用 200 条轨迹逐项核对。"""
    v, r = np.asarray(v), np.asarray(r)
    delta = r + gamma * np.append(v[1:], boot) - v
    a = np.zeros(len(v), np.float32); acc = 0.
    for j in range(len(v)-1, -1, -1):
        acc = delta[j] + gamma * lam * acc; a[j] = acc
        if cut is not None and cut[j]: acc = 0.
    return a + v


def macro_targets(values, rewards, macro, lam, cut=None, boot=0.):
    idx = np.flatnonzero(macro)
    if not len(idx): return idx, np.empty(0, np.float32)
    cs = np.r_[0., np.cumsum(rewards, dtype=np.float64)]
    r = cs[np.r_[idx[1:], len(rewards)]] - cs[idx]
    v = np.asarray(values)[idx].astype(np.float64)
    # train.gae_macro 使用 float64 delta + float32 advantage，保留相同舍入路径。
    delta = r + np.append(v[1:], boot) - v
    a = np.zeros(len(idx), np.float32); acc = 0.
    for j in range(len(idx)-1, -1, -1):
        acc = delta[j] + lam * acc; a[j] = acc
        if cut is not None and cut[idx[j]]: acc = 0.
    return idx, a + np.asarray(values)[idx]


def bucket(x, edges):
    return np.searchsorted(edges, np.asarray(x), side='left')


def quantiles(x):
    x = np.asarray(x); x = x[np.isfinite(x)]
    return {'n': len(x), 'mean': float(x.mean()), 'p05': float(np.quantile(x, .05)), 'p50': float(np.median(x)),
            'p95': float(np.quantile(x, .95)), 'max': float(x.max())} if len(x) else {'n': 0}


def save_npz(path, meta, arrays, ragged=None):
    data = dict(arrays)
    if ragged is not None:
        data.update({f'obs_{f.name}': getattr(ragged, f.name) for f in dataclasses.fields(Ragged) if getattr(ragged, f.name) is not None})
    data['_meta'] = np.frombuffer(json.dumps(meta, ensure_ascii=False, allow_nan=False).encode(), dtype=np.uint8)
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    with tmp.open('wb') as f: np.savez_compressed(f, **data)
    os.replace(tmp, path)


def load_npz(path):
    with np.load(path, allow_pickle=False) as z:
        data = {k: z[k] for k in z.files}
    meta = json.loads(data.pop('_meta').tobytes())
    obs = {k[4:]: data.pop(k) for k in list(data) if k.startswith('obs_')}
    return meta, data, Ragged(**obs) if obs else None


class Writer:
    """有界队列，出错只标记诊断不可用。训练线程不等待磁盘。"""
    def __init__(self, directory, name, max_gb=8.):
        self.root = Path(directory); self.name = name; self.session = uuid.uuid4().hex[:12]
        self.q = queue.Queue(maxsize=4); self.closed = False; self.errors = 0; self.dropped = 0; self.written = 0
        self.counters = {}
        self.last_error = None; self.max_bytes = max_gb * (1 << 30); self.last_space = 0.; self.space_ok = True
        self.thread = threading.Thread(target=self._work, daemon=True, name='value-probe-writer'); self.thread.start()

    def submit(self, fn, *args):
        try: self.q.put_nowait((fn, args)); return True
        except queue.Full: self.dropped += 1; return False

    def _work(self):
        while True:
            job = self.q.get()
            try:
                if job is None: return
                fn, args = job; fn(*args)
            except Exception as e:
                self.errors += 1; self.last_error = f'{type(e).__name__}: {str(e)[:300]}'
                print(f'价值诊断 {self.name} 写入失败：{self.last_error}', flush=True)
            finally:
                self.q.task_done()

    def status(self):
        return {'session': self.session, 'written': self.written, 'errors': self.errors, 'dropped': self.dropped,
                'queued': self.q.qsize(), 'last_error': self.last_error, 'space_ok': self.space_ok, **self.counters}

    def append(self, record):
        self.root.joinpath('metrics').mkdir(parents=True, exist_ok=True)
        record = dict(record, schema=SCHEMA, time=time.time(), producer=self.name, session=self.session)
        p = self.root / 'metrics' / f'{self.name}_{self.session}_{time.strftime("%Y%m%d%H", time.gmtime())}.jsonl'
        with p.open('a', encoding='utf-8') as f: f.write(json.dumps(record, ensure_ascii=False, allow_nan=False, separators=(',', ':')) + '\n')
        self.written += 1
        tmp = self.root / f'{self.name}_status.json.tmp'; dst = tmp.with_suffix('')
        tmp.write_text(json.dumps(dict(self.status(), time=time.time()), ensure_ascii=False)); os.replace(tmp, dst)

    def archive_allowed(self):
        if time.time() - self.last_space > 60:
            self.last_space = time.time()
            n = 0
            for folder in ('episodes', 'batches'):
                for p in (self.root/folder).glob('*.npz'):
                    try: n += p.stat().st_size
                    except FileNotFoundError: pass  # 同步进程已把这个完整文件搬到本机
            self.space_ok = n < self.max_bytes
        return self.space_ok

    def close(self):
        if self.closed: return
        self.closed = True
        try: self.q.put(None, timeout=2.)
        except queue.Full: return
        self.thread.join(timeout=5.)


class ActorProbe:
    def __init__(self, cfg, rank):
        self.cfg, self.rank = cfg, rank; self.ctx = context(cfg); self.seq = 0
        self.writer = Writer(Path(cfg.run_dir)/'value_probe', f'actor{rank}', cfg.value_probe_max_gb)

    def begin(self, t):
        try:
            self._begin(t)
        except Exception as e:
            t.probe = None; self.writer.errors += 1; self.writer.last_error = f'begin: {e}'

    def _begin(self, t):
        self.seq += 1
        key = str(t.spec.get('seed') or digest(t.spec))
        root_key = digest(['act_start',(t.req or {})['act_start']]) if (t.req or {}).get('act_start') else key
        fold = int(digest(['split', root_key])[:8], 16) % 10
        selected = t.mode == 1 and int(digest(['sample', key])[:16], 16) / (1 << 64) < self.cfg.value_probe_episode_frac
        self.writer.counters['started'] = self.seq
        self.writer.counters['selected'] = self.writer.counters.get('selected',0) + int(selected)
        t.probe = {'episode_id': f'{self.writer.session}-{self.rank}-{self.seq}', 'root_id': digest(root_key),
                   'split': 'test' if fold == 0 else 'validation' if fold == 1 else 'train',
                   'selected': selected, 'segments': [], 'version_start': t.version, 'reward_events': []}

    def finish(self, t, final):
        if t.probe and t.probe['selected'] and t.obs:
            self.writer.counters['completed'] = self.writer.counters.get('completed',0)+1
            self.writer.submit(self._finish, t, {'win': bool(final.victory) and not final.stuck, 'stuck': bool(final.stuck),
                                               'terminated': bool(getattr(final,'done',final.victory)) and not final.stuck,
                                               'act': int(final.act), 'floor': int(final.floor)})

    def _finish(self, t, final):
        n = len(t.obs); cfg = self.cfg
        macro = np.array([t.mode == 1 and not o.in_combat and int(o.decision) not in (0, 1, 12) for o in t.obs], bool)
        v = np.asarray(t.values, np.float32); r = np.asarray(t.rewards, np.float32); forced = np.asarray(t.forced, bool)
        idx, full_lam = macro_targets(v, r, macro, cfg.macro_lam, forced)
        _, full_no_cut = macro_targets(v, r, macro, cfg.macro_lam)
        mc = np.cumsum(r[::-1], dtype=np.float64)[::-1][idx]
        segments = list(t.probe['segments']) + [{'start': int(t.seg0), 'end': n, 'boot': 0., 'terminal': final.get('terminated',not final['stuck'])}]
        seg_target = np.full(n, np.nan, np.float32); distance = np.zeros(n, np.int32); segment_id = np.zeros(n, np.int32)
        for sn, seg in enumerate(segments):
            a, b = seg['start'], seg['end']; ii, rr = macro_targets(v[a:b], r[a:b], macro[a:b], cfg.macro_lam, forced[a:b], seg['boot'])
            seg_target[a+ii] = rr; distance[a+ii] = np.arange(len(ii), 0, -1); segment_id[a:b] = sn
        act = np.array([int(o.act)+1 for o in t.obs], np.int8); kind = np.array([int(o.decision) for o in t.obs], np.int8)
        multi = np.array([o.num_actions > 1 for o in t.obs], bool)
        ancient=np.asarray(t.anc,np.int8) if len(t.anc)==n else np.zeros(n,np.int8)
        valid = final.get('terminated',not final['stuck'])  # 卡住或无动作但未终局，是截断，不能当真实 MC 标签。
        groups = {name: grouped(v[idx], y, act=act[idx], kind=kind[idx], multi=multi[idx], macro=np.ones(len(idx), bool),ancient=ancient[idx])
                  for name, y in [('segment', seg_target[idx]), ('full_lambda', full_lam), ('full_lambda_no_cut', full_no_cut),
                                  *([('mc_behavior', mc)] if valid else [])]}
        arrays = {'values': v, 'rewards': r, 'logp': np.asarray(t.logp, np.float32), 'logp_mu': np.asarray(t.logp_mu, np.float32),
                  'versions': np.asarray(t.vers, np.int64), 'actions': np.asarray(t.actions, np.int64), 'forced': forced,
                  'macro': macro, 'act': act, 'kind': kind, 'multi': multi, 'ancient':ancient,'macro_index': idx, 'segment_id': segment_id,
                  'macro_to_boundary': distance[idx], 'segment_target': seg_target[idx], 'full_lambda_target': full_lam,
                  'full_lambda_no_cut': full_no_cut, 'mc_behavior_target': mc}
        component_sum = np.zeros(n, np.float64)
        for name in sorted(set(e[1] for e in t.probe['reward_events'])):
            a = np.zeros(n, np.float64)
            for step, key, amount in t.probe['reward_events']:
                if key == name: a[step] += amount
            arrays['reward_component_'+name] = a; component_sum += a
        arrays['reward_component_residual'] = r.astype(np.float64)-component_sum
        # 简单进度/资源基线所需公开特征；按整局分组拟合，不能随机拆帧。
        resources=[]
        for j in idx:
            o=t.obs[j]; ti=o.tok_ints; g=o.tok_feats[np.flatnonzero(ti[:,0]==1)[0]]
            resources.append([1., float(o.act), float(o.floor)/50., float(o.hp)/max(1,float(o.max_hp)), float(o.gold)/500.,
                              float(((ti[:,0]==4)&(ti[:,2]==7)).sum())/40., float(((ti[:,0]==5)&(ti[:,2]==1)).sum())/20.])
        arrays['baseline_features']=np.asarray(resources,np.float32).reshape(-1,7)
        arrays['label_win']=np.full(len(idx),float(final['win']) if valid else np.nan,np.float32)
        arrays['label_reach_act2']=np.full(len(idx),float(final['act']>=1) if valid else np.nan,np.float32)
        arrays['label_reach_act3']=np.full(len(idx),float(final['act']>=2) if valid else np.nan,np.float32)
        arrays['label_remaining_floors']=np.asarray([max(0,final['floor']-int(t.obs[j].floor)) for j in idx],np.float32)
        entries=[]
        for j,o in enumerate(t.obs):
            if o.extra and 'combat' in o.extra:
                c=o.extra['combat'];ti=o.tok_ints
                entries.append({'step':j,'act':int(o.act),'floor':int(o.floor),'hp':int(o.hp),'max_hp':int(o.max_hp),
                                'potions':int(((ti[:,0]==6)&(ti[:,2]==1)).sum()),'room':c.get('room'),'encounter':c.get('encounter')})
        bosses=[c for c in entries if c['room']=='Boss']
        future=np.array([next((k for k,c in enumerate(bosses) if c['step']>=j),-1) for j in idx],int)
        arrays['label_next_boss_reached']=np.asarray([float(k>=0) if valid else np.nan for k in future],np.float32)
        arrays['label_next_boss_win']=np.asarray([float(k>=0 and k<len(t.bosses) and t.bosses[k][1]) if valid else np.nan for k in future],np.float32)
        arrays['label_next_boss_hp_fraction']=np.asarray([bosses[k]['hp']/max(1,bosses[k]['max_hp']) if valid and k>=0 else np.nan for k in future],np.float32)
        arrays['label_next_boss_potions']=np.asarray([bosses[k]['potions'] if valid and k>=0 else np.nan for k in future],np.float32)
        meta = dict(t.probe, schema=SCHEMA, type='episode', **self.ctx, rank=self.rank, final=final,
                    version_end=int(t.vers[-1]), segments=segments, request=t.req, policy_versions=sorted(set(map(int,t.vers))),
                    target_policy='recorded_behavior_sequence_not_frozen', mc_valid=valid,
                    snapshots=[{'step': j, 'snapshot': o.extra['snapshot']} for j,o in enumerate(t.obs) if o.extra and 'snapshot' in o.extra],
                    primitive_steps=n, macro_steps=len(idx), combat_entries=entries,boss_results=t.bosses,created=time.time())
        archive = None
        if self.writer.archive_allowed() and len(idx):
            archive = f"episodes/{t.probe['episode_id']}.npz"
            # 只保存局外原始观测；完整时间轴的动作、奖励和行为概率保留。特权字段原样另列。
            save_npz(self.writer.root/archive, meta, arrays, Ragged.from_obs([t.obs[j] for j in idx], np.ones(len(idx), np.int64)))
        record = {'type': 'episode', 'episode_id': t.probe['episode_id'], 'root_id': t.probe['root_id'], 'split': t.probe['split'],
                  'version_start': t.version, 'version_end': int(t.vers[-1]), 'policy_span': int(max(t.vers)-min(t.vers)),
                  'reward_id': self.ctx['reward_id'], 'behavior_id': self.ctx['behavior_id'], 'archive': archive,
                  'mc_valid': valid, 'win': final['win'], 'stuck': final['stuck'], 'steps': n, 'macro_steps': len(idx),
                  'segments': len(segments), 'horizon': quantiles(distance[idx]), 'groups': groups,
                  'calibration_mc': calibration(v[idx], mc) if valid else None,
                  'segment_vs_mc': moments(seg_target[idx], mc) if valid else None,
                  'full_vs_mc': moments(full_lam, mc) if valid else None,
                  'reward_component_totals': {k.removeprefix('reward_component_'):float(a.sum()) for k,a in arrays.items() if k.startswith('reward_component_')},
                  'reward_residual_max': float(np.abs(arrays['reward_component_residual']).max()),
                  'advantage_sign': {'n': len(idx), 'segment_full_disagree': int(((seg_target[idx]-v[idx])*(full_lam-v[idx]) < 0).sum()),
                                     'segment_mc_disagree': int(((seg_target[idx]-v[idx])*(mc-v[idx]) < 0).sum()) if valid else None},
                  'writer': self.writer.status()}
        self.writer.append(record)


def reward_event(t, name, amount, index=-1):
    """在原奖励赋值处旁记，不参与或替代原赋值。"""
    if t.probe and t.probe['selected'] and amount:
        step = index if index >= 0 else len(t.rewards)+index
        if step >= 0: t.probe['reward_events'].append([step, name, float(amount)])


def diagnostic_gradients(net, b, target, old_logp, adv, weights, macro, cfg, population_macro=None):
    """只返回 autograd.grad，不写 .grad；按共享编码器和输入嵌入汇总。"""
    import torch
    import torch.nn.functional as F
    params = [p for name,p in net.named_parameters() if p.requires_grad and not name.startswith(('cand_', 'act_emb', 'none_ref', 'value_', 'phi_', 'priv_', 'fut_'))]
    shared_n=len(params);params+=list(net.value_mlp_m.parameters())
    logits, value = net(b); value = value.float(); logits = logits.float()
    m = torch.as_tensor(macro, device=value.device, dtype=torch.bool)
    ret = torch.as_tensor(target, device=value.device, dtype=torch.float32)
    A = torch.as_tensor(adv, device=value.device); W = torch.as_tensor(weights, device=value.device)
    population_macro = float(m.float().mean()) if population_macro is None else population_macro
    sw = torch.where(m, population_macro/m.float().mean(), (1-population_macro)/(~m).float().mean())
    dist = torch.distributions.Categorical(logits=logits)
    lp = dist.log_prob(b['probe_action'])
    ratio = (lp-torch.as_tensor(old_logp, device=value.device)).exp()
    losses = {'macro_value': F.mse_loss(value[m],ret[m]), 'combat_value': F.mse_loss(value[~m],ret[~m]),
              'policy': -(sw*W*torch.minimum(ratio*A,ratio.clamp(1-cfg.clip,1+cfg.clip)*A)).sum()/(sw*W).sum().clamp(min=1)}
    ent_weight = torch.where(m,cfg.ent_coef_macro,cfg.ent_coef)
    losses['other'] = losses['policy'] + cfg.vf_coef*(1-population_macro)*losses['combat_value'] - (sw*ent_weight*dist.entropy()).mean()
    gradients = {k: torch.autograd.grad(l, params, retain_graph=i<len(losses)-1, allow_unused=True)
                 for i,(k,l) in enumerate(losses.items())}
    def dot(a,b):
        return sum((x.float()*y.float()).sum() for x,y in zip(a,b) if x is not None and y is not None)
    head_grad=gradients['macro_value'][shared_n:];head_norm=float(dot(head_grad,head_grad).sqrt())
    gradients={k:g[:shared_n] for k,g in gradients.items()}
    norm = {k: dot(g,g).sqrt() for k,g in gradients.items()}
    result = {'n_macro': int(m.sum()), 'n_combat': int((~m).sum()), 'scope': 'shared_encoder_and_input_embeddings',
              'population_macro': population_macro,
              'macro_head_norm':head_norm,
              'norm': {k: float(x.detach()) for k,x in norm.items()},
              'weighted_value_norm': {'macro':float(norm['macro_value'])*cfg.vf_coef*population_macro,
                                      'combat':float(norm['combat_value'])*cfg.vf_coef*(1-population_macro)}, 'cosine': {}}
    for a,bkey in [('macro_value','combat_value'),('macro_value','policy'),('combat_value','policy'),('macro_value','other')]:
        den = float(norm[a]*norm[bkey]); result['cosine'][a+'__'+bkey] = float(dot(gradients[a],gradients[bkey])/den) if den>1e-12 else None
    return result


class LearnerProbe:
    def __init__(self, cfg):
        self.cfg=cfg; self.ctx=context(cfg); self.writer=Writer(Path(cfg.run_dir)/'value_probe', 'learner', cfg.value_probe_max_gb)
        self.calls=0; self.errors=0

    def observe(self, net, data, trajs, targets, fresh, macro, version, *, actions, behavior_logp, advantages, policy_weights):
        self.calls+=1
        if self.calls!=1 and self.calls % self.cfg.value_probe_every: return None
        started=time.perf_counter()
        original_fast=net.fast
        try:
            # 打包 jagged CUDA 注意力通常要求半精度；诊断明确使用常规 FP32 路径。
            # 仅此同步调用期间切换，finally 恢复，不影响随后的编译训练前向。
            net.fast=False
            result=self._observe(net,data,trajs,targets,fresh,macro,version,actions,behavior_logp,advantages,policy_weights)
            result['seconds']=time.perf_counter()-started; self.writer.submit(self.writer.append,result)
            return {'seconds':result['seconds'],'sample_n':result['sample_n'],'errors':self.errors,**self.writer.status()}
        except Exception as e:
            self.errors+=1
            error={'type':'error','version':version,'error':f'{type(e).__name__}: {str(e)[:300]}'}
            self.writer.submit(self.writer.append,error); print('价值诊断失败（训练继续）：'+error['error'],flush=True)
            return {'errors':self.errors,'last_error':error['error'],'seconds':time.perf_counter()-started}
        finally:
            net.fast=original_fast

    def _observe(self,net,data,trajs,targets,fresh,macro,version,actions,behavior_logp,advantages,policy_weights):
        import torch
        n=len(data); old=np.concatenate([t['values'] for t in trajs]); kind=data.header[:,1]
        # 幕来自 GLOBAL 特征 0。
        glob=np.flatnonzero(data.tok_ints[:,0]==1)
        if len(glob)!=n: raise ValueError('每个观测必须恰好有一个 GLOBAL 记录')
        act=np.rint(data.tok_feats[glob,0]).astype(np.int64)+1
        multi=np.diff(data.cand_off)>1; mode=data.modes(); lag=version-np.concatenate([t['vers'] for t in trajs])
        horizon=np.zeros(n,np.int32); terminal=np.zeros(n,np.int8); sources=np.empty(n,dtype='<U6'); roots=[]; splits=[]; pos=0
        for t in trajs:
            ii=np.flatnonzero(t['macro']); horizon[pos+ii]=np.arange(len(ii),0,-1)
            terminal[pos:pos+t['len']]=int(t.get('probe_terminated',not t.get('segment') and not t.get('stuck')))
            sources[pos:pos+t['len']]='remote' if t.get('remote') else 'local'
            meta=t.get('value_probe') or {}; roots.extend([meta.get('episode_id',f'unknown-{pos}')]*t['len'])
            splits.extend([meta.get('split','unknown')]*t['len']);pos+=t['len']
        labels={'act':act,'kind':kind,'multi':multi,'macro':macro,'mode':mode,'source':sources,
                'lag':bucket(lag,[0,1,2,4,6]),'horizon':bucket(horizon,[1,4,8,16,32]),'terminal':terminal,'split':np.asarray(splits)}
        labels['ancient']=np.concatenate([t['anc'] if t.get('anc') is not None and len(t['anc'])==t['len'] else np.zeros(t['len'],np.int8) for t in trajs]).astype(np.int8)
        use=np.flatnonzero(fresh)
        record={'type':'batch','version':version,'reward_id':self.ctx['reward_id'],'behavior_id':self.ctx['behavior_id'],
                'n':len(use),'unique_episodes':len(set(np.asarray(roots)[use])), 'sample_method':'evenly_spaced_within_macro_and_combat',
                'old_groups':grouped(old[use],targets[use],**{k:v[use] for k,v in labels.items()}),
                'macro_horizon':quantiles(horizon[fresh&macro]), 'macro_fraction':float((fresh&macro).sum()/max(1,len(use)))}
        def pick(mask,k):
            ids=np.flatnonzero(mask)
            return ids[np.linspace(0,len(ids)-1,min(k,len(ids)),dtype=int)] if len(ids) else ids
        idx=np.r_[pick(fresh&macro,self.cfg.value_probe_samples),pick(fresh&~macro,max(16,self.cfg.value_probe_samples//4))]
        r=data.select(idx); device=next(net.parameters()).device
        devices=[device.index if device.index is not None else torch.cuda.current_device()] if device.type=='cuda' else []
        with torch.random.fork_rng(devices=devices):
            b=collate_ragged(r).to(device)
            with torch.no_grad():
                logits,v=net(b); current=v.float().cpu().numpy(); probs=logits.float().softmax(-1)
                # 去特权：特权特征清零，整局预生成内容（类型 17，整条记录都是特权，决策记录 259）整条去掉
                public=dataclasses.replace(r,tok_priv=np.zeros_like(r.tok_priv)).drop_kind(int(TokenKind.FUTURE_CONTENT)).drop_kind(int(TokenKind.ORACLE))
                pub_logits,pv=net(collate_ragged(public).to(device)); pub=pv.float().cpu().numpy()
                mask=torch.isfinite(logits)&torch.isfinite(pub_logits)
                policy_priv_delta=float((logits.float()[mask]-pub_logits.float()[mask]).abs().max()) if mask.any() else 0.
                selected_prob=probs.gather(1,torch.as_tensor(actions[idx],device=device)[:,None]).squeeze(1).cpu().numpy()
            sample_labels={k:v[idx] for k,v in labels.items()}
            record['sample_n']=len(idx)
            record['paired']={name:grouped(pred,targets[idx],**sample_labels) for name,pred in [('old',old[idx]),('current',current),('public',pub)]}
            record['current_calibration']=calibration(current[macro[idx]],targets[idx][macro[idx]])
            record['current_vs_old']=moments(current[macro[idx]],old[idx][macro[idx]])
            record['privileged_policy_max_delta']=policy_priv_delta
            record['advantage_sign_changes']=int(((targets[idx]-old[idx])*(targets[idx]-current)<0)[macro[idx]].sum())
            record['gradients']=None
            if (self.calls==1 or self.calls%self.cfg.value_probe_grad_every==0) and macro[idx].any() and (~macro[idx]).any():
                b['probe_action']=torch.as_tensor(actions[idx],device=device)
                record['gradients']=diagnostic_gradients(net,b,targets[idx],behavior_logp[idx],advantages[idx],policy_weights[idx],macro[idx],self.cfg,record['macro_fraction'])
        meta={'schema':SCHEMA,'type':'batch','version':version,**self.ctx,'episode_id':np.asarray(roots)[idx].tolist(),
              'split':np.asarray(splits)[idx].tolist(),'producer_session':self.writer.session,'information':'privileged_and_public_ablated',
              'prediction_precision':'fp32_probe','target':'unchanged_production_lambda_return','created':time.time()}
        arrays={'old_value':old[idx],'current_value':current,'public_value':pub,'target':targets[idx],
                'actions':actions[idx],'behavior_logp':behavior_logp[idx],'advantages':advantages[idx],
                'policy_logp':np.concatenate([t['logp'] for t in trajs])[idx],
                'policy_weights':policy_weights[idx],'current_action_probability':selected_prob,'lag':lag[idx],
                'macro':macro[idx],'act':act[idx],'kind':kind[idx],'multi':multi[idx],'macro_to_boundary':horizon[idx]}
        def archive():
            if self.writer.archive_allowed(): save_npz(self.writer.root/'batches'/f'{self.writer.session}_v{version}.npz',meta,arrays,r)
        self.writer.submit(archive)
        return record
