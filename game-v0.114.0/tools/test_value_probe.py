"""价值诊断的数值、无侵入性与序列归档测试。无需游戏或 GPU。"""
import copy
import json
import os
from pathlib import Path
import queue
import sys
import tempfile
import unittest
from types import SimpleNamespace

sys.path[:0]=[str(Path(__file__).resolve().parents[1]/'python'),str(Path(__file__).resolve().parent)]
import numpy as np
import torch
torch.set_num_threads(1)
from sts2env.protocol import Obs, ActionKind, DecisionKind
from sts2rl.actor import Trajectory, _finish, _steps, send_segment, ancient_rows, explore_spec
from sts2rl.config import TrainConfig
from sts2rl.features import Ragged, collate_ragged
from sts2rl.model import ModelConfig, PolicyValueNet
from sts2rl.train import gae, gae_macro
from sts2rl.value_probe import (ActorProbe, LearnerProbe, moments, stats, add_moments, macro_targets,
                               gae_copy, load_npz, save_npz, grouped, reward_event)
from value_probe_summary import ingest, build_summary
from value_probe_analysis import continuation_summary


def observation(macro=True, act=0, floor=1):
    h=np.zeros(34,np.int32);h[0]=19;h[1]=3 if macro else 1;h[2]=4;h[3]=2;h[11]=not macro
    ti=np.zeros((4,5),np.int32);ti[:,3]=-1;ti[:,0]=[1,2,4,4];ti[:,1]=[1,2,3,4];ti[:,2]=[0,1,7 if macro else 3,7 if macro else 4]
    tf=np.zeros((4,20),np.float32);tf[0,[0,1,3,4,5,13]]=[act,floor,100,60,75,not macro];tf[1,[0,1,5]]=[60,75,1]
    tp=np.zeros((4,2),np.float32);tp[0]=[3,4];tp[3,0]=.7
    ci=np.array([[7 if macro else 2,2,-1,3,0],[8 if macro else 1,-1,-1,0,0]],np.int32)
    return Obs(h,ti,tf,tp,ci,np.zeros((2,12),np.float32))


class ValueProbeTest(unittest.TestCase):
    def test_ancient_marker_does_not_depend_on_exploration(self):
        vocab=['<pad>','RELIC.ONE','RELIC.TWO','CARD.TEST']
        o=observation();o.header[1]=int(DecisionKind.EVENT)
        o.tok_ints[2:,0]=9;o.tok_ints[2:,4]=[1,2]
        o.cand_ints[:,0]=int(ActionKind.EVENT_OPTION);o.cand_ints[:,1]=[2,3]
        before=copy.deepcopy(o);rng=np.random.get_state();trng=torch.get_rng_state().clone()
        # 关闭全部探索、只开普通事件探索时，旧逻辑不会进入先古标记赋值分支。
        self.assertIsNone(explore_spec(o,vocab,{}))
        self.assertIsNone(explore_spec(o,vocab,{'event':.1}))
        for probs in ({},{'event':.1},{'ancient':.1}):
            explore_spec(o,vocab,probs)
            self.assertTrue(ancient_rows([o],[1],vocab)[0])
        # 探索已过期时完全不调用 explore_spec，也必须正确标记。
        self.assertTrue(ancient_rows([o],[1],vocab)[0])
        non_event=copy.deepcopy(o);non_event.header[1]=int(DecisionKind.SHOP)
        one_relic=copy.deepcopy(o);one_relic.tok_ints[3,4]=0
        combat=copy.deepcopy(o);combat.header[11]=1
        np.testing.assert_array_equal(ancient_rows([o,non_event,one_relic,combat,o],[1,1,1,1,0],vocab),[True,False,False,False,False])
        self.assertEqual(ancient_rows([],[],vocab).dtype,np.bool_)
        for name in ['header','tok_ints','tok_feats','tok_priv','cand_ints','cand_feats']:
            np.testing.assert_array_equal(getattr(o,name),getattr(before,name))
        for x,y in zip(rng,np.random.get_state()):np.testing.assert_equal(x,y)
        self.assertTrue(torch.equal(trng,torch.get_rng_state()))

    def test_merge_and_constant_bias(self):
        y=np.arange(100,dtype=float);v=y+2
        a=add_moments([moments(v[:37],y[:37]),moments(v[37:],y[37:])])
        self.assertEqual(a,moments(v,y));s=stats(a)
        self.assertAlmostEqual(s['ev'],1);self.assertAlmostEqual(s['mse'],4);self.assertAlmostEqual(s['bias'],2)
        self.assertIsNone(stats(moments([0,1],[1,1]))['ev'])
        self.assertEqual(moments([float('nan'),1],[2,1])[-1],1)

    def test_200_random_trajectories_match_production_targets(self):
        rng=np.random.default_rng(456)
        for _ in range(200):
            n=int(rng.integers(1,250));v=rng.normal(size=n).astype(np.float32);r=rng.normal(size=n).astype(np.float32)
            macro=rng.random(n)<.25;cut=rng.random(n)<.06;boot=float(rng.normal());lam=float(rng.choice([0,.8,.95,1]))
            a,target=gae(v,r,1.,lam,cut,boot)
            np.testing.assert_array_equal(gae_copy(v,r,lam,cut,boot),target)
            gae_macro(v,r,macro,lam,a,target,cut,boot)
            idx,got=macro_targets(v,r,macro,lam,cut,boot)
            np.testing.assert_array_equal(got,target[idx])

    def test_npz_no_pickle_roundtrip(self):
        r=Ragged.from_obs([observation(),observation(False)],np.ones(2,np.int64))
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'x.npz';save_npz(p,{'schema':1},{'v':np.array([1,2])},r)
            meta,arr,rr=load_npz(p)
            self.assertEqual(meta['schema'],1);np.testing.assert_array_equal(arr['v'],[1,2])
            for k in r.__dataclass_fields__:np.testing.assert_array_equal(getattr(r,k),getattr(rr,k))

    def test_200_episode_metadata_does_not_change_rewards_or_messages(self):
        with tempfile.TemporaryDirectory() as d:
            cfg=TrainConfig(run_dir=d,value_probe=True,value_probe_episode_frac=1.)
            probe=ActorProbe(cfg,0)
            try:
                for i in range(200):
                    a=Trajectory(1,40,{'seed':f'test{i}'})
                    a.obs=[observation(),observation(False),observation(act=1,floor=18)]
                    a.values=[.4,.5,.6];a.rewards=[.02,.5,0.];a.actions=[0,1,0];a.vers=[40,40,41]
                    a.logp=[-.6]*3;a.logp_mu=[-.6]*3;a.forced=[False]*3;a.anc=[False]*3;a.phi=[0.]*3
                    a.hp_start=60;a.max_hp=75
                    b=copy.deepcopy(a);probe.begin(b)
                    qa,qb=queue.Queue(),queue.Queue()
                    final=SimpleNamespace(victory=i%2==0,stuck=False,hp=30,floor=49,act=2)
                    _finish(cfg,a,final,qa,0);_finish(cfg,b,final,qb,0)
                    np.testing.assert_array_equal(a.rewards,b.rewards)
                    x,y=qa.get()[2],qb.get()[2]
                    for key in ('values','rewards','macro','logp','logp_mu','actions','vers'):
                        np.testing.assert_array_equal(x[key],y[key])
                    self.assertEqual(x['reward'],y['reward']);self.assertIn('value_probe',y)
                    before=copy.deepcopy(b.rewards);reward_event(b,'audit',3.,0);self.assertEqual(before,b.rewards)
                # 同一根种子不因进程/会话改变训练/验证/测试划分。
                c=Trajectory(1,42,{'seed':'test199'});probe.begin(c);self.assertEqual(c.probe['split'],b.probe['split'])
                x=Trajectory(1,42,{'seed':'new1'});y=Trajectory(1,43,{'seed':'new2'})
                x.req={'act_start':{'Act':2,'fixture':'same-root'}};y.req=copy.deepcopy(x.req)
                probe.begin(x);probe.begin(y);self.assertEqual(x.probe['root_id'],y.probe['root_id']);self.assertEqual(x.probe['split'],y.probe['split'])
            finally:probe.writer.close()

    def test_segment_boot_and_invalid_terminal(self):
        with tempfile.TemporaryDirectory() as d:
            cfg=TrainConfig(run_dir=d,value_probe=True,value_probe_episode_frac=1.)
            probe=ActorProbe(cfg,0);t=Trajectory(1,3,{'seed':'segment-test'});probe.begin(t)
            t.obs=[observation(),observation(False)];t.values=[1.,2.];t.rewards=[0.,.5];t.actions=[0,0]
            t.vers=[3,3];t.logp=[-.7]*2;t.logp_mu=[-.7]*2;t.forced=[False]*2;t.anc=[False]*2;t.phi=[0.]*2
            q=queue.Queue();send_segment(t,4.,q,0)
            self.assertEqual(q.get()[2]['boot'],4.);self.assertEqual(t.probe['segments'][0]['end'],2)
            t.obs.append(observation());t.values.append(4.);t.rewards.append(1.);t.actions.append(0)
            t.vers.append(4);t.logp.append(-.7);t.logp_mu.append(-.7);t.forced.append(False)
            probe._finish(t,{'win':False,'stuck':True,'act':0,'floor':2})
            p=next((Path(d)/'value_probe/episodes').glob('*.npz'));meta,a,_=load_npz(p)
            self.assertFalse(meta['mc_valid']);self.assertTrue(np.isnan(a['label_win']).all())
            np.testing.assert_allclose(a['segment_target'],[4.5,1.]);probe.writer.close()

    def test_learner_probe_preserves_rng_parameters_gradients_and_policy(self):
        with tempfile.TemporaryDirectory() as d:
            cfg=TrainConfig(run_dir=d,value_probe=True,value_probe_every=1,value_probe_grad_every=1,value_probe_samples=4,
                            model=ModelConfig(d=32,layers=1,heads=2,ff_mult=2))
            device=os.environ.get('VALUE_PROBE_TEST_DEVICE','cpu')
            torch.manual_seed(817);net=PolicyValueNet(cfg.model).to(device).train()
            net.fast=device=='cuda'
            obs=[observation(i%2==0,act=i%3,floor=i+1) for i in range(8)]
            data=Ragged.from_obs(obs,np.ones(8,np.int64));b=collate_ragged(data).to(device)
            macro=np.array([i%2==0 for i in range(8)]);target=np.arange(8,dtype=np.float32)/4;actions=np.zeros(8,np.int64)
            with torch.autocast(device.split(':')[0],dtype=torch.bfloat16,enabled=device=='cuda'):
                logits,old=net(b)
            old=old.detach().float().cpu().numpy();lp=torch.distributions.Categorical(logits=logits.float()).log_prob(torch.zeros(8,dtype=torch.long,device=device)).detach().cpu().numpy()
            for p in net.parameters():p.grad=torch.ones_like(p)*.3
            params={k:v.detach().clone() for k,v in net.state_dict().items()};grads=[p.grad.clone() for p in net.parameters()]
            rng=torch.get_rng_state().clone();npstate=np.random.get_state()
            cuda_rng=torch.cuda.get_rng_state().clone() if device=='cuda' else None
            traj={'values':old,'logp':lp,'vers':np.full(8,5),'macro':macro,'len':8,'segment':True,'value_probe':{'episode_id':'e','split':'test'}}
            probe=LearnerProbe(cfg)
            result=probe.observe(net,data,[traj],target,np.ones(8,bool),macro,5,actions=actions,behavior_logp=lp,
                                 advantages=np.array([1,-1]*4,np.float32),policy_weights=np.ones(8,np.float32))
            probe.writer.q.join();probe.writer.close()
            self.assertEqual(result['errors'],0,result)
            self.assertTrue(torch.equal(rng,torch.get_rng_state()))
            if cuda_rng is not None:self.assertTrue(torch.equal(cuda_rng,torch.cuda.get_rng_state()))
            for x,y in zip(npstate,np.random.get_state()):np.testing.assert_equal(x,y)
            for k,v in net.state_dict().items():self.assertTrue(torch.equal(params[k],v),k)
            for p,g in zip(net.parameters(),grads):self.assertTrue(torch.equal(p.grad,g))
            self.assertEqual(net.fast,device=='cuda')
            with torch.autocast(device.split(':')[0],dtype=torch.bfloat16,enabled=device=='cuda'):
                again=net(b)[0]
            self.assertTrue(torch.equal(logits,again))
            event=json.loads(next((Path(d)/'value_probe/metrics').glob('*.jsonl')).read_text().splitlines()[-1])
            self.assertEqual(event['privileged_policy_max_delta'],0.)
            self.assertIsNotNone(event['gradients'])
            self.assertIn('macro/act/1',event['old_groups'])

    def test_incremental_collection_handles_partial_lines_and_duplicates(self):
        import sqlite3
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);(root/'b/metrics').mkdir(parents=True);p=root/'b/metrics/a.jsonl'
            event={'schema':1,'type':'error','time':100.,'error':'fixture'};line=json.dumps(event).encode()+b'\n'
            p.write_bytes(line+line[:10]);db=sqlite3.connect(':memory:')
            self.assertEqual(ingest(db,root),0);self.assertEqual(db.execute('select count(*) from events').fetchone()[0],1)
            with p.open('ab') as f:f.write(line[10:])
            ingest(db,root);ingest(db,root)
            self.assertEqual(db.execute('select count(*) from events').fetchone()[0],2)
            self.assertEqual(build_summary(db,root)['all_counts']['error'],2)

    def test_completed_episode_archive_to_dashboard_summary(self):
        import shutil,sqlite3
        with tempfile.TemporaryDirectory() as d:
            cfg=TrainConfig(run_dir=str(Path(d)/'remote'),value_probe=True,value_probe_episode_frac=1.)
            p=ActorProbe(cfg,0);t=Trajectory(1,3,{'seed':'complete'});p.begin(t)
            t.obs=[observation(),observation(False)];t.values=[1.,2.];t.rewards=[0.,.5];t.actions=[0,0]
            t.vers=[3,3];t.logp=[-.7]*2;t.logp_mu=[-.7]*2;t.forced=[False]*2;t.anc=[False]*2;t.phi=[0.]*2
            send_segment(t,10.,queue.Queue(),0)
            t.obs.append(observation());t.values.append(10.);t.rewards.append(4.);t.actions.append(0)
            t.vers.append(4);t.logp.append(-.7);t.logp_mu.append(-.7);t.forced.append(False)
            reward_event(t,'boss1',.5,1);reward_event(t,'victory',4.,2)
            p._finish(t,{'win':True,'stuck':False,'terminated':True,'act':0,'floor':2});p.writer.close()
            local=Path(d)/'local';shutil.copytree(Path(cfg.run_dir)/'value_probe',local/'b')
            meta,a,_=load_npz(next((local/'b/episodes').glob('*.npz')))
            np.testing.assert_allclose(a['segment_target'],[10.5,4.])
            np.testing.assert_allclose(a['full_lambda_target'],[4.8,4.],rtol=1e-6)
            np.testing.assert_allclose(a['mc_behavior_target'],[4.5,4.])
            self.assertEqual(float(np.abs(a['reward_component_residual']).max()),0.)
            db=sqlite3.connect(':memory:');ingest(db,local);s=build_summary(db,local)
            self.assertEqual(s['valid_mc_episodes'],1);self.assertEqual(s['archive_files'],1)
            self.assertAlmostEqual(s['episode_targets']['mc_behavior']['macro']['mse'],24.125)

    def test_fixed_policy_continuations_and_rejection_of_duplicates(self):
        rows=[]
        for i in range(200):
            for j in range(4):
                rows.append(dict(schema=1,kind='value',root_id=f'r{i}',world_id=f'w{j}',policy_id='fixed',reward_id='reward',
                    info_id=f'info{i}',information='public',prediction=float(i%5),**{'return':float(i%5)+(-1 if j%2 else 1)},provenance='test fixture'))
        out=continuation_summary(rows);self.assertEqual(out['repeated_roots'],200)
        self.assertGreater(out['noise_ceiling']['within'],0);self.assertEqual(out['single_outcome']['mse'],1.)
        with self.assertRaises(ValueError):continuation_summary(rows+[rows[0]])
        mixed=copy.deepcopy(rows);mixed[0]['policy_id']='other'
        with self.assertRaises(ValueError):continuation_summary(mixed)


if __name__=='__main__':unittest.main(verbosity=2)
