"""从主线配置生成专项配置：python3 mkcfg.py 主线配置.json 输出.json '{"改动": ...}'"""
import json, sys
c = json.load(open(sys.argv[1]))
base = {
    # 只打第三幕双 Boss 连打（决策记录 287）
    "run_fraction": 0.0, "chain_problems": "/root/b3s/problems.jsonl", "boss_win_bonus": 0.0, "boss_dmg_coef": 0.0,
    # 不连主线的桥、不做主线的评测与诊断
    "bridge_port": 0, "bridge_bind": "", "bridge_token_file": "", "eval_runs": 0, "eval_every_s": 0.0, "eval_combats": 0, "value_probe": False,
    "eval_replay_all": False, "solver_share": 0.0, "probe_share": 0.0,
    # 与版本号挂钩的主线日程：专项从第 0 版起，全部取满额或关闭
    "hl_start": 0, "hl_coef_from": None, "milestone_from": None, "bv_start": None, "bv_shaping": 0.0, "wp_shaping": 0.0,
    "probe_split": None, "probe_base": None, "combat_fight_boot": False, "explore_probs": {}, "combat_explore": {},
    # 专项日程
    "combat_anchor_coef": 0.5, "combat_anchor_ckpt": "/root/diag/cyc1/student_e.pt", "combat_anchor_rows": "combat",
}
c.update(base)
c.update(json.loads(sys.argv[3]) if len(sys.argv) > 3 else {})
json.dump(c, open(sys.argv[2], "w"), ensure_ascii=False, indent=1)
print({k: c[k] for k in ("lr", "actors", "envs_per_actor", "batch_steps", "combat_anchor_until", "value_warmup_until", "weight_decay", "lr_decay_until", "lr_final_frac") if k in c})
