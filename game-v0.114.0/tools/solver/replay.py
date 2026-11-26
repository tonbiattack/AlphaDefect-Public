"""模型（贪心）打重建出来的同一场 Boss 战：python replay.py 代码目录 模型.pt 配置.jsonl"""
import json, sys
import numpy as np, torch
sys.path.insert(0, sys.argv[1])
from sts2env import Sts2Env
from sts2rl.actor import MODE_COMBAT
from sts2rl.features import Ragged, collate_ragged
from sts2rl.model import ModelConfig, PolicyValueNet, load_state
torch.set_num_threads(4)
net = PolicyValueNet(ModelConfig(d=256, layers=4)).eval()
load_state(net, torch.load(sys.argv[2], map_location="cpu", weights_only=False)["state"])
with Sts2Env() as env:
    for k, line in enumerate(open(sys.argv[3])):
        x = json.loads(line)
        o = env.reset_combat(x["spec"]); hp0, n, turn = o.hp, 0, 0
        while not (o.done or o.num_actions == 0) and n < 600:
            with torch.no_grad():
                lg, _ = net(collate_ragged(Ragged.from_obs([o], np.full(1, MODE_COMBAT))).to("cpu"))
            o = env.step(int(lg.argmax(-1))); n += 1
        print(json.dumps({"i": k, "act": x["label"]["act"], "encounter": x["label"]["encounter"], "start_hp": hp0, "model_win": bool(o.victory),
                          "model_end_hp": int(o.hp), "model_steps": n}), flush=True)
