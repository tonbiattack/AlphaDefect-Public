"""房间结果记录（2026-10-27 起，决策记录 146）：给“房间结果模型”（监督学习，预测一场战斗掉多少血、会不会死，辅助选路搜索）攒数据。

采集进程按 cfg.room_log 的比例抽样，记两类战斗，每场一行 JSON：
- run：整局里真打的每场战斗（普通、精英、Boss）。开打前最近一次选路线时的玩家快照、遭遇、房间类型、幕、层，
  开打与结束时的血量、最大生命、手上药水瓶数，输赢；
- practice：战斗课程里用整局真实快照重打的战斗（Boss 名额、各幕普通战与精英战、中途快照），同样的字段，另记课程名额（slot）
  与这场用了几瓶药水。随机构筑、估计牌组的战斗没有真实快照，不记。
快照是游戏的 SerializablePlayer（牌组、遗物、药水、血量、金币等，约 3.6 KB），原样存，模型的输入以后再定。

文件：run_dir/room_log/r<采集进程>_<年月日时>.jsonl.gz，每小时换一个文件，约 30 秒落一次盘；进程退出时没落盘的几十行丢了也无妨。
"""
from __future__ import annotations

import gzip
import json
import os
import time


class RoomLog:
    def __init__(self, run_dir: str, rank: int, rate: float, rng):
        self.rate, self.rng, self.rank = rate, rng, rank
        self.dir = os.path.join(run_dir, "room_log")
        self.hour, self.f, self.last_flush, self.n = None, None, time.time(), 0
        if rate > 0:
            os.makedirs(self.dir, exist_ok=True)

    def sample(self) -> bool:
        return self.rate > 0 and (self.rate >= 1 or self.rng.random() < self.rate)

    def write(self, rec: dict) -> None:
        try:
            hour = time.strftime("%Y%m%d%H")
            if hour != self.hour:
                if self.f is not None:
                    self.f.close()
                self.f = gzip.open(os.path.join(self.dir, f"r{self.rank}_{hour}.jsonl.gz"), "at", encoding="utf-8")
                self.hour = hour
            rec["t"] = round(time.time(), 1)
            self.f.write(json.dumps(rec, ensure_ascii=False, separators=(",", ":")) + "\n")
            self.n += 1
            if time.time() - self.last_flush > 30:
                self.f.flush(); self.last_flush = time.time()
        except Exception as e:      # 记不下来不影响采集
            print(f"采集进程 {self.rank} 房间结果记录写入失败：{e}", flush=True)
            self.rate = 0.0
