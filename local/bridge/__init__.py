"""ゲーム MOD とつなぐブリッジ。

    protocol   1 行 1 JSON のメッセージ定義（版・検証）
    autopilot  接続 1 本ぶんの状態機械（停止・一時停止・速度・重複防止・ログ）
    agent      観測フレーム → 推論（local.infer.Policy を共有）
    server     TCP の受け口と、標準入力からの操作
    sim        実フレームで通し試験するためのシミュレータ（ゲーム不要）

ゲーム側（Windows の MOD）は `game-mod/` にあり、この同じプロトコルで話す。
"""

from . import protocol
from .autopilot import Autopilot, AutopilotError, Decision, SessionLog, SessionStats

__all__ = ["Autopilot", "AutopilotError", "Decision", "SessionLog", "SessionStats", "protocol"]
