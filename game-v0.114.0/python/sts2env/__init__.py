"""杀戮尖塔2 无头引擎的 Python 接口。

Sts2Env：单个引擎进程；VecEnv：多个进程并行。观测格式见仓库根目录的 观测格式.md。
"""
from .protocol import Obs, DecisionKind, ActionKind, TokenKind, Zone
from .env import Sts2Env, VecEnv, EngineError

__all__ = ["Obs", "DecisionKind", "ActionKind", "TokenKind", "Zone", "Sts2Env", "VecEnv", "EngineError"]
