"""Defect 战斗课程训练入口：python -I train_defect.py [--参数 值 ...]，参数见 sts2rl/config.py。"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from sts2rl.train import main  # noqa: E402

if __name__ == "__main__":
    main()
