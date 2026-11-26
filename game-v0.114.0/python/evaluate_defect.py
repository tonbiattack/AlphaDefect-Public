"""评测入口：python -I evaluate_defect.py --checkpoint ... 或 --random，参数见 sts2rl/evaluate.py。"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from sts2rl.evaluate import main  # noqa: E402

if __name__ == "__main__":
    main()
