"""pytest 的共享配置：修正 import 路径。

【为什么需要这个文件？】
pytest 运行 tests/ 下的测试时，只会自动把 tests/ 目录加进 sys.path。
但我们的测试要 import 两个不在标准位置的东西：
  - src/ 下的模块（在项目根目录）
  - scripts/prepare_data.py（在 scripts/ 目录）
conftest.py 是 pytest 的"前置脚本"：跑任何测试之前先执行它，
在这里把两个目录加进 sys.path，测试文件里就能正常 import 了。
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent   # 项目根目录（tests/ 的上一级）
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
