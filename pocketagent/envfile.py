"""极简 .env 读取器（避免额外第三方依赖）。

规则：
- 每行 KEY=VALUE，支持 # 注释与行尾注释、export 前缀
- 不覆盖已存在的环境变量（保证命令行/进程环境优先级更高）
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Union


def load_dotenv(path: Optional[Union[str, os.PathLike]] = None) -> bool:
    """加载 .env 到 os.environ（不覆盖已有变量）。返回是否加载了内容。"""
    candidates: list[Path] = []
    if path is not None:
        candidates.append(Path(path))
    else:
        # 依次尝试：当前目录 与 仓库根目录（向上查找第一个含 .env 的父目录）
        if (Path.cwd() / ".env").is_file():
            candidates.append(Path.cwd() / ".env")
        else:
            here = Path(__file__).resolve()
            for parent in here.parents:
                if (parent / ".env").is_file():
                    candidates.append(parent / ".env")
                    break

    loaded = False
    for p in candidates:
        if not p.exists():
            continue
        for raw in p.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("export "):
                line = line[len("export ") :].strip()
            if "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip()
            # 去掉引号与行尾注释
            if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
                value = value[1:-1]
            if " #" in value:
                value = value.split(" #", 1)[0].strip()
            if key and key not in os.environ:
                os.environ[key] = value
                loaded = True
    return loaded
