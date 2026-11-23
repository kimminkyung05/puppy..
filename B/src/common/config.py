"""설정 파일 읽기와 실행 로그 기록."""
import json
import os
import time
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]


def load_yaml(name: str) -> dict:
    """configs/ 아래 yaml을 읽는다. 예: load_yaml('paths')

    환경 변수 DOGMIX_PATHS가 있으면 paths 설정을 그 파일로 바꾼다
    (로컬 테스트용 경로, 사람마다 다른 Drive 경로 등).
    """
    path = ROOT / "configs" / f"{name}.yaml"
    if name == "paths" and os.environ.get("DOGMIX_PATHS"):
        path = Path(os.environ["DOGMIX_PATHS"])
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def log_run(logs_root: str, record: dict) -> None:
    """실행 한 번을 runs.jsonl에 한 줄로 남긴다 (실패한 실행도 남긴다)."""
    Path(logs_root).mkdir(parents=True, exist_ok=True)
    record = {"time": time.strftime("%Y-%m-%d %H:%M:%S"), **record}
    with open(Path(logs_root) / "runs.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
