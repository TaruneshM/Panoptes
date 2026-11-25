"""Run a calibrated shop-camera analysis from the CLI or Gradio."""
import argparse
import csv
import json
import math
from pathlib import Path
from tempfile import TemporaryDirectory
from uuid import uuid4

import cv2
import numpy as np
import yaml

from check import ANSWERS, ask
from shopwatch import find_visits

CONFIG = Path(__file__).with_name("config.yaml")
FIELDS = ["visit", "zone", "person", "at_sec", "secs", "check", "answer", "reason", "image"]


def load_config(path=CONFIG):
    path = Path(path)
    cfg = yaml.safe_load(path.read_text())
    if not isinstance(cfg, dict):
        raise ValueError("Expected a YAML configuration mapping")
    required = "video detector threshold stride min_visit_sec frames_per_visit vlm_model entrance_line zones checks".split()
    missing = set(required) - cfg.keys()
    if missing:
        raise ValueError(f"Missing settings: {', '.join(sorted(missing))}")
    if cfg["detector"] not in ("nano", "small", "medium"):
        raise ValueError("Choose nano, small or medium for detector")
    for name in ("stride", "frames_per_visit"):
        if type(cfg[name]) is not int or cfg[name] < 1:
            raise ValueError(f"{name} must be a positive integer")
    if not 0 <= cfg["threshold"] <= 1 or not math.isfinite(cfg["min_visit_sec"]) or cfg["min_visit_sec"] < 0:
        raise ValueError("Check threshold (0–1) and min_visit_sec (nonnegative seconds)")
    if not isinstance(cfg["zones"], dict) or not cfg["zones"] or not isinstance(cfg["checks"], list):
        raise ValueError("Define at least one zone and a list of checks")
    names = set()
    for check in cfg["checks"]:
        if not isinstance(check, dict) or any(not isinstance(check.get(k), str) or not check[k].strip()
                                               for k in ("name", "zone", "question")):
            raise ValueError("Each check needs a name, zone and question")
        if check["zone"] not in cfg["zones"] or check["name"] in names:
            raise ValueError("Each check needs a distinct name and a configured zone")
        names.add(check["name"])
    cfg["video"] = str(path.parent / cfg["video"])
    return cfg


def evidence_strip(frames):
    tiles = []
    for jpg in frames:
        crop = cv2.imdecode(np.frombuffer(jpg, np.uint8), cv2.IMREAD_COLOR)
        width = max(1, round(240 * crop.shape[1] / crop.shape[0]))
        tiles.append(cv2.resize(crop, (width, 240)))
    return cv2.hconcat(tiles)


def run(cfg, out="output"):
    people, visits = find_visits(cfg)
    root = Path(out).resolve()
    root.mkdir(exist_ok=True, parents=True)
    destination = root / f"run-{uuid4().hex[:12]}"
    rows = []
    counts = {c["name"]: dict.fromkeys(ANSWERS, 0) for c in cfg["checks"]}
    with TemporaryDirectory(dir=root) as temp:
        stage = Path(temp)
        for number, visit in enumerate(visits, 1):
            image = f"visit_{number:03}.jpg"
            if not cv2.imwrite(str(stage / image), evidence_strip(visit["frames"])):
                raise OSError(f"Could not write {image}")
            base = {k: visit[k] for k in ("zone", "person", "at_sec", "secs")}
            base.update(visit=number, image=image)
            matched = [c for c in cfg["checks"] if c["zone"] == visit["zone"]]
            if not matched:
                rows.append(base)
            for check in matched:
                answer, reason = ask(cfg["vlm_model"], check["question"], visit["frames"])
                counts[check["name"]][answer] += 1
                rows.append(dict(base, check=check["name"], answer=answer, reason=reason))
        took = counts.get("took_item", {})
        total = sum(took.values())
        summary = dict(people_in=int(people), zone_visits=len(visits), checks=counts,
                       take_rate_pct=round(100*took["yes"]/total, 1) if total else None)
        with (stage / "visits.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=FIELDS)
            writer.writeheader()
            writer.writerows(rows)
        (stage / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        stage.rename(destination)
    return summary, rows, destination


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", nargs="?", help="MP4 file (overrides config.yaml)")
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--out", default="output")
    args = parser.parse_args()
    try:
        config = load_config(args.config)
        if args.video:
            config["video"] = args.video
        summary, _, folder = run(config, args.out)
    except (ValueError, RuntimeError, OSError, yaml.YAMLError) as exc:
        parser.exit(1, f"Analysis failed: {exc}\n")
    print(json.dumps(summary, indent=2))
    print(f"Results: {folder}")
