"""Track people and collect chronological evidence from shelf visits."""
from dataclasses import dataclass, field
from pathlib import Path
import math

import cv2
import numpy as np


@dataclass
class Stay:
    first: int
    last: int
    samples: list = field(default_factory=list)
    interval: int = 1
    latest: bytes = b""

    def observe(self, index, frame, box, count):
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = box
        dx, dy = (x2 - x1) / 4, (y2 - y1) / 4
        crop = frame[max(0, int(y1-dy)):min(h, math.ceil(y2+dy)),
                     max(0, int(x1-dx)):min(w, math.ceil(x2+dx))]
        if not crop.size:
            return
        scale = min(1, 640 / max(crop.shape[:2]))
        if scale < 1:
            crop = cv2.resize(crop, (max(1, round(crop.shape[1]*scale)),
                                     max(1, round(crop.shape[0]*scale))), interpolation=cv2.INTER_AREA)
        ok, jpg = cv2.imencode(".jpg", crop)
        if not ok:
            raise ValueError("Could not encode evidence frame")
        self.latest = jpg.tobytes()
        self.last = index
        if not self.samples or index - self.samples[-1][0] >= self.interval:
            self.samples.append((index, self.latest))
            if len(self.samples) >= 4 * count:
                self.samples = self.samples[::2]
                self.interval *= 2

    def evidence(self, count):
        candidates = dict(self.samples)
        candidates[self.last] = self.latest
        indices = sorted(candidates)
        if count == 1:
            return [self.latest]
        picks = np.linspace(0, len(indices)-1, min(count, len(indices))).round().astype(int)
        return [candidates[indices[i]] for i in picks]


def inspect_video(path):
    if not Path(path).is_file():
        raise ValueError(f"Video file does not exist: {path}")
    capture = cv2.VideoCapture(str(path))
    try:
        ok, frame = capture.read()
        fps = capture.get(cv2.CAP_PROP_FPS)
        if not ok or not math.isfinite(fps) or fps <= 0:
            raise ValueError("Could not read the clip. Use a decodable, fixed-frame-rate MP4.")
        return frame, fps
    finally:
        capture.release()


def geometry(cfg, frame):
    h, w = frame.shape[:2]
    line = np.asarray(cfg["entrance_line"])
    zones = {name: np.asarray(points) for name, points in cfg["zones"].items()}
    if line.shape != (2, 2) or np.array_equal(*line):
        raise ValueError("entrance_line needs two distinct points")
    for name, points in [("entrance_line", line), *zones.items()]:
        if points.ndim != 2 or points.shape[1] != 2 or points.dtype.kind not in "iu":
            raise ValueError(f"{name} needs integer [x, y] points")
        if np.any(points < 0) or np.any(points >= [w, h]):
            raise ValueError(f"Calibrate {name} for this {w} × {h} frame in config.yaml")
    for name, points in zones.items():
        if len(points) < 3 or not cv2.contourArea(points.astype(np.int32)):
            raise ValueError(f"{name} needs a polygon enclosing a floor area")
    return line, zones


def find_visits(cfg):
    first_frame, fps = inspect_video(cfg["video"])
    endpoints, polygons = geometry(cfg, first_frame)
    import supervision as sv
    from trackers import ByteTrackTracker
    from rfdetr import RFDETRNano, RFDETRSmall, RFDETRMedium

    model = {"nano": RFDETRNano, "small": RFDETRSmall, "medium": RFDETRMedium}[cfg["detector"]]()
    stride = cfg["stride"]
    tracker = ByteTrackTracker(frame_rate=fps / stride)
    anchor = [sv.Position.BOTTOM_CENTER]
    line = sv.LineZone(start=sv.Point(*endpoints[0]), end=sv.Point(*endpoints[1]), triggering_anchors=anchor)
    zones = {name: sv.PolygonZone(polygon=poly, triggering_anchors=anchor) for name, poly in polygons.items()}
    live, visits = {}, []
    count = cfg["frames_per_visit"]

    def close(key, end):
        stay = live.pop(key)
        duration = (min(end, stay.last + stride) - stay.first) / fps
        if stay.latest and duration >= cfg["min_visit_sec"]:
            visits.append(dict(zone=key[0], person=key[1], at_sec=round(stay.first/fps, 2),
                               secs=round(duration, 2), frames=stay.evidence(count)))

    capture = cv2.VideoCapture(str(cfg["video"]))
    index = 0
    try:
        while capture.grab():
            if index % stride == 0:
                ok, frame = capture.retrieve()
                if not ok:
                    raise ValueError(f"Could not decode frame {index}")
                detections = model.predict(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), threshold=cfg["threshold"])
                people = tracker.update(detections[detections.class_id == 1])
                if people.tracker_id is not None:
                    people = people[people.tracker_id >= 0]
                line.trigger(people)
                seen = set()
                for name, zone in zones.items():
                    inside = people[zone.trigger(people)]
                    for box, person in zip(inside.xyxy, inside.tracker_id if len(inside) else ()):
                        key = (name, int(person))
                        if key in live and index - live[key].last - stride >= fps:
                            close(key, index)
                        if key not in live:
                            live[key] = Stay(index, index, interval=max(stride, round(fps/2)))
                        live[key].observe(index, frame, box, count)
                        seen.add(key)
                for key in list(live.keys() - seen):
                    if index - live[key].last >= fps:
                        close(key, index)
            index += 1
    finally:
        capture.release()
    for key in list(live):
        close(key, index)
    return line.in_count, sorted(visits, key=lambda v: (v["at_sec"], v["zone"], v["person"]))
