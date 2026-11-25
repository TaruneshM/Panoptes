"""Exercise video I/O, real tracking, zones and exports with controlled inference."""
import csv
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import cv2
import numpy as np
import supervision as sv

from run import load_config, run
from shopwatch import Stay


class PipelineTest(unittest.TestCase):
    def test_video_to_evidence(self):
        # An inward crossing, dwell, one-second absence, then a second visit at EOF.
        positions = [45, 46, 48, 52, 55, 58] + [60]*22 + [None]*14 + [60]*23
        remaining = iter(positions)

        class Detector:
            def predict(self, image, threshold):
                y = next(remaining)
                if y is None:
                    return sv.Detections.empty()
                return sv.Detections(xyxy=np.array([[20, y-36, 48, y]], np.float32),
                                     class_id=np.array([1]), confidence=np.array([.99]))

        models = SimpleNamespace(RFDETRNano=Detector, RFDETRSmall=Detector, RFDETRMedium=Detector)
        reply = SimpleNamespace(message=SimpleNamespace(content='{"answer":"unsure","reason":"Hands obscured"}'))
        chat = Mock(return_value=reply)
        with TemporaryDirectory() as temp:
            path = Path(temp) / "clip.mp4"
            writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10, (96, 96))
            self.assertTrue(writer.isOpened())
            for i in range(len(positions)):
                writer.write(np.full((96, 96, 3), i*3, np.uint8))
            writer.release()
            cfg = load_config()
            cfg.update(video=str(path), stride=1, entrance_line=[[85, 50], [5, 50]],
                       zones={"shelf_1": [[5, 55], [85, 55], [85, 85], [5, 85]]})
            with patch.dict("sys.modules", rfdetr=models, ollama=SimpleNamespace(chat=chat)):
                summary, rows, folder = run(cfg, temp)
            self.assertEqual(summary["people_in"], 1)
            self.assertEqual(summary["zone_visits"], 2)
            self.assertEqual(summary["checks"]["took_item"]["unsure"], 2)
            self.assertEqual(chat.call_count, 2)
            for call in chat.call_args_list:
                self.assertEqual(len(call.kwargs["messages"][1]["images"]), 4)
            with (folder / "visits.csv").open() as f:
                saved = list(csv.DictReader(f))
            self.assertEqual(len(saved), len(rows))
            for row in saved:
                self.assertEqual(cv2.imread(str(folder / row["image"])).shape[0], 240)
            self.assertEqual(json.loads((folder / "summary.json").read_text()), summary)

    def test_long_visit_keeps_endpoints_with_bounded_samples(self):
        stay = Stay(0, 0)
        for i in range(200):
            frame = np.full((8, 8, 3), i, np.uint8)
            stay.observe(i, frame, (0, 0, 8, 8), 4)
        self.assertLess(len(stay.samples), 16)
        values = [cv2.imdecode(np.frombuffer(jpg, np.uint8), 0)[0, 0] for jpg in stay.evidence(4)]
        self.assertEqual(values, sorted(values))
        self.assertEqual((values[0], values[-1]), (0, 199))
