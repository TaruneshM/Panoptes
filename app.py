"""Local interface for uploading a clip and reviewing visit evidence."""
import argparse

import cv2
import gradio as gr
import yaml

from run import FIELDS, load_config, run
from shopwatch import geometry, inspect_video


def preview(path):
    if not path:
        return None, ""
    try:
        frame, _ = inspect_video(path)
    except (ValueError, OSError, yaml.YAMLError) as exc:
        return None, str(exc)
    try:
        line, zones = geometry(load_config(), frame)
    except (ValueError, OSError, yaml.YAMLError) as exc:
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), str(exc)
    cv2.arrowedLine(frame, tuple(line[0]), tuple(line[1]), (255, 190, 45), 3)
    for name, poly in zones.items():
        cv2.polylines(frame, [poly.astype("int32")], True, (100, 225, 70), 3)
        cv2.putText(frame, name, tuple(poly[0]), cv2.FONT_HERSHEY_SIMPLEX, .6, (100, 225, 70), 2)
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), "Check the entrance line and shelf floor area before analysis."


def coordinates(evt: gr.SelectData):
    x, y = evt.index
    return f"Selected pixel [{x}, {y}]. Copy your entrance endpoints and shelf corners into config.yaml."


def analyse(path):
    if not path:
        raise gr.Error("Upload a shop video first.")
    try:
        cfg = load_config()
        cfg["video"] = path
        summary, rows, folder = run(cfg)
    except (ValueError, RuntimeError, OSError, yaml.YAMLError) as exc:
        raise gr.Error(str(exc)) from exc
    captions = {}
    for row in rows:
        caption = captions.setdefault(row["image"], [f"Visit {row['visit']} · {row['zone']}"])
        if row.get("check"):
            caption.append(f"{row['check']}: {row['answer']}")
    gallery = [(str(folder / name), " · ".join(labels)) for name, labels in captions.items()]
    return summary, [[r.get(k, "") for k in FIELDS[:-1]] for r in rows], gallery, str(folder / "visits.csv")


def build_app():
    with gr.Blocks(title="EdgeEye · Shelf visits", analytics_enabled=False) as page:
        gr.Markdown("# EdgeEye\nUpload a fixed-camera shop clip to count entrances and review shelf visits.")
        with gr.Row():
            video = gr.File(label="Shop clip", file_types=[".mp4"], type="filepath")
            overlay = gr.Image(label="Camera zones", interactive=False)
        status = gr.Textbox(label="Calibration", interactive=False, value="Set the camera coordinates in config.yaml.")
        video.change(preview, video, [overlay, status])
        overlay.select(coordinates, outputs=status)
        start = gr.Button("Analyze clip", variant="primary")
        summary = gr.JSON(label="Visit summary")
        table = gr.Dataframe(headers=FIELDS[:-1], label="Visit checks", interactive=False)
        gallery = gr.Gallery(label="Visit evidence · earliest to latest", columns=2)
        csv_file = gr.File(label="Download visit log")
        start.click(analyse, video, [summary, table, gallery, csv_file], concurrency_limit=1)
    return page


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--share", action="store_true", help="Create a public Gradio link (e.g. for Colab)")
    build_app().launch(share=parser.parse_args().share)
