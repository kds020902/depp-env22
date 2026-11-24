"""The web page of the adopted system (Hugging Face Spaces, Gradio): one photo in, the answer predict.py would give out.

    python app.py            # http://127.0.0.1:7860

The page adds nothing to the decision: a photo is opened with images.open_image() and answered by predict.Classifier, the
same code path as `python src/predict.py`. tools/build_space.py collects the files a Space needs.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "src"))

from decision import HOLD_BACK_BELOW  # noqa: E402
from images import open_image  # noqa: E402
from labels import KOREAN, SPECIES_NAMES, UNKNOWN_MESSAGE, condition_of  # noqa: E402

MODEL = HERE / "model.pth"
MAX_PIXELS = 40_000_000          # a phone photo is 12-50 MP; larger files are refused before they are decoded in full
MAX_FILE = "20mb"
TITLE = "채소 종류 · 신선도 판정"
INTRO = ("사진 한 장을 올리면 **오이 · 감자 · 토마토 · 피망 · 당근 · 양파 · 고구마 · 대파** 가운데 무엇인지와 "
         "정상인지 상했는지를 알려 줍니다. 채소 하나만, 밝은 곳에서 가까이 찍은 사진이 가장 잘 맞습니다.")
NOTES = f"""
**읽는 법**
- 확률이 {100 * HOLD_BACK_BELOW:g}% 미만이면 답을 확정하지 않고 **판단 보류**로 가장 가까운 답만 알려 줍니다.
- **{UNKNOWN_MESSAGE}** 는 위 여덟 채소의 사진이 아닌 것 같다는 뜻입니다.
- 대파는 종류만 알아보고 상태는 판정하지 않습니다.
- 빨간 부분은 모델이 상했다고 본 곳입니다(7×7칸 단위의 대략적인 위치). 오이 · 감자 · 토마토 · 피망 · 당근에만 나옵니다.

**한계**
- 인터넷 사진으로 학습한 학생 프로젝트입니다. 직접 찍은 사진에서는 정확도가 다를 수 있습니다.
- 썩은 고구마를 정상으로, 흙 묻은 정상 감자를 상했다고 답하는 경우가 있습니다.
- 먹어도 되는지를 판단해 주지 않습니다. 상태는 직접 확인하세요.
- 올린 사진은 판정에만 씁니다. 서버의 임시 파일은 약 10분 뒤 지웁니다.
"""


def class_label(index: int) -> str:
    return f"{KOREAN[SPECIES_NAMES[index // 2]]} {KOREAN[condition_of(index)]}"


def headline(r: dict) -> str:
    """The answer a visitor reads first, from predict.Classifier.classify()'s result."""
    if r["unknown"]:
        return f"## {UNKNOWN_MESSAGE}\n{r['message'][len(UNKNOWN_MESSAGE):].strip(' ()')}"
    if r["held"]:
        title, _, advice = r["message"].partition(": ")
        return f"## {title}\n{advice}"
    return f"## {class_label(r['index'])}\n확률 {100 * r['probability']:.0f}%"


def shown_probabilities(r: dict, top: int = 5) -> dict[str, float]:
    """Class probabilities for the bar list; none for a photo that was turned away. Green onion has one bar (kind only)."""
    if r["unknown"]:
        return {}
    shares: dict[str, float] = {}
    for index, p in enumerate(r["probabilities"].tolist()):
        shares[class_label(index)] = shares.get(class_label(index), 0.0) + p
    return dict(sorted(shares.items(), key=lambda item: -item[1])[:top])


def check_size(path: str | Path) -> None:
    from PIL import Image
    with Image.open(path) as image:      # reads the header only
        if image.width * image.height > MAX_PIXELS:
            raise ValueError(f"사진이 너무 큽니다({image.width}×{image.height}). 더 작은 사진을 올려 주세요.")


def answer(system, path: str | Path | None, hold_back_below: float = HOLD_BACK_BELOW):
    """(headline, probabilities, heat map or None) for one uploaded file."""
    if not path:
        return "사진을 올려 주세요.", {}, None
    try:
        check_size(path)
        photo = open_image(path)
        r = system.classify(photo, hold_back_below)
    except (OSError, ValueError, RuntimeError) as error:
        detail = str(error) if isinstance(error, ValueError) and "사진이 너무 큽니다" in str(error) else "사진 파일을 읽지 못했습니다. JPG나 PNG 사진을 올려 주세요."
        return f"## 판정하지 못했습니다\n{detail}", {}, None
    heat = None
    if r["rot_map"] is not None:
        from networks import rot_map_overlay
        heat = rot_map_overlay(photo, r["rot_map"], r["scale"], side_by_side=False)
    return headline(r), shown_probabilities(r), heat


def build(system):
    import gradio as gr
    with gr.Blocks(title=TITLE, delete_cache=(600, 600)) as page:   # uploads are temporary files: removed after 10 minutes
        gr.Markdown(f"# {TITLE}\n{INTRO}")
        with gr.Row():
            with gr.Column():
                photo = gr.Image(type="filepath", label="사진", sources=["upload", "clipboard", "webcam"], height=360)
                button = gr.Button("판정하기", variant="primary")
            with gr.Column():
                verdict = gr.Markdown("사진을 올려 주세요.")
                bars = gr.Label(label="가까운 답", num_top_classes=5)
                heat = gr.Image(label="상한 곳 보기", height=300, interactive=False)
        gr.Markdown(NOTES)
        run = lambda path: answer(system, path)  # noqa: E731
        button.click(run, photo, [verdict, bars, heat])
        photo.upload(run, photo, [verdict, bars, heat])
    return page


def main():
    import torch
    from predict import Classifier
    torch.set_num_threads(int(os.environ.get("THREADS", "2")))   # the free Space has 2 CPU cores
    system = Classifier(MODEL, torch.device("cpu"))
    build(system).queue(max_size=20, default_concurrency_limit=1).launch(max_file_size=MAX_FILE)


if __name__ == "__main__":
    main()
