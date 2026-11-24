---
title: 채소 종류·신선도 판정
emoji: 🥔
colorFrom: green
colorTo: yellow
sdk: gradio
sdk_version: 6.32.0
python_version: "3.11"
app_file: app.py
pinned: false
short_description: 채소 8종의 종류와 정상/비정상을 사진으로 판정
preload_from_hub:
  - laion/CLIP-ViT-B-16-DataComp.XL-s13B-b90K
---

# 채소 종류·신선도 판정 (TV-13)

사진 한 장으로 오이·감자·토마토·피망·당근·양파·고구마·대파를 알아보고, 정상인지 상했는지를 답하는 학생 프로젝트입니다.
코드와 평가 기록은 [GitHub 저장소](https://github.com/kds020902/depp-env22)에 있습니다.

이 폴더는 `python tools/build_space.py OUT`이 저장소에서 모아 만든 것입니다. 직접 고치지 말고 저장소에서 고친 뒤 다시 만드세요.
