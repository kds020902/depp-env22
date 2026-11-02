# 데이터 출처와 다시 받는 방법

모델이 본 사진은 세 곳에 있다.
1. 저장소 안(`dataset/`)
2. 작업 폴더의 공개 데이터셋 사진
3. 작업 폴더의 인터넷 사진

라이선스 때문에 2와 3은 git에 올리지 않았다. 실험 스크립트는 작업 폴더 위치를 환경변수 `DEPP_WORK`로 받는다(꼭 설정해야 한다).

## 1. 저장소 안: `dataset/` (1,860장, Train 1,260 / Valid 300 / Test 300)
- **muhriddinmuxiddinov "Fruits and Vegetables Dataset"** (Kaggle): 1,800장
- **swoyam2609 "Fresh and Stale Classification"** (Kaggle): 60장
- `dataset/manifest.csv` 행마다 원본 경로(`source_archive_path`), 주소, SHA-256, 촬영 세션(`session_id`)이 있다.
- 채소 모델(EfficientNet-B0)은 이 Train만으로 학습했다.

## 2. CLIP 분류기 추가 학습 사진 (작업 폴더, 다섯 채소 1,773장 + "다섯 채소 아님" 2,850장)

| 출처 | 쓴 곳 | 라이선스 |
|---|---|---|
| [mdatikurrahman3111 Vegetables Quality Dataset 2](https://www.kaggle.com/datasets/mdatikurrahman3111/vegetables-quality-dataset-2) | 당근 등 | CC0 1.0 |
| [vetion vege-quality](https://www.kaggle.com/datasets/vetion/vege-quality) | 피망·당근 등 | ODbL 1.0 / DbCL 1.0 |
| [sujaykapadnis Cucumber Disease Recognition](https://www.kaggle.com/datasets/sujaykapadnis/cucumber-disease-recognition-dataset) | 오이 (방글라데시 밭) | CC BY 4.0 |
| [razwansultan Tomato Disease (leaf and fruit)](https://www.kaggle.com/datasets/razwansultan/tomato-disease-dataset-both-leaf-and-fruit) | 토마토 | MIT |
| [jimpax datasetv13](https://www.kaggle.com/datasets/jimpax/datasetv13) | 밭 토마토 | CC BY 4.0 |
| [user2036 vegetable dataset without duplicacy](https://www.kaggle.com/datasets/user2036/vetable-dataset-without-duplicacy) | 상한 피망 | CC0 1.0 |
| [Mendeley CUCUMBER (zncrtr2yhn)](https://data.mendeley.com/datasets/zncrtr2yhn/1) | 오이 (검은 천 위) | CC BY 4.0 |
| [Zenodo Hybrid Potato Tuber Dataset (20616991)](https://zenodo.org/records/20616991) | 감자 | CC BY 4.0 |
| [Zenodo Cucumber Dataset (10081197)](https://zenodo.org/records/10081197) | 아시아 오이 140장 | CC BY 4.0 |
| iNaturalist, Wikimedia Commons, Flickr | 약점 사진 등 | CC 라이선스 사진만 |
| [kritikseth Fruit and Vegetable Image Recognition](https://www.kaggle.com/datasets/kritikseth/fruit-and-vegetable-image-recognition), [Fruits-262](https://www.kaggle.com/datasets/aelchimminut/fruits262) | "다섯 채소 아님": 과일·다른 채소 | 데이터셋 표기 따름 |
| Caltech-101, Places365 | "다섯 채소 아님": 일상 사진 | 연구용 공개 |
| [LVIS fruits and vegetables](https://www.kaggle.com/datasets/henningheyen/lvis-fruits-and-vegetables-dataset) | 거르기 기준 맞추기(실제 장면 채소 192장) | 데이터셋 표기 따름 |

- **Kaggle 데이터셋:** 각자 Kaggle 계정으로 받아야 한다(이 저장소는 계정 정보를 다루지 않는다). 받은 뒤 작업 폴더에 같은 이름으로 풀면 된다.
- **사진 목록:** 사진별 목록(경로, 출처, 라이선스)은 각 실험의 README와 작업 폴더 기록에 있다. 예: `added_photos.csv`, `experiments/balanced_clip_head/b1_counts.json`.

## 3. 평가 사진 (인터넷, 작업 폴더)
사진 목록(CSV)은 저장소에 있고, 아래 도구로 다시 받을 수 있다.

```
python tools/fetch_listed_photos.py results/t3_test/t3_test.csv  $DEPP_WORK/t3/img
python tools/fetch_listed_photos.py results/t4_test/t4_test.csv  $DEPP_WORK/t4/img
python tools/fetch_listed_photos.py results/10class/outside_test2/outside_test2.csv  $DEPP_WORK/weak/t2img
```

- iNaturalist 사진은 원본 그대로 받아진다. 확인해 보니 작업 폴더 사진과 바이트까지 같았다.
- Commons 사진은 폭 800px로 받는다.
- 원본이 지워진 사진은 "missing"으로 알려 준다.

## 학습 사진이 어디서 왔나 (편향)
- **채소 모델:** 거의 한 데이터셋(muhriddinmuxiddinov, 97%)으로 학습했다. 이 데이터셋은 깨끗한 배경의 사진이 많다.
- **CLIP 분류기 다섯 채소 3,033장:**
    - 데이터셋 사진이 약 96%다. 대부분 한 곳에서 찍은 묶음이다.
    - 사람들이 실제로 찍어 올린 인터넷 사진(iNaturalist·Commons·Flickr)은 약 130장, 약 4%다(작업 폴더 기록으로 센 대략값).
- **그래서** 학습과 같은 출처의 Test에서는 점수가 높고, 처음 보는 인터넷 사진에서는 낮다([성적표](results/scorecard/README.md)).
