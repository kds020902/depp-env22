# 추가 실험: 밭 토마토 사진 추가 학습

결과: **채택하지 않았다.** 어려운 사례 세트 점수는 올랐지만 기존 Test가 1장 떨어져서, 학습 전에 정한 채택 기준을 넘지 못했다. 최종 모델(`model.pth`)은 그대로다.

## 왜 했나

꼭지 둘레에만 곰팡이가 핀 토마토 사진(몸통은 멀쩡함)을 최종 모델이 "토마토 정상 0.86"으로 틀렸다. Grad-CAM으로 보니 모델은 곰팡이가 아니라 빨간 몸통을 보고 판단했다. 이렇게 일부만 상한 토마토를 더 학습시키려고 데이터를 추가했다.

## 추가한 데이터

- **출처:** [Tomato Maturity Detection and Quality Grading Dataset](https://data.mendeley.com/datasets/s42kpg8h37/1) (Mendeley Data, DOI 10.17632/s42kpg8h37.1, CC BY 4.0)
    - 방글라데시 농장 토마토를 나무 판 위에 놓고 휴대폰으로 찍은 원본 사진이다(Fresh 1,350장, Rotten 636장).
    - 같은 데이터셋의 "Augment" 폴더(회전·반전 복사본)는 쓰지 않았다.
- **Train에 추가한 사진:** 2023년 3~4월 촬영분에서 눈으로 검수한 비정상 60장 + 정상 60장 (`added_train_photos.csv`)
    - 정상과 비정상이 같은 나무 배경이라, 배경이 정답 단서가 되지 않는다.
- **어려운 사례 세트:** 2023년 1~2월 촬영분에서 비정상 30장 + 정상 30장 (`hard_case_set.csv`)
    - 학습에는 넣지 않았다. 촬영일이 두 달 떨어져 있어 Train과 다른 토마토다.
- **겹침 확인:** 기존 데이터셋과 보내 받은 사진의 복사본은 없었다(ResNet18 특징 유사도 최대 0.916).
- **기존 분할은 그대로:** Valid 180장과 Test 180장은 바꾸지 않았다. Train만 840장 → 960장이 됐다(정상 480 / 비정상 480).

## 검색했지만 쓰지 않은 데이터

| 데이터셋 | 쓰지 않은 이유 |
|---|---|
| [Fruit and Vegetable Disease (Healthy vs Rotten)](https://www.kaggle.com/datasets/muhammad0subhan/fruit-and-vegetable-disease-healthy-vs-rotten) | 토마토 1,200장 전부가 이미 쓰고 있는 Kaggle 데이터의 복사본이었다 |
| [Fresh and Rotten Classification](https://www.kaggle.com/datasets/swoyam2609/fresh-and-stale-classification)의 토마토 | 정상과 비정상이 서로 다른 장식 틀에 들어 있어, 틀 모양이 정답 단서가 된다 |

## 결과

학습 설정은 최종 모델과 같다(EfficientNet-B0 미세조정, 같은 seed와 하이퍼파라미터).

| | 최종 모델 (채택 유지) | 추가 학습 모델 |
|---|---:|---:|
| Valid 6클래스 | 92.2% (166/180) | 93.3% (168/180) |
| **Test 6클래스** | **95.0% (171/180)** | 94.4% (170/180) |
| Test 정상/비정상 | 96.7% | 96.7% |
| 어려운 사례: 비정상을 비정상으로 | 27/30 | 30/30 |
| 어려운 사례: 정상을 정상으로 | 27/30 | 27/30 |
| 어려운 사례 6클래스 | 54/60 | 57/60 |
| 보내 받은 꼭지 곰팡이 토마토 | 정상 0.86 (틀림) | 정상 0.86 (틀림) |

- **채택 기준:** 학습 전에 "어려운 사례 점수가 오르고, 기존 Test가 떨어지지 않을 것"으로 정했다. Test가 1장 떨어져서 채택하지 않았다.
- **차이의 크기:** 1장 차이는 측정 오차 범위다. 두 모델은 Test에서 실질적으로 같다.
- **어려운 사례 세트의 한계:** 이 세트의 비정상 토마토는 대부분 전체가 쭈글쭈글해서, 최종 모델도 이미 27/30을 맞혔다.
- **문제 사진은 그대로:** 꼭지 둘레만 상한 토마토는 두 모델 모두 틀렸다. 이 유형은 추가한 데이터에도 거의 없었다.
- **다음에 할 일:** "몸통은 멀쩡하고 일부만 상한" 사진을 직접 찍거나 따로 모아야 한다.

## 다시 실행하기

`added_train_photos.csv`의 120장을 Mendeley에서 받은 원본 경로로 `dataset/manifest.csv`의 Train에 더해 `dataset/manifest_plus_field.csv`로 저장한 뒤, 같은 명령으로 학습한다. 기존 행의 `image_path`가 `dataset/` 기준이라 이 파일도 `dataset/`에 둔다.

```bash
python src/train.py --manifest dataset/manifest_plus_field.csv --out results/effb0_plus_field --arch efficientnet_b0 --mode finetune --epochs 30 --patience 8
```

이 폴더의 `metrics.json`, `history.csv`, `*_predictions.csv`가 그 실행 결과다. 가중치는 채택하지 않아서 올리지 않았다.
