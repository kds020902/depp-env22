# 학습 데이터 크게 늘리기 (채택 안 함)

모델(EfficientNet-B0, 설정 그대로)은 두고, 학습 사진을 1,260장에서 2,146장으로 늘려 봤다. 미리 정한 1·2단계 기준(인터넷 사진, Valid)은 넘었다. 하지만 마지막에 한 번만 채점한 새 시험지 160장과 Test에서 현재 모델보다 낮게 나와 채택하지 않았다. `model.pth`와 `dataset/`은 그대로다.

## 추가한 사진 (Train에만, 886장)

| 채소 | 출처 (라이선스) | 정상 | 상함 |
|---|---|---:|---:|
| 오이 | [Cucumber Disease Recognition](https://www.kaggle.com/datasets/sujaykapadnis/cucumber-disease-recognition-dataset) (CC BY 4.0), 밭 사진 | 60 | 60 |
| 감자 | [vegetables-quality-dataset-2](https://www.kaggle.com/datasets/mdatikurrahman3111/vegetables-quality-dataset-2) (CC0), 휴대폰 사진 | 60 | 60 |
| 토마토 | 같은 데이터 | 60 | 60 |
| 토마토 | [Tomato Disease Dataset](https://www.kaggle.com/datasets/razwansultan/tomato-disease-dataset-both-leaf-and-fruit) (MIT), 밭·웹 사진. 상함은 곰팡이·탄저병·역병·배꼽썩음병 | 50 | 50 |
| 토마토 | [Hybrid Tomato Disease Dataset](https://www.kaggle.com/datasets/jimpax/datasetv13) (CC BY 4.0), 밭 사진. 상함은 배꼽썩음병 | 40 | 40 |
| 당근 | vegetables-quality-dataset-2 | 60 | 60 |
| 당근 | [vege-quality](https://www.kaggle.com/datasets/vetion/vege-quality) (ODbL), 웹 사진 | 40 | 40 |
| 피망 | vege-quality | 73 | 34 |
| 피망 | [vegetable_dataset_without_duplicacy](https://www.kaggle.com/datasets/user2036/vetable-dataset-without-duplicacy) (CC0), 웹 사진 | – | 39 |

- 모두 눈으로 검수했다. 오이·감자·토마토·당근 휴대폰 사진은 [rotten_phone](../rotten_phone/README.md), 피망은 [pepper_web](../pepper_web/README.md)에서 검수했다.
- 정상과 상함은 같은 출처에서 비슷한 수로 넣었다.
- **같은 사진 제외:** 3차 데이터 1,860장, 인터넷 사진 221장, 새 시험지 160장과 같은 사진(해시가 거의 같거나 반전·회전 불변 유사도 0.95 이상)을 마지막에 다시 검사했다. 걸린 사진은 0장이었다.
- 쓴 사진 목록은 `added_photos.csv`에 있다.
- Train은 채소마다 정상·상함 179~290장이 됐다.

## 평가 사진

| 이름 | 사진 | 쓰임 |
|---|---|---|
| Valid | 300장 | epoch 고르기, 점수가 떨어지지 않는지 확인 |
| 인터넷 사진 | 221장 ([results/10class/outside_test](../../results/10class/outside_test/README.md)) | 채택 1단계 판단 |
| **새 시험지** | 160장 (`outside_test_160.csv`) | 마지막에 한 번만 채점 |
| Test | 300장 | 마지막에 한 번만 채점 |

**새 시험지(160장)는 학습에 한 번도 쓰지 않은 데이터셋 6곳에서 골랐다.** 학습 전에 정답을 정해 고정했다.

| 데이터셋 | 라이선스 | 고른 사진 |
|---|---|---|
| AgriFreshNET | CC BY 4.0 | 오이·토마토, 천 위 휴대폰 사진 |
| VegNet | CC BY 4.0 | 피망 |
| Mendeley "Fresh CARROT" | CC BY 4.0 | 정상 당근 |
| TriModal Ripeness 6 | CC BY 4.0 | 상한 당근 |
| VegQual | CC BY 4.0 | 감자, 사진에서 잘라냄 |
| Potato Disease Recognition | CC BY 4.0 | 상한 감자 |

## 미리 정한 규칙 (`PREREG.md`)

1. 인터넷 사진에서 두 시드 합 ≥ 343장 (현재 모델 B 167 + S7 162 + 14)
2. Valid에서 두 시드 합 ≥ 549장
3. 1·2를 넘으면 Valid가 높은 쪽으로 새 시험지와 Test를 한 번 채점한다. 새 시험지에서 현재 모델보다 5장 이상 많고, Test가 272장 이상일 때만 `model.pth`를 바꾼다.

## 결과

| 이름 | 시드 | Valid | 인터넷 사진 221장 | 그중 상한 토마토 45장 |
|---|---:|---:|---:|---:|
| B (현재 모델) | 42 | 281 | 167 | 17 |
| S7 | 7 | 274 | 162 | 16 |
| G42 (+886장) | 42 | 280 | **178** | **30** |
| G7 (+886장) | 7 | 273 | **179** | **32** |

1단계(178+179 = 357 ≥ 343)와 2단계(280+273 = 553 ≥ 549)는 넘었다. 그래서 G42로 마지막 채점을 했다.

| | 새 시험지 160장 | Test 300장 |
|---|---:|---:|
| B (현재 모델) | **148 (92.5%)** | **275 (91.7%)** |
| G42 | 138 (86.3%) | 268 (89.3%) |

**판정:** 3단계에서 둘 다 기준에 못 미쳐 채택하지 않았다. 새 시험지는 현재 모델보다 10장 적었고(기준은 5장 이상 많을 것), Test는 268장이었다(기준 272장).

## 무엇을 배웠나

- **인터넷 사진에서 좋아진 것이 다른 사진으로 이어지지 않았다.**
    - 밭에서 썩은 토마토 사진을 넣자 인터넷 사진의 상한 토마토는 17장에서 30장으로 크게 늘었다.
    - 하지만 새 시험지에서는 상한 토마토를 B와 G42 모두 20/20 맞혀 차이가 없었다.
    - 기준을 정하는 데 쓴 사진 하나만 보고 판단했다면 잘못 채택했을 것이다. 따로 떼어 둔 새 시험지가 이것을 막았다.
- **새로 생긴 실수: 흙 묻은 정상 당근을 상했다고 본다.** 새 시험지의 정상 당근 20장 중 11장을 상한 당근으로 봤다(B는 20장 모두 맞힘). 추가한 상한 당근에는 흙이 묻고 거친 당근이 많아서, "거친 표면 = 상함"을 배운 것으로 보인다.
- **Test는 7장 떨어졌다.** B만 맞힌 사진이 18장, G42만 맞힌 사진이 11장이었다. 상한 토마토(24→22), 상한 피망(27→24), 상한 오이(27→25)에서 줄었다.
- **두 모델 모두 못 하는 것:** 갈색으로 쪼그라든 오이를 상한 감자로 본다(새 시험지 상한 오이 10장 중 B 2장, G42 1장).

## 다시 돌리기

`added_photos.csv`의 사진을 각 Kaggle 데이터에서 받아 긴 변 640픽셀로 줄인다. 이것을 3차 데이터의 Train 폴더에 더하고, Test 행을 뺀 manifest로 학습한다. 학습 설정은 `model.pth`와 같다(`--epochs 30 --patience 8`, 시드 42·7).
