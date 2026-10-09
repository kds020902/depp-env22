# 휴대폰으로 찍은 정상·상한 채소 사진 추가 (채택 안 함)

인터넷에 올라온 사진에서 현재 모델은 75.6%만 맞힌다([results/10class/outside_test](../../results/10class/outside_test/README.md)). 학습 데이터가 대부분 인터넷 스톡 사진이라, 사람이 직접 찍은 정상·상한 채소 사진을 Train에 더하면 나아지는지 시험했다. 미리 정한 기준을 넘지 못해서 `model.pth`와 `dataset/`은 그대로 두었다. Test는 채점하지 않았다.

## 추가한 사진 (Train에만, 480장)

| 채소 | 출처 | 정상 | 상함 |
|---|---|---:|---:|
| 오이 | [Cucumber Disease Recognition](https://www.kaggle.com/datasets/sujaykapadnis/cucumber-disease-recognition-dataset) (CC BY 4.0), 밭에서 찍은 사진 | 60 | 60 (belly rot·피시움 과실 부패) |
| 감자 | [vegetables-quality-dataset-2](https://www.kaggle.com/datasets/mdatikurrahman3111/vegetables-quality-dataset-2) (CC0), 흰 종이 위 휴대폰 사진 | 60 | 60 (싹, 곰팡이) |
| 토마토 | 같은 데이터 | 60 | 60 (터지고 물러짐) |
| 당근 | 같은 데이터 | 60 | 60 (마름, 꼭지 부패, 검게 변함) |

- **정상·상함을 같은 출처에서 같은 수로 넣었다.** 상한 사진만 새 출처에서 넣으면 "이런 배경이면 상함"을 배울 수 있어서다.
- **피망은 넣지 않았다.** 정상과 상함이 모두 있는 출처를 찾지 못했다.
- **복사본 제거:** 3차 데이터 1,860장, 인터넷 평가 사진과 같은 사진이 든 묶음은 통째로 뺐다.
- **고르는 순서:** 같은 사진의 복사본은 한 장만 남겼다. 비슷한 배경·구도 묶음을 무작위 순서(seed 0)로 돌아가며 한 장씩 보고, 60장이 통과할 때까지 검수했다.
- **검수 기준:** 3차 기준에 더해, 감자·당근은 칼집이나 갈라짐만 있고 썩거나 싹이 나지 않은 사진, 부러진 당근, 반으로 자른 토마토는 상함에서 뺐다. 보라색 당근은 정상에서 뺐다.
    - 클래스마다 본 수와 통과한 수는 `review_counts.json`에 있다.
    - 쓴 사진의 원본 경로는 `added_photos.csv`에 있다.

## 미리 정한 규칙 (학습 전에 기록, `PREREG.md`)

- 시드 42(R42)·7(R7)로 한 번씩 학습한다. 설정은 `model.pth`와 같다.
- 비교 대상은 현재 모델(B)과 시드만 바꾼 S7이다.
- 다음 두 조건을 모두 만족할 때만 채택한다.
    1. **인터넷 사진 221장:** 두 시드 합이 B+S7보다 14장 이상 많을 것(평균 +3%p). 기준선은 167+162+14 = 343장이다.
    2. **Valid 300장:** 두 시드 합이 281+274−6 = 549장 이상일 것(평균 1%p 넘게 떨어지지 않음).
- 인터넷 사진 시험지는 학습 전에 정답을 정해 고정했다.

## 결과

| 이름 | Train | 시드 | Valid | 인터넷 사진 221장 | 종류만 | 정상·상함만 |
|---|---|---:|---:|---:|---:|---:|
| B (현재 모델) | 3차 1,260장 | 42 | 281 | 167 | 198 | 186 |
| S7 | 3차 1,260장 | 7 | 274 | 162 | 193 | 186 |
| R42 | + 휴대폰 사진 480장 | 42 | 276 | 166 | 193 | 185 |
| R7 | + 휴대폰 사진 480장 | 7 | 279 | 167 | 196 | 183 |

- **판정:** 인터넷 사진 두 시드 합이 333장으로 기준(343장)에 못 미쳐 채택하지 않았다. Valid 합(555장)은 기준(549장)을 넘었다.
- **가장 약한 상한 토마토는 그대로였다.** 인터넷 사진 45장 중 B 17장, R42 16장, R7 16장을 맞혔다.
    - 새로 넣은 상한 토마토는 흰 종이 위에서 터지고 물러진 사진이다.
    - 인터넷 사진의 상한 토마토는 밭에 달린 채 아랫부분이 검게 썩거나, 덜 익은 초록색이거나, 곰팡이가 핀 것이다.
    - 썩은 모양이 달라서 도움이 되지 않은 것으로 보인다.
- **Valid는 조금 떨어졌다.** 평균 1%p 이내라 기준 안이다.

`dev_results.csv`에 같은 값이 있다.

## 다시 돌리기

두 Kaggle 데이터를 내려받는다(로그인 없이 받을 수 있다). `added_photos.csv`의 `archive_path` 사진을 긴 변 640픽셀로 줄여 3차 데이터의 Train 폴더에 더한다. 그다음 Test 행을 뺀 manifest로 학습한다.

```bash
python src/train.py --manifest <새 manifest_notest.csv> --out runs/R42 --arch efficientnet_b0 --mode finetune --epochs 30 --patience 8 --seed 42
python src/train.py --manifest <새 manifest_notest.csv> --out runs/R7 --arch efficientnet_b0 --mode finetune --epochs 30 --patience 8 --seed 7
python results/10class/outside_test/evaluate.py run --photos outside_photos --model runs/R42/model.pth
```
