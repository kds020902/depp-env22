# 피망 웹 사진 추가 (채택 안 함)

정상 피망이 10개 클래스 중 가장 약했다(Valid 26/30, Test 25/30). 정상 피망 Train 106장 중 89장이 데이터 제공자가 시장에서 직접 찍은 사진이고, 상한 피망 Train은 모두 웹 사진이어서 "웹 사진이면 상함"으로 배웠을 가능성이 있다. 그래서 다른 Kaggle 데이터에서 정상·상한 피망 웹 사진을 찾아 Train에 더해 봤다. 미리 정한 기준을 넘지 못해서 `model.pth`와 `dataset/`은 그대로 두었다. Test는 채점하지 않았다.

## 추가한 사진

- **출처:** [vetion/vege-quality](https://www.kaggle.com/datasets/vetion/vege-quality) (Kaggle, 버전 1, ODbL/DbCL)의 `Paprika Merah Bagus`(정상)·`Paprika Merah Jelek`(상함). 인터넷 스톡 사진을 224×224로 줄인 것이다.
- **정상·상함을 같은 출처에서 같은 수로:** 피망 Train을 정상 106→140장, 상함 106→140장으로 다른 채소와 같게 채웠다. 상한 사진만 있는 출처([user2036](https://www.kaggle.com/datasets/user2036/vetable-dataset-without-duplicacy), CC0)도 검수했지만, 출처 자체가 "상함"의 단서가 될 수 있어 쓰지 않았다.
- **Valid·Test는 그대로:** 새 사진은 Train에만 넣었다.

## 검수

1. **같은 사진 빼기:** 후보 611개 파일(픽셀까지 같은 파일을 빼면 361장)을 복사본 묶음 265개로 묶었다. 3차 데이터 1,860장과 같은 사진이 들어 있는 묶음 34개(89장)는 통째로 뺐다. 기준은 3차와 같다(pHash/dHash, 반전·회전 불변 ResNet18 코사인 0.92, ORB 특징점 매칭).
2. **눈 검수:** 복사본 묶음마다 1장씩 231장을 봤다. 3차 기준(종 모양 피망만, 썬 조각·합성·일러스트 제외, 상한 흔적이 분명할 것)에 더해 구운 피망처럼 보이는 사진도 뺐다. vetion은 정상 86장 중 75장, 상함 57장 중 36장이 통과했다.
3. **Valid·Test와 닮은 사진:** 통과한 사진 중 Valid·Test 사진과 유사도 0.86 이상인 43장을 모두 눈으로 비교했다.
    - Valid 상한 피망과 워터마크만 다른 같은 스톡 사진 2장을 뺐다.
    - Test 정상 피망과 같아 보이는 사진 1장, 확신이 서지 않는 사진 1장도 뺐다.
4. **최종:** 남은 vetion 사진은 정상 73장, 상함 34장이다. 상함 34장은 모두 넣고, 정상 73장 중 34장은 무작위(seed 0)로 골랐다.

사진별 결정은 `visual_review.json`에 있다.

## 미리 정한 규칙 (학습 전에 기록)

- 3차 Train에 피망 68장을 더한 데이터로 시드 42·7을 한 번씩 학습한다(설정은 `model.pth`와 같음). Test 행을 뺀 manifest를 쓴다.
- 비교 대상은 현재 모델(시드 42, Valid 281)과 시드만 바꾼 S7(Valid 274)이다.
- 두 시드 Valid 합이 281+274+6 = 561장 이상(평균 +1.0%p)이고, Valid 피망 합이 54+52 = 106장 이상일 때만 채택한다.
- 채택하면 Valid가 높은 쪽으로 Test를 한 번 채점하고, 91.7% 이상일 때만 `model.pth`를 바꾼다.

## 결과 (Valid 300장)

| 이름 | Train | 시드 | Valid 정확도 | Valid 손실 | 고른 epoch | 정상 피망 | 상한 피망 |
|---|---|---:|---:|---:|---:|---:|---:|
| B (현재 모델) | 3차 1,260장 | 42 | 93.7% (281) | 0.330 | 16 / 24 | 26/30 | 28/30 |
| S7 | 3차 1,260장 | 7 | 91.3% (274) | 0.337 | 17 / 25 | 25/30 | 27/30 |
| P42 | + 피망 68장 | 42 | 92.3% (277) | 0.336 | 26 / 30 | 29/30 | 25/30 |
| P7 | + 피망 68장 | 7 | 91.3% (274) | 0.296 | 15 / 23 | 30/30 | 25/30 |

- **판정:** 두 시드 합이 551장으로 기준 561장에 못 미쳐 채택하지 않았다. 피망 합(109장)은 기준(106장)을 넘었다.
- **좋아진 점:** 정상 피망 오답이 두 시드에서 9장에서 1장으로 줄었다. 현재 모델이 토마토로 본 빨간 피망 2장, 상함으로 본 2장을 두 시드 모두 맞혔다.
- **나빠진 점:** 상한 피망은 두 시드 모두 25/30으로 떨어졌다. 두 시드 모두 상한 피망 1장을 정상으로, 상한 토마토 1장을 상한 피망으로 봤다. 나머지 클래스도 1~2장씩 오르내려 전체로는 4장 적었다.
- 두 시드를 합치면 정상 피망은 8장 늘었지만 상한 피망이 5장, 나머지 클래스가 7장 줄어 전체로는 4장 적었다. 시드만 바꿔도 7장이 달라지므로, 정상 피망이 좋아진 것 외에는 개선이라고 볼 근거가 없다.

`dev_results.csv`에 같은 값이 있다.

## 다시 만들기

```bash
curl -L -o vq.zip "https://www.kaggle.com/api/v1/datasets/download/vetion/vege-quality"
unzip -q vq.zip -d vetion
python experiments/pepper_web/build_pepper_dataset.py --vetion vetion --base dataset --out pepper_web_dataset
# Test 행을 뺀 manifest로 학습
python -c "import csv; r=list(csv.DictReader(open('pepper_web_dataset/manifest.csv', encoding='utf-8'))); w=csv.DictWriter(open('pepper_web_dataset/manifest_notest.csv', 'w', newline='', encoding='utf-8'), fieldnames=list(r[0])); w.writeheader(); w.writerows(x for x in r if x['split'] != 'test')"
python src/train.py --manifest pepper_web_dataset/manifest_notest.csv --out runs/P42 --arch efficientnet_b0 --mode finetune --epochs 30 --patience 8 --seed 42
python src/train.py --manifest pepper_web_dataset/manifest_notest.csv --out runs/P7 --arch efficientnet_b0 --mode finetune --epochs 30 --patience 8 --seed 7
```

`build_pepper_dataset.py`는 `dataset/`을 바꾸지 않고, 사진을 더한 복사본을 `--out` 폴더에 만든다.
