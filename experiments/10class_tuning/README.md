# 10개 클래스 모델 추가 학습 (채택 안 함)

10개 클래스 모델(`model.pth`)을 더 좋게 만들 수 있는지 세 가지를 시험했다. 기준을 넘은 후보가 없어서 `model.pth`는 그대로 두었다.

## 미리 정한 규칙 (학습 전에 기록)

- 후보는 Test 행을 뺀 manifest로 학습하고, Valid로만 고른다. 후보의 Test 점수는 계산하지 않았다.
- 통과선 = 현재 모델 Valid + max(1.0%p, 시드만 바꿨을 때의 Valid 차이)
- 통과한 후보는 시드를 바꿔 한 번 더 돌리고, 두 시드 평균이 현재 모델·S7 평균보다 1.0%p 이상 높을 때만 채택한다. 그다음 Test를 한 번 채점해 91.7%보다 낮지 않을 때만 `model.pth`를 바꾼다.

## 결과 (Valid 300장)

| 이름 | 바꾼 것 | Valid 정확도 | Valid 손실 | 고른 epoch |
|---|---|---:|---:|---:|
| B (현재 모델) | 기본 설정, 시드 42 | 93.7% (281) | 0.330 | 16 / 24 |
| S7 | 시드만 7로 | 91.3% (274) | 0.337 | 17 / 25 |
| R | 학습 사진을 30% 확률로 썸네일 크기(짧은 변 48~112픽셀)로 줄였다가 키움 | 91.3% (274) | 0.341 | 22 / 30 |
| L | 최대 50 epoch, patience 12 | 93.7% (281) | 0.321 | 16 / 28 |

- 시드만 바꿔도 Valid가 2.3%p(7장) 달라졌다. 그래서 통과선은 93.7% + 2.3%p = 96.0%(288장)였다.
- R은 정상 피망 웹 썸네일 오답을 줄이려는 시도였지만, Valid 정상 피망이 25/30으로 현재 모델(26/30)보다 늘지 않았다.
- L은 Valid 손실만 조금 낮았고 정확도는 같았다.

`dev_results.csv`에 같은 값이 있다. 다시 돌리는 명령(Test 행을 뺀 manifest를 `manifest_notest.csv`로 만든 뒤):

```bash
python src/train.py --manifest manifest_notest.csv --out runs/S7 --arch efficientnet_b0 --mode finetune --epochs 30 --patience 8 --seed 7
python src/train.py --manifest manifest_notest.csv --out runs/R --arch efficientnet_b0 --mode finetune --epochs 30 --patience 8 --seed 42 --lowres-prob 0.3
python src/train.py --manifest manifest_notest.csv --out runs/L --arch efficientnet_b0 --mode finetune --epochs 50 --patience 12 --seed 42
```
