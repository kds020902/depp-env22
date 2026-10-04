# 채소 이미지 분류 1차 학습 모델

**실제로 학습을 완료한 ResNet18 전이학습 모델**이다. 채소 종류와 공개 데이터의 상태 라벨을 조합한 6개 클래스를 예측한다.

- 총 데이터 600장: 실제 가중치 학습 480장, 학습 중 검증 120장.
- 채소별 정상:비정상 5:5. 정확한 출처·수량은 DATA_USED.md와 manifest.csv에 있다.
- 생성일: 2026-09-27. 이번 1차 실행에는 Kaggle v2 자료만 사용했다.
- 사전 학습: ImageNet으로 이미 학습된 ResNet18 가중치를 가져왔다. ImageNet 사진 자체를 이번 학습 데이터에 추가하지 않았다.

## 실제로 학습한 부분

합성곱 본체와 BatchNorm은 고정했다. 마지막 분류층을 512차원 입력에서 6개 출력으로 교체한 뒤, 교차엔트로피 손실과 Adam으로 이 층의 가중치를 학습했다. 학습된 파라미터는 3,078개다. 특징은 한 번 계산해 저장하지 않고 메모리에서 재사용했으며, 데이터 증강은 적용하지 않았다.

학습 전후 분류층 가중치 차이 L2는 2.692549이고, 고정된 본체의 체크섬은 동일하다. 학습 여부 검증 결과는 metrics.json의 training_verification에 있다.

총 19 epoch를 실행했고 검증 손실이 가장 낮은 4 epoch를 저장했다. 사전 학습 모델을 단순히 이름만 바꿔 저장한 파일이 아니다.

## 검증 결과

| 항목 | 결과 |
|---|---:|
| 6개 조합 분류 정확도 | 86.67% |
| 채소 종류 정확도 | 96.67% |
| Fresh와 Rotten 상태 정확도 | 90.00% |
| 6개 클래스 Macro F1 | 0.8669 |

종류·상태 정확도는 6개 출력 확률을 종류별·상태별로 합산한 뒤 가장 큰 값을 선택해 계산했다. predict.py의 기본 species/condition은 가장 높은 조합 클래스에 속한 값이며, 별도로 marginal_prediction 열도 제공한다.

**위 수치는 모델 선택에도 사용한 검증 세트 결과이며, 독립적인 최종 시험 정확도가 아니다.** 직접 찍은 사진에서의 성능은 아직 측정하지 않았다. 배경·촬영 환경 차이와 남아 있을 수 있는 유사 개체 때문에 실제 성능은 달라질 수 있다.

## 파일 구성

- model.pth: 학습 완료 가중치와 클래스·전처리·학습 메타데이터. 추론할 때 별도 가중치 다운로드가 필요 없다.
- predict.py / model_utils.py: 새 사진 또는 폴더를 분류하는 코드.
- train.py: 같은 데이터로 다시 학습할 수 있는 코드.
- 학습결과_확인.ipynb: 학습 수량·곡선·예측을 살펴보는 노트북.
- images/: 실제 사용한 600장. manifest.csv가 학습/검증 구분과 원본 출처를 지정한다.
- metrics.json / history.csv / training_log.txt: 실제 학습 기록.
- val_predictions.csv / confusion_matrix.png: 검증 사진별 결과와 혼동행렬.
- DATA_USED.md / data_counts.csv / data_review/: 사용 자료 및 선별·분할 기록.

## 내 사진 넣고 실행하기

Python 3.12와 PyTorch 2.8.0, torchvision 0.23.0의 CPU 환경에서 검증했다. ZIP을 풀고 이 README가 있는 폴더에서 터미널을 연다.

```bash
python -m pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install "Pillow>=10.0,<12.0" "numpy>=1.26,<3.0"
python predict.py --model model.pth --input my_photos --output results.csv
```

my_photos 폴더를 만들고 직접 찍은 JPG 또는 PNG 사진을 넣는다. 사진 한 장을 넣으려면 --input my_photos/example.jpg로 지정한다. 폴더에 넣기만 하면 자동 실행되는 방식은 아니며 위 명령을 실행해야 한다.

결과의 cucumber=오이, potato=감자, tomato=토마토, fresh=원자료의 정상, rotten=원자료의 비정상 라벨이다. 점수는 보정되지 않은 softmax 값이며 정답 확률로 보장되지 않는다. 등록된 3종 밖의 물체도 6개 클래스 중 하나로 출력하므로, 이번 시험은 우선 3종에 한정한다.

## 다시 학습하기

```bash
python train.py --manifest manifest.csv --out retrained --device cpu
```

처음 다시 학습할 때는 공식 ImageNet 사전 학습 가중치 다운로드가 필요할 수 있다. 실제 촬영 시험 사진을 보고 설정을 반복 수정하지 말고, 설정 조정은 검증 데이터로 한다. 이 실행에는 정상 사진만 학습한 대조군 비교나 이상 부위 검출 학습이 포함되지 않았다.

## 참고

- [실제 사용 데이터](https://www.kaggle.com/datasets/muhriddinmuxiddinov/fruits-and-vegetables-dataset)
- [PyTorch 전이학습 공식 안내](https://docs.pytorch.org/tutorials/beginner/transfer_learning_tutorial.html)
- [ImageNet 사전 학습 가중치](https://download.pytorch.org/models/resnet18-f37072fd.pth)
