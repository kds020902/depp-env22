# 데이터셋 다시 만들기

`build_dataset_v2.py`는 두 Kaggle 원본과 검수 기록(`data_review/`)으로 2차 데이터셋(1,200장)을 다시 만든다. 사람이 한 눈 검수 결과는 `data_review/v2_visual_review.json`에 저장돼 있어서 다시 검수할 필요는 없다.

```bash
pip install -r dataset_tools/requirements.txt

# 1) 원본 내려받기 (Kaggle 로그인 없이 받을 수 있는 공개 데이터)
curl -L -o fv.zip "https://www.kaggle.com/api/v1/datasets/download/muhriddinmuxiddinov/fruits-and-vegetables-dataset?datasetVersionNumber=2"
curl -L -o fs.zip "https://www.kaggle.com/api/v1/datasets/download/swoyam2609/fresh-and-stale-classification"
unzip -q fv.zip -d src1
unzip -q fs.zip 'dataset/*/freshcucumber/*' 'dataset/*/rottencucumber/*' -d src2

# 2) 1차 manifest 꺼내기
git show 03db425:manifest.csv > v1_manifest.csv

# 3) 다시 만들기 (CPU 4코어 기준 약 10분)
python dataset_tools/build_dataset_v2.py --src1 src1 --src2 src2 --v1-manifest v1_manifest.csv --out rebuilt
```

결과는 `rebuilt/images/`와 `rebuilt/manifest.csv`로 나온다. 순서는 다음과 같다.

1. 1차 사진 중 라벨 재검수로 뺀 4장을 제외한다.
2. 눈 검수를 통과한 사진을 더한다. 두 번째 출처 오이는 테두리를 잘라낸 뒤 정상·비정상 30장씩 고른다.
3. 지각 해시, 반전·회전 불변 ResNet18 특징, ORB 특징점 매칭으로 같은 사진의 복사본을 한 그룹으로 묶는다.
4. 클래스마다 Train 140 / Valid 30 / Test 30으로 나눈다.

자세한 기준은 저장소 맨 위의 `DATA_USED.md`에 있다.
