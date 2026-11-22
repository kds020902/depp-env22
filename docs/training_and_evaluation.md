# 학습·평가 절차 (기준 모델 TV-12)

기준 모델은 TV-12(`model.pth`)로 고정한다. 다시 돌리거나 후보와 비교할 때 필요한 것은 아래 두 파일에 모았다.

| 파일 | 내용 | 만드는 명령 |
|---|---|---|
| [`results/baseline/tv12.json`](../results/baseline/tv12.json) | 파일 해시, 분할, 시드, 설정, 환경, 학습 명령, 판정 기준, Valid·Test 지표, 블라인드 채점 파일 | `python tools/baseline_card.py` |
| [`results/baseline/tv12_dev.json`](../results/baseline/tv12_dev.json) | 개발 사진 1,935장에서 TV-12의 답과 지표 | `python tools/compare_on_dev.py results/baseline/tv12_dev.json` |

- 두 파일이 지금 저장소와 맞는지는 `python -m pytest -q`가 확인한다(`tests/test_baseline_record.py`, GitHub Actions에서도 실행).
- `model.pth`나 `manifest.csv`를 바꾸면 이 테스트가 실패한다. 그때 위 두 명령을 다시 실행한다.

## 1. 데이터 분할 → 학습 → 체크포인트 → 평가

| 단계 | 명령 | 읽는 것 | 남기는 것 |
|---|---|---|---|
| 데이터 분할 | `dataset/tools/build_dataset_v3.py` (확인: `tools/rebuild_and_compare.py`) | 원본 사진 | `dataset/manifest.csv`의 `split` 열 (train 1,260 / val 300 / test 300) |
| 채소 모델 학습 | `src/train.py --manifest dataset/manifest.csv --out OUT --seed 42 ...` | `split`이 train·val인 행 | `OUT/model.pth`, `metrics.json`, `history.csv`, `val_predictions.csv`, `test_predictions.csv` |
| 고르기 | `train.py` 안에서 | Valid 교차 엔트로피가 가장 낮은 에폭 | 체크포인트의 `best_epoch` (TV-12: 12) |
| 확률 보정·거르기 | `train.py` 안에서, 필요하면 `src/calibrate.py` | Valid | 체크포인트의 `calibration`, `unknown_check` |
| CLIP 분류기 | `src/train_clip_head.py --model 후보.pth ...` | Train + 추가 사진 CSV | 같은 파일의 `clip_head` |
| 새 채소 분류기 | `src/train_new_veg_head.py --model 후보.pth --data experiments/new_vegetables/v7/head_data.npz --tau 0.8` | 저장된 CLIP 특징 6,271행 | 같은 파일의 `new_vegetable_head` |
| 거르기에 글 점수 | `experiments/root_and_tomato/build.py`의 `refit_text_evidence` | TV-12의 글 임베딩 | `clip_head["text_evidence"]`, 거르기 기준 |
| 판정 | `src/decision.py`의 `decide()` 한 곳 | 체크포인트 | 답, 확률, 거름, 판단 보류 |
| 평가 (개발) | `tools/compare_on_dev.py` | 저장된 개발 세트 | 클래스별 표와 JSON |
| 평가 (블라인드) | `results/t*_test/score.py` | 봉인된 목록 | `scores_once.json` (한 번만) |

- 전체 명령은 [README의 "다시 학습하기"](../README.md#다시-학습하기)에 있다. 채소 모델의 정확한 옵션은 `results/baseline/tv12.json`의 `vegetable_model.command`에도 있다.
- 학습 단계는 모두 `--model`로 받은 파일을 제자리에서 고친다. 그래서 후보는 `candidate.pth`로 복사해 그 파일을 넘긴다.
- Test 300장은 `train.py`가 끝에 한 번 채점해 체크포인트에 적는다. 설정은 Valid만 보고 고른다.

## 2. 재실행에 필요한 것이 있는 곳

| 필요한 것 | 있는 곳 |
|---|---|
| 커밋 | `tv12.json`의 `commits` (`model.pth`, `manifest.csv`, `decision.py`를 마지막으로 바꾼 커밋) |
| 데이터 분할의 해시 | `tv12.json`의 `files_sha256["dataset/manifest.csv"]`. 사진마다의 해시는 manifest의 `sha256` 열(테스트가 확인) |
| 시드 | `tv12.json`의 `vegetable_model.seed` (42). 체크포인트에도 있다 |
| 설정 | `tv12.json`의 `vegetable_model.hyperparameters`, `decision_settings` |
| 환경 | `requirements.txt`(정확한 버전), Python 3.11, CPU. `tv12.json`의 `environment` |
| 체크포인트 | `model.pth`와 그 SHA-256 |
| 지표 | Valid·Test: `tv12.json`의 `vegetable_model.metrics`. 개발: `tv12_dev.json`. 블라인드: `results/t*_test/scores*.json`(해시는 `tv12.json`) |

### 그대로 다시 만들 수 없는 것 (알려진 한계)
- **CLIP 분류기의 학습 사진**은 라이선스 때문에 저장소에 없다. `tools/rebuild_head_photos.py`로 4,591장을 되찾아 다시 만든 것이 B0′이고, TV-12와 답이 조금 다르다(`experiments/root_and_tomato`).
- **채소 모델**은 같은 시드로 다시 학습해도 CPU 스레드 수와 라이브러리 버전에 따라 가중치가 달라질 수 있다. 이번에 다시 학습해 보지는 않았다.
- 그래서 **기준선은 "다시 학습한 결과"가 아니라 `model.pth` 파일 자체**로 잡는다. 후보는 이 파일에서 한 부분만 바꿔 만든다.

## 3. 기준 모델과 후보를 같은 개발 사진으로 비교하기

```bash
python tools/compare_on_dev.py OUT.json 후보이름=candidate.pth [다른이름=다른.pth]
```

- **변수 하나만 바꾼다.** 후보의 채소 모델이 `model.pth`와 다르면 이 도구는 멈춘다. 개발 세트가 채소 모델의 출력으로 저장되어 있기 때문이다. 채소 모델을 다시 학습한 후보는 사진으로 비교한다(`experiments/root_and_tomato/evaluate.py`).
- **개발 세트(블라인드 시험지 아님):**

| 세트 | 장 수 | 라벨 |
|---|---:|---|
| Valid | 300 | 있음 |
| 인터넷 채소 | 221 | 있음 |
| 밭·휴대폰 160장 시험지(개발용이 된 것) | 160 | 있음 |
| 아시아 오이 | 37 | 있음 |
| 새 채소 Valid | 89 | 있음 |
| 감자·고구마 웹 사진 | 162 | 있음 |
| 과일 / 일상 사진 | 374 / 400 | 없음 (걸러야 맞음) |
| 실제 장면 채소 조각 | 192 | 없음 (거르면 안 됨) |

- **따로 적는 숫자:** 클래스별 재현율, 상함 → 정상 오류, 판단 보류율, 거른 비율, 답을 낸 사진의 정확도, 기준 모델과 답이 달라진 사진 수

### TV-12의 개발 기준선 (라벨 있는 969장)

| 클래스 | 장 수 | 맞음 | 상함 → 정상 |
|---|---:|---:|---:|
| 오이 정상 | 99 | 89 (89.9%) | – |
| 오이 상함 | 42 | 33 (78.6%) | 1 |
| 감자 정상 | 126 | 97 (77.0%) | – |
| 감자 상함 | 133 | 126 (94.7%) | 4 |
| 토마토 정상 | 99 | 92 (92.9%) | – |
| 토마토 상함 | 95 | 79 (83.2%) | 8 |
| 피망 정상 | 70 | 69 (98.6%) | – |
| 피망 상함 | 42 | 42 (100%) | 0 |
| 당근 정상 | 78 | 76 (97.4%) | – |
| 당근 상함 | 43 | 42 (97.7%) | 1 |
| 양파 정상 | 29 | 26 (89.7%) | – |
| 양파 상함 | 5 | 5 (100%) | 0 |
| 고구마 정상 | 63 | 51 (81.0%) | – |
| 고구마 상함 | 25 | 18 (72.0%) | 3 |
| 대파 (종류만) | 20 | 14 (70.0%) | – |
| **전체** | **969** | **859 (88.6%)** | **17 / 385 (4.4%)** |

| 판단 보류 (0.9 미만) | 거름 | 답을 낸 사진의 정확도 |
|---:|---:|---:|
| 165 (17.0%) | 25 (2.6%) | 747 / 779 (95.9%) |

- 과일 374장 중 347장, 일상 사진 400장 중 400장을 거른다. 실제 장면 채소 192장 중 20장을 거른다.
- **주의:** 이 사진들은 여러 번 본 개발 사진이다. Valid와 새 채소 Valid는 기준값을 맞추는 데도 썼다. 그래서 이 표는 "실제 성능"이 아니라 **후보와 나란히 놓는 기준**이다. 실제 성능은 블라인드 시험지 결과를 본다(T3 79.6%, T8 72.2%).

## 4. CLIP 글 설명 (판정에 들어가는 문장)

코딩 에이전트에게 주는 지시문([`agent_prompts.md`](agent_prompts.md))과 다르다. 이것은 모델이 사진과 비교하는 영어 문장이다.

| 쓰임 | 코드 | 문장 | 답에 주는 영향 |
|---|---|---|---|
| 클래스 판정 | `decision.new_vegetable_logits`, `new_vegetable_head["text_embeddings"]` | 16개 클래스마다 3~4문장의 평균 (`tv12.json`의 `text_prompts`) | 두 번째 분류기 점수에 0.25배로 더한다. 양파·고구마·대파로 넘길지에 쓰인다 |
| 거르기 | `decision.text_evidence`, `clip_head["text_evidence"]` | 다섯 채소 5단어, 다른 것 86단어 | "다섯 채소가 아님" 점수에 0.25배로 더한다 |
| 가까운 항목 안내 | `clip_model.ClipModel.message`, `TEMPLATES` | 91개 단어 × 5문장 | 거른 사진에 "…로 보입니다"만 보여 준다. 답은 바꾸지 않는다 |

- 다섯 채소용 CLIP 분류기(`clip_head`)의 클래스 점수에는 글이 들어가지 않는다.
- 첫 사례(상한 고구마를 상한 감자로 답함)의 문구 비교는 [`experiments/sweetpotato_prompts`](../experiments/sweetpotato_prompts/README.md)에 있다. 결과: 문구만으로는 달라지지 않았다.
