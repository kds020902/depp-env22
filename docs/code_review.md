# depp-env22 코드 평가

1. **현업 기준 72/100점(36/50)**이다. 현재 TV-13의 8종 판정 흐름과 봉인 평가의 관리 수준은 좋다. 감점이 큰 곳은 빠른 CI의 예측 공백, 실행 기록의 버전 혼동, 실험 스크립트의 재실행 의존성이다.
2. **발표 전에는 세 가지면 된다.** README의 모델 표기와 발표용 실행 절차를 맞추고, CSV·HTML 출력이 같은 파일일 때 거부하고, CLIP 가중치 없이 실행되는 CLI 회귀 테스트를 빠른 CI에 연결한다.
3. **Spaces에 올리기 전에는** 전용 실행 환경·업로드 한도·콜드 스타트·전체 테스트를 검증한다. 학습 함수 분리와 옛 실험 정리는 그 뒤에 한다.

평가 기준 커밋은 작업 브랜치 `claude/great-planck-vnvyxx`의 [`799c2f6`](https://github.com/kds020902/depp-env22/commit/799c2f60653604d5a7472c7fe2ec6dc3b2fd6061)이다. 확인 시점의 원격 HEAD와 같았다. [협업 노트 맨 위 세 항목](https://github.com/kds020902/depp-env22/blob/799c2f60653604d5a7472c7fe2ec6dc3b2fd6061/docs/codex_handoff.md#L7), `README.md`, `docs/training_and_evaluation.md`를 읽었다. 검토 범위는 TV-13의 8종 시스템과 학습·평가 도구 및 보관된 실험이다.

## 점수표

| 평가 기준 | 점수 | 근거 번호 |
|---|---:|---|
| 가독성·복잡도 | 4/5 | 1 |
| 구조·결합도 | 4/5 | 2 |
| 중복·사용 경로 정리 | 3/5 | 3 |
| 오류 처리·파일 보존 | 3/5 | 4 |
| 테스트·CI | 3/5 | 5 |
| 문서·운영 인계 | 4/5 | 6 |
| 재현성·실험 이력 | 3/5 | 7 |
| 데이터 누수 방지 | 4/5 | 8 |
| 평가의 엄밀함 | 4/5 | 9 |
| 학습·추론 일치 | 4/5 | 10 |
| **전체 · 동일 가중치** | **36/50 · 72%** | |

| 영역 | 점수 | 강점 | 주요 감점 |
|---|---:|---|---|
| 가. 예측·판정 | **80/100** | 첫 단계 → 새 채소 → 감자 재확인을 읽는 순서대로 배치했다. `src/decision.py:155–199` | 현재 CLI가 같은 CSV·HTML 경로를 받아들이고 HTML이 앞선 CSV를 덮는다. `src/predict.py:89–96,274–299` |
| 나. 학습·보정 | **65/100** | 목록과 실제 사진 해시를 검사하고 Valid 손실로 에폭을 고른다. `src/train.py:34–73,325–335` | `main()`이 164줄이고, 보정 및 CLIP 학습은 입력 체크포인트에 직접 쓴다. `src/train.py:222–385`, `src/calibrate.py:38–51`, `src/train_clip_head.py:171–176` |
| 다. 평가 도구 | **75/100** | 개발 캐시 호환 검사와 T9 일회 실행 장치가 있다. `tools/compare_on_dev.py:148–158`, `results/t9_test/score.py:147–173` | 옛 채점기의 기본 모델 이름과 실제 파일이 어긋난다. `results/t9_test/score.py:191–214`, `results/t7_test/score.py:43–57` |
| 라. 테스트 | **75/100** | 모델 체크섬, 회귀 결과, 봉인과 분할 경계까지 검사한다. `tests/test_project.py:40–69`, `tests/test_t9_score.py:84–132` | 예측 CLI 5개 전부 기본 CI에서는 건너뛴다. `.github/workflows/tests.yml:8–23`, `tests/test_predict_outputs.py:59–84,152–170`, `tests/test_project.py:136–170` |
| 마. 실험 | **50/100** | PC1 실험은 저장 특징으로 원 기준선을 되살릴 수 있다. `experiments/potato_check/evaluate.py:58–82` | 남겨 둔 스크립트가 삭제한 목록과 외부 작업 폴더의 스크립트에 의존한다. `experiments/new_vegetables/v7/build_pool.py:19–22`, `experiments/new_vegetables/v7/export_head_data.py:5–13` |
| 바. 문서 | **80/100** | 모델·시험 버전, 채택 근거, 재현의 한계를 설명한다. `README.md:5–27`, `docs/training_and_evaluation.md:34–48,96–104` | README 파일 트리에는 현재 모델이 TV-12로 남아 있고, 추론·학습 환경이 설치 파일 하나에 섞여 있다. `README.md:330–336`, `requirements.txt:1–17` |

영역 점수는 각 부분의 현재 운영 위험을 반영한 별도 평가다. 전체 점수는 위 10개 기준만 합산했다.

**학생 프로젝트로서의 수준.** 단일 추론 경로, 모델·파일 해시, 고정 예측 테스트, 사전 조건과 채택 이력을 연결한 점이 돋보인다. 발표에서 T9의 조건과 화면상 개선을 구분해 설명할 근거도 충분하다. 팀 인계나 공개 호스팅 단계에서는 빠른 CLI 회귀 보호와 재실행 방법을 정리할 필요가 있다. 학생 프로젝트라는 이유로 점수를 높이지 않았다.

## 평가 기준과 출처

일반 코드 검수 기준은 Google의 설계·기능·복잡도·테스트·이름·문서 지침을 사용했다. 줄 길이는 Python 스타일 지침의 79자와 실측 비교용 88자를 같이 표시했다. 줄 길이만으로 감점하지 않았다. [Google Engineering Practices](https://google.github.io/eng-practices/review/reviewer/looking-for.html) · [PEP 8](https://peps.python.org/pep-0008/#maximum-line-length)

ML 기준은 Google의 공통 전처리·훈련/서빙 차이 지침, Breck 등의 데이터·모델·인프라·모니터링 점검표, scikit-learn의 평가 데이터 분리와 그룹 독립성 지침에 근거했다. Spaces 관련 제안은 Hugging Face의 앱·SDK 설정과 의존성 문서를 참조했다. [Rules of Machine Learning 규칙 32](https://developers.google.com/machine-learning/guides/rules-of-ml/#rule_32_re-use_code_between_your_training_pipeline_and_your_serving_pipeline_whenever_possible) · [The ML Test Score](https://research.google/pubs/the-ml-test-score-a-rubric-for-ml-production-readiness-and-technical-debt-reduction/) · [scikit-learn의 누수 방지](https://scikit-learn.org/stable/common_pitfalls.html#data-leakage) · [GroupKFold](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GroupKFold.html) · [Spaces 설정](https://huggingface.co/docs/hub/spaces-config-reference) · [Spaces 의존성](https://huggingface.co/docs/hub/spaces-dependencies)

각 항목에서 **1점**은 기본 동작 또는 증거가 없거나 실패가 반복됨, **2점**은 일부 경로만 관리, **3점**은 일반 경로를 관리하나 실질적인 공백이 있음, **4점**은 주요 경로가 일관되고 일부 한계가 있음, **5점**은 주요·예외 경로를 함께 검증하고 새 사람이 같은 결과를 확인할 수 있는 수준이다. 항목별 판단은 다음과 같다.

| 기준 | 3점의 증거 | 5점의 증거 |
|---|---|---|
| 1. 가독성·복잡도 | 핵심 흐름을 추적할 수 있으나 긴 함수·숨은 문맥이 남음 | 주요 진입점의 책임과 데이터 흐름이 이름과 작은 함수로 드러남 |
| 2. 구조·결합도 | 공통 모듈은 있으나 파일 경로·부수 효과에 내부 로직이 결합됨 | 공개 추론 경계가 좁고 환경·출력·판정을 독립적으로 검증 |
| 3. 중복·사용 경로 | 주요 경로는 공유하되 실험 복제와 과거 옵션이 섞임 | 호출부·호환 범위·보존 근거를 확인하고 한 군데에서 동작을 정의 |
| 4. 오류·보존 | 일반 실패와 일부 충돌을 막지만 중간 실패·덮어쓰기 공백이 있음 | 쓰기 전 검증, 부분 실패·재시도·충돌 대응을 테스트 |
| 5. 테스트·CI | 단위·회귀 테스트는 있고 핵심 명령 통합이 기본 실행에서 빠짐 | 빠른 통합과 실제 모델 테스트를 자동화하며 실패·경계도 검증 |
| 6. 문서·인계 | 실행 방법은 있으나 버전·과거 기록·환경을 추론해야 함 | 최신 모델·실행 환경·시험 사용 여부·알려진 한계를 한 번에 확인 |
| 7. 재현성과 기록 | 해시와 실행 이력은 있지만 필수 입력·명령 일부가 복구 불가 | 모델·데이터·환경·실험 입력과 결과의 출처를 고정하고 검증 |
| 8. 누수 방지 | 중복 일부를 막고 촬영 그룹 검증이 불완전 | 이미지·근접 중복·촬영자/세션 분리와 후보 선택 데이터의 경계를 검증 |
| 9. 평가 엄밀함 | 별도 평가가 있으나 지표·사전 조건·실행 이력에 공백 | 사전 판정·봉인·버전·정의가 일치하고 재사용 및 결과 쓰기가 통제됨 |
| 10. 학습·추론 일치 | 일부 전처리나 판정을 공유하나 차이를 검출할 검증이 부족 | 학습·평가·추론의 입력, 모델 버전, 출력 의미가 공통 계약과 회귀로 연결 |

## 코드 분포

### 폴더별

`git ls-files`에 있는 파일만 세었다. 물리 줄 수는 Python `splitlines()` 기준으로 빈 줄·주석을 포함한다. 모든 파일 수는 바이너리를 포함한다. 텍스트 줄 수에서는 NPZ·이미지 등 바이너리와 UTF-8로 읽히지 않는 파일을 제외했다. 텍스트 줄에는 CSV·JSON·Markdown 기록도 포함된다.

| 폴더 | 추적 파일 | 텍스트 파일 | 텍스트 줄 | Python 파일 | Python 줄 |
|---|---:|---:|---:|---:|---:|
| `src/` | 15 | 15 | 2,362 | 15 | 2,362 |
| `tools/` | 8 | 8 | 894 | 8 | 894 |
| `tests/` | 10 | 10 | 1,096 | 9 | 998 |
| `experiments/` | 147 | 135 | 81,498 | 49 | 4,347 |
| `results/` | 120 | 108 | 24,967 | 18 | 1,766 |
| `dataset/tools/` | 2 | 2 | 563 | 2 | 563 |
| **합계** | **302** | **278** | **111,380** | **101** | **10,930** |

실험은 상위 폴더가 **39 → 16개(-59.0%)**로 줄었다. 현재도 `experiments/`와 `results/`에 Python **6,113줄(55.9%)**이 있다. 기록 파일을 포함한 텍스트 줄 수를 실제 코드량으로 해석하면 규모가 과장된다.

### `src/` 파일별

| 파일 | 줄 | 함수 | 가장 긴 줄(문자) |
|---|---:|---:|---:|
| `calibrate.py` | 56 | 1 | 117 |
| `calibration.py` | 71 | 8 | 119 |
| `clip_model.py` | 99 | 8 | 130 |
| `decision.py` | 244 | 18 | 141 |
| `images.py` | 88 | 8 | 123 |
| `labels.py` | 24 | 1 | 119 |
| `make_plots.py` | 134 | 5 | 154 |
| `model_utils.py` | 72 | 0 | 238 |
| `networks.py` | 193 | 12 | 121 |
| `predict.py` | 305 | 15 | 147 |
| `report.py` | 234 | 4 | 182 |
| `train.py` | 389 | 9 | 155 |
| `train_clip_head.py` | 181 | 3 | 161 |
| `train_new_veg_head.py` | 118 | 5 | 166 |
| `watermark_aug.py` | 154 | 11 | 132 |
| **합계** | **2,362** | **108** | **238** |

함수는 AST의 `FunctionDef`와 `AsyncFunctionDef`로 세었다. 메서드·중첩 함수도 포함한다. 길이는 `def`부터 마지막 문장까지의 물리 줄 수이며 데코레이터는 제외했다.

| 가장 긴 함수 | 위치 | 줄 수 |
|---|---|---:|
| `train.main` | `src/train.py:222–385` | **164** |
| `train_clip_head.main` | `src/train_clip_head.py:71–177` | **107** |
| `train_new_veg_head.main` | `src/train_new_veg_head.py:46–114` | **69** |
| `RandomWatermark._one` | `src/watermark_aug.py:86–142` | **57** |
| `predict.main` | `src/predict.py:256–301` | **46** |

| 줄 길이 | 전체 2,362줄 | 빈 줄 제외 2,043줄 |
|---|---:|---:|
| 0~79자 | 1,652 (69.9%) | 1,333 (65.2%) |
| 80~88자 | 144 (6.1%) | 144 (7.0%) |
| 89~100자 | 164 (6.9%) | 164 (8.0%) |
| 101~120자 | 280 (11.9%) | 280 (13.7%) |
| 121자 이상 | 122 (5.2%) | 122 (6.0%) |

88자 초과는 전체 **566줄(24.0%)**이다. 238자 최장 줄은 보관 실험용 재내보내기 모듈 `src/model_utils.py:17`의 import다. 실행 흐름의 복잡도는 아래 코드 근거와 함께 판단했다.

### 실제 테스트와 공백

| 실제 실행 | 결과 |
|---|---|
| `python -m pytest -q` | **55 통과, 5 건너뜀**, 176.64초 |
| `FULL_TESTS=1 python -m pytest -q` | **60 통과, 402.98초** |
| `python tools/baseline_card.py --check` | **일치** |
| GitHub 빠른 CI · 정리를 적용한 `d46439f` | [성공](https://github.com/kds020902/depp-env22/actions/runs/45405337439) |

로컬은 Windows, Python 3.13.3, CPU torch 2.14.1, torchvision 0.29.1, pytest 8.4.2였다. 로컬 기본 실행도 `FULL_TESTS`를 비워 CI와 같은 55개를 골랐다. GitHub 빠른 CI는 Ubuntu/Python 3.11/CPU torch 2.8이며 CLIP을 설치하지 않는다. 전체 실행은 이미 캐시된 약 599MB의 CLIP 가중치로 오프라인 실행했다. 두 환경의 실행 시간을 직접 비교할 수 없다.

| 범위 | 빠른 CI에서 확인하는 것 | 비어 있는 계약 |
|---|---|---|
| 데이터·학습 입구 | manifest 전 파일 해시·그룹, 잘못된 학습/보정 입력, 데이터 빌드 비교. `tests/test_project.py:40–51`, `tests/test_manifest_checks.py:33–101`, `tests/test_compare_dataset_builds.py:38–70` | 실제 학습 1 epoch와 저장→불러오기, 전체 CLIP 재학습은 검사하지 않음 |
| 판정·평가 | 체크포인트·채소 모델 출력, 가상 특징 판정, 개발셋 비교, T9 봉인·집계·동시 실행 경계, 고구마 사진 분할. `tests/test_project.py:53–117`, `tests/test_baseline_record.py:18–83`, `tests/test_t9_score.py:16–194`, `tests/test_sweetpotato_split.py:24–133` | 실제 운영 입력에서의 품질, 옛 스크립트의 현재 체크아웃 재실행은 검사하지 않음 |
| 예측 CLI | 경로 보호, 두 옵션 삭제, 이름 생성, 리포트 안내 문구. `tests/test_predict_outputs.py:32–57,86–150`, `tests/test_project.py:146–162` | 실제 예측과 출력 쓰기를 통과하는 아래 5개가 기본 CI에서 빠짐 |

**`FULL_TESTS=1`에서만 실행되는 5개(전체의 8.3%)**는 같은 이름의 사진 두 장의 히트맵이 서로 덮이지 않음, 모든 사진이 실패하면 종료 코드 오류, 기존 히트맵을 다음 입력에서 제외, 판단 보류 끄기, CLI 골든 예측이다. `tests/test_predict_outputs.py:59–84,152–170`, `tests/test_project.py:136–145,165–171`. 이들 외에 실제 웹 앱, 콜드 스타트, 동시 요청과 큰/손상 업로드, 완전한 학습은 **확인 못 함**이다. 줄·분기 단위의 커버리지 %는 측정하지 않았다.

### 현행 예측에서 빠진 기능과 남아 있는 호출부

| 항목 | 정의 | 추적한 호출부와 분류 |
|---|---|---|
| 토마토 조각 검사 `needs_crop_check`, `apply_crop_check`, `five_crops` | `src/decision.py:220–244`, `src/images.py:84–88` | `experiments/tomato_crops/evaluate.py:29–57`, `results/t6_test/score.py:29–30,65–72`, `results/t7_test/score.py:27–28,59–63`가 호출한다. `decide()`와 예측 CLI에서는 호출하지 않는다. |
| 별도 조각 함수 `five_crops` | `experiments/improve_without_photos/crops.py:28–42` | 같은 이름의 독립 실험 구현이다. 위 `src/images.py` 함수의 호출부로 세지 않았다. |
| `train_new_veg_head.py --untrained` | `src/train_new_veg_head.py:55–66` | `experiments/no_greenonion/build.py:46–48`가 명시 인자를 넘긴다. 현재 기본 대파 상태 제외는 `UNTRAINED_CLASSES`도 사용한다. 옵션 삭제 시 기본 동작을 별도로 보존해야 한다. |
| `train_clip_head.py --clip-arch`, `--clip-pretrained` | `src/train_clip_head.py:96–107,121,145` | `experiments/bigger_eye/build.py:99–102`가 사용한다. 현재 8종 예측의 옵션은 아니며, 학습기에서 인코더와 캐시의 신원을 결정한다. |
| 옛 가져오기 모듈 `model_utils.py` | `src/model_utils.py:1–19` | `experiments/watermark/common.py:24`, `results/t3_test/score.py:29`, `tests/test_t9_score.py:67` 등에서 현재 참조된다. |

따라서 위 후보들은 **현행 제품 경로 밖의 호환 코드**다. 파일을 바로 지우면 아직 남아 있는 역사적 호출부가 깨진다. `--no-clip`·`--five-only`는 현행 CLI에서 제거됐다는 테스트가 있다. `tests/test_project.py:146–162`.

### 지운 실험 폴더 의존성

협업 노트의 **11개**를 확인하기 위해 `73b30d6`과 비교하고 직접 파일 경로, 외부 작업 자료, 주석을 나눴다. **현재 체크아웃에서 삭제된 추적 파일을 직접 여는 Python 소스는 4개**다. **과거 실험의 외부 작업 파일·캐시에 기대는 관련 코드 7개**도 추적했다. 협업 노트가 같은 기준으로 11개를 셌는지와 해당 외부 파일의 실제 보유 여부는 **확인 못 함**이다. 아래 파일은 실행하지 않았다.

| 구분 | 파일·줄 | 확인한 읽기 |
|---|---|---|
| 삭제 경로 직접 | `experiments/new_vegetables/v5/build_pool.py:20–22`, `experiments/new_vegetables/v6/build_pool.py:19–21`, `experiments/new_vegetables/v7/build_pool.py:19–21` | `asian_cucumber/asian_check.csv`, `fruit_other/fruit_check.csv` |
| 삭제 경로 직접 | `results/t4_test/score.py:42–46,95–98` | `t3_weak_spots/PREREG.md`, 기본 `t3_weak_spots/dev_results.json` |
| 외부 작업 자료 | `experiments/watermark/common.py:156–188`, `experiments/bigger_eye/build.py:65`, `experiments/bigger_eye/evaluate.py:37,61–64`, `experiments/bigger_eye/lists.py:54` | 옛 약점 실험의 `t5/lists/dev.csv` 및 `t3ws_dev_*.npz` |
| 외부 작업 자료 | `experiments/improve_without_photos/crops.py:91–93`, `experiments/improve_without_photos/text_mix.py:44–55` | 위 공용 도구를 통해 옛 개발 목록이나 캐시를 읽음 |
| 외부 작업 자료 | `experiments/not_five_class/evaluate.py:30–39` | `ext3`·`ext4` 특징·검수 라벨과 아시아 오이 작업 자료 |

예를 들어 `results/10class/outside_test/evaluate.py:7`의 삭제 경로는 사용 예시에 있는 문자열이라 직접 읽기로 세지 않았다. 반대로 `experiments/new_vegetables/v7/export_head_data.py:8–10`은 `DEPP_WORK/v7/explore.py` 본문을 읽고 `exec()`에 넘긴다. 이것은 별도의 외부 의존성 위험이다.

### 측정에 쓴 명령

아래 명령을 고정한 체크아웃에서 돌렸다. Python 실측은 저장소 바깥의 임시 스크립트에서 수행했다. `git ls-files -z`의 UTF-8 텍스트를 세고, `ast.parse()`의 `lineno`·`end_lineno`로 함수 시작·끝과 길이를 계산했다. `dataset/manifest.csv`는 `csv.DictReader`로 세 분할의 `sha256`, `pixel_sha256`, `group_id`, `session_id` 교집합을 계산했다.

```text
git rev-parse HEAD
git ls-remote origin refs/heads/claude/great-planck-vnvyxx
git ls-files -z -- src/ tools/ tests/ experiments/ results/ dataset/tools/
git ls-tree -r --name-only 73b30d6 experiments/
git ls-tree -d 799c2f6:experiments
rg -n 'apply_crop_check|needs_crop_check|five_crops|--untrained|--clip-arch|--clip-pretrained' src tests tools experiments results dataset/tools -g '*.py'
rg -n 'FULL_TESTS|skipif|pytest -q' .github tests README.md
rg -n 'asian_cucumber|fruit_other|t3_weak_spots|DEV_LISTS|t3ws_dev|ext3|ext4' experiments results -g '*.py'
git grep -n -E 'T3WS_LISTS|t3ws_dev|jimpax_reviewed' 73b30d6 -- experiments/
python -m ruff check --select F,E701,E702 --statistics src tools tests experiments results dataset/tools
python -m pytest -q
$env:FULL_TESTS = '1'
python -m pytest -q
python tools/baseline_card.py --check
Get-FileHash -Algorithm SHA256 model.pth, tests/golden.json, results/t9_test/t9_test.csv, results/t9_test/t9_main.csv, results/t9_test/scores_once.json
git status --porcelain=v1
```

추가로 임시 PNG 한 장과 가짜 `Classifier`를 사용해 `predict.main()`의 CSV·HTML 충돌을 실행으로 재현했다. 두 출력에 동일한 경로를 준 뒤 최종 파일이 HTML이고 입력 PNG의 바이트가 같은지 확인했다. `predict.parse_args()`는 `--hold-back-below nan --uncertain-below 2`를 받는 것도 확인했다. Ruff 0.17.6에서 여섯 폴더의 Python 101개를 AST로 읽는 데는 모두 성공했다. Ruff는 선택 규칙에서 944개를 보고했다. `src/`는 F 계열 0개, E702 4개다. `experiments/`·`results/`에는 E701·E702가 736개, F821이 132개다. 동적 `exec`로 초기화되는 이름도 포함되므로 F821 132개를 런타임 오류 건수로 해석할 수 없다.

읽기만 확인한 범위는 기존 채점과 실험의 호출 관계, 체크포인트 기록, Spaces 설정 요건이다. 외부 작업 폴더를 복원해 실험을 재실행하거나 T3~T9를 채점하지 않았다. 학습 재실행, Spaces 콜드 스타트와 처리량, 사용자 데모 페이지의 현행 동작은 **확인 못 함**이다.

## 기준별 근거

1. **가독성·복잡도 · 4/5.** 좋음 — `src/decision.py:155–179`의 `decide()`는 첫 단계의 결합·거름 → 새 채소 분류 → 감자 재확인을 순서대로 호출한다. 호출자 관점에서 어디서 답이 바뀌는지 찾기 쉽다. 개선 — `src/decision.py:111–152`에서는 새 채소의 예외와 여러 텐서가 순서 있는 튜플로 오가며, `five`·`five_unknown`의 의미를 주석까지 봐야 한다. `src/train.py:222–385`는 준비·학습·조기 종료·보정·저장이 한 함수에 들어 있어 변경 원인 추적이 어렵다.
2. **구조·결합도 · 4/5.** 좋음 — `src/predict.py:137–191`이 모델 실행을 모으고, `src/report.py:76–99`가 표시를 담당한다. `src/decision.py:167–179`는 배치 판정의 한 진입점이다. 개선 — CLI 파일 안의 `Classifier`에 초기화·이미지 변환·히트맵·유사 채소 문구가 함께 있어 웹에서 재사용할 때 경계를 정해야 한다. `src/predict.py:140–191`. 옛 실험에서 `sys.path`와 파일 위치에 의존하는 것도 결합도를 높인다. `experiments/tomato_crops/evaluate.py:24–30`.
3. **중복·경로 정리 · 3/5.** 좋음 — 추론 8종 경로 하나를 강제하고 두 CLIP 헤드의 누락을 초기화 시 알린다. `src/predict.py:137–155`, `tests/test_project.py:146–162`. 개선 — 보관 실험에서 별도 조각 함수가 있고 비슷한 `build_pool.py`가 v5~v7에 반복된다. `src/images.py:84–88`, `experiments/improve_without_photos/crops.py:28–42`, `experiments/new_vegetables/v5/build_pool.py:20–22`, `experiments/new_vegetables/v7/build_pool.py:19–21`. 발표 전에 이력을 묶어 다시 쓰는 효과는 작다.
4. **오류 처리·파일 보존 · 3/5.** 좋음 — 입력 덮어쓰기·히트맵 디렉터리를 검사하고, 개별 이미지 오류를 행으로 남겨 계속 진행한다. `src/predict.py:80–105,278–301`. T9는 전용 임시 파일과 원자적 교체를 사용한다. `results/t9_test/score.py:147–173`. 개선 — 대상 출력끼리의 충돌을 걸러내지 않는다. `src/predict.py:89–96,274–299`에서 **출력 덮어쓰기 재현**. 임계값 파싱도 범위를 확인하지 않는다. `src/predict.py:41–63`. 학습기 저장은 중간 실패 시 원본 후보를 보존하도록 설계돼 있지 않다. `src/calibrate.py:48–52`, `src/train_new_veg_head.py:111–114`.
5. **테스트·CI · 3/5.** 좋음 — 모델 체크섬과 입력 가짜 특징의 판정을 고정하고, T9 봉인·동시 실행·누락 데이터의 실패도 검사한다. `tests/test_project.py:53–103`, `tests/test_t9_score.py:84–132`. 개선 — 정상적인 예측 명령의 끝까지 가는 테스트 5개가 모두 옵션 뒤에 있다. `.github/workflows/tests.yml:8–23`, `tests/test_predict_outputs.py:59–84,152–170`, `tests/test_project.py:136–170`. 빠른 CI에는 출력 충돌과 실제 다중 산출물 회귀가 남는다.
6. **문서·운영 인계 · 4/5.** 좋음 — README가 채택된 모델과 과거 점수의 버전을 구분하고, 재현 문서가 사진 접근 한계까지 설명한다. `README.md:5–27`, `docs/training_and_evaluation.md:34–48`. 개선 — 같은 README 파일 트리의 현재 버전은 TV-12로 적혀 있다. `README.md:330–336`. PC1이 바꾼 답은 두 감자 클래스 내 CLIP 점수의 비율인데 보고서의 확률 안내는 Valid 보정 일반 설명만 한다. `src/decision.py:182–198`, `src/predict.py:235–247`.
7. **재현성·실험 이력 · 3/5.** 좋음 — 모델·데이터·결과의 해시, 시드, 설정을 하나로 묶고 테스트한다. `tools/baseline_card.py:32–48,74–92`, `tests/test_baseline_record.py:18–34`. 개선 — 첫 CLIP 분류기의 원본 학습 입력을 완전히 복원할 수 없으며, 새 채소 내보내기는 외부 `explore.py` 전체 환경에 기대고 있다. `docs/training_and_evaluation.md:44–48`, `experiments/new_vegetables/v7/export_head_data.py:5–13`. 사전 계획 일부는 사후에 문장을 지웠으므로 인용 시 원문 커밋 `73b30d6`에 연결해야 한다. `docs/codex_handoff.md:34–40`.
8. **데이터 누수 방지 · 4/5.** 좋음 — 학습 목록은 파일 바이트와 그룹 분리를 확인하고, T9에서는 T8 중복 1장을 주 판정에서 뺐다. `src/train.py:34–73`, `results/t9_test/RULES.md:65–75`. 독립 집계에서도 분할 간 동일 SHA-256·픽셀 해시·`group_id`가 0개였다. 개선 — `session_id`는 유사 장면 기준으로 중복 그룹과 다르고, Train과 Test에 **13개 세션 ID**가 걸쳐 있으며 Test **17/300장(5.7%)**이 해당한다. `dataset/README.md:102–115`, `dataset/tools/build_dataset_v3.py:130–155`, `src/train.py:66–72`. 같은 이미지가 새었다는 뜻은 아니며, 새 평가부터 세션을 분리할 근거로 쓸 수 있다.
9. **평가 엄밀함 · 4/5.** 좋음 — T9의 89장·90장, 다섯 조건, 봉인, 파일 기록을 사전에 정의하고 일회 실행을 보호한다. `results/t9_test/RULES.md:65–103`, `results/t9_test/score.py:93–99,111–121,147–173`. 개선 — 예전 채점기의 봉인은 최적화 모드에서 빠지는 `assert`이며, T3·T4에는 고정 결과 파일을 직접 쓰는 경로가 있다. `results/t3_test/score.py:46–50,121–139`, `results/t4_test/score.py:42–46,95–122`. T9 집계의 신뢰성을 그보다 오래된 도구에 그대로 적용할 수 없다.
10. **학습·추론 일치 · 4/5.** 좋음 — 전처리 함수를 공유하고 체크포인트 클래스 순서와 전처리 설정을 검증한다. 새 CLIP 분류기 학습에서도 추론의 첫 단계를 호출한다. `src/images.py:23–36`, `src/networks.py:111–125`, `src/train_new_veg_head.py:41–44`. 개발 평가의 캐시도 가중치·전처리·CLIP 입력이 다르면 거절한다. `tools/compare_on_dev.py:148–158`. 개선 — 아래 옛 평가 경로는 버전 명세와 실제 체크포인트의 일치를 확인하지 않는다. `results/t9_test/score.py:191–214`, `results/user_photos_2/score.py:28–38`.

### 옛 채점기의 TV-12 표기

- **TV-13 파일을 TV-12로 넣는 코드 경로**를 `results/t4_test/score.py:101–105`와 `results/t4_test/score_reference.py:36–39`, `results/t5_test/score.py:95–101`, `results/t6_test/score.py:130–141`, `results/t7_test/score.py:47–57`, `results/t8_test/score.py:90–100`, `results/t9_test/score.py:191–214`, `results/user_photos_2/score.py:28–38`에서 확인했다. 일부는 삭제된 입력이나 외부 파일 때문에 더 일찍 막힐 수 있다.
- **T9 본 채점은 이미 저장된 결과 때문에 방어 장치가 막는다.** `--rehearsal`은 이 장치를 쓰지 않고 현재 파일을 로드하므로 TV-12와 PC1 비교가 무의미해질 수 있다. 저장된 결과 자체의 문제라는 증거는 없다. T3는 별도 TV-11/ZS8 계산을 직접 사용하므로 같은 TV-12 오표기 목록으로 세지 않았다. `results/t3_test/score.py:87–118,125–137`.
- 현행 개발 비교에서 참고할 예가 이미 있다. `experiments/potato_check/evaluate.py:58–66`는 현재 파일에서 `potato_check`를 빼서 TV-12를 비교하고, `tests/test_baseline_record.py:29–34`에서 과거 개발 정답과 대조한다. 이 방식의 확인 범위는 TV-12와 TV-13 사이의 이 변경에 한정된다. 예전 버전 전체의 복원 방식으로 쓰면 안 된다.

## 고칠 것

아래는 **제안**이다. 변경을 맡는 Claude는 사용자가 선택한 항목부터 처리한다. 작업량은 저장소에 익숙한 개발자 기준의 상대 크기다. 예측 경로나 비즈니스 판정 로직을 건드리는 모든 제안은 **기존 골든 예측과 T9 기록의 동일성**을 합격 조건으로 둔다. `FULL_TESTS=1 python -m pytest -q`와 `python tools/baseline_card.py --check`를 돌리고, T9 JSON의 SHA-256이 `a151957f4b6db7f1819d23c4764c46fc91cddf37e19a1b927afba811ac9cee84`인지 비교한다. **T9를 다시 채점하지 않는다.** 실제 판정을 바꾸는 개선은 별도 후보·새 블라인드 시험으로 평가한다. 아래에 추가할 테스트나 앱 파일은 구현 후의 예시다.

### 발표 전에 꼭

| 무엇·어디 | 어떻게 | 확인 명령 | 작업량 · 판정 위험 |
|---|---|---|---|
| 발표 자료와 최신 실행을 일치. `README.md:281–299,330–336`, `docs/training_and_evaluation.md` | 파일 트리의 TV-12를 TV-13으로, 과거 점수의 버전은 명시 유지. 캐시 준비→TV-13 실행→대표 사진의 CSV·HTML·히트맵 확인을 짧게 적는다. PC1 확률과 판단 보류를 화면 설명에 반영한다. | `python tools/baseline_card.py --check`, `python src/predict.py --model model.pth --input DEMO_PHOTO --output TMP/demo.csv --report TMP/demo.html --heatmap TMP/heatmaps`. 발표용 사진과 임시 출력 사용. | 작음 · 없음 |
| 예측 출력끼리의 충돌과 임계값 검사. `src/predict.py:41–63,80–105,256–299` | 쓰기 전에 기존 `same_file()`을 재사용해 CSV·HTML 대상의 동일 파일을 거부한다. 입력·모델 경로도 쓰기 대상에서 보호하고, 임계값은 유한한 0~1만 받는다. | `python -m pytest -q tests/test_predict_outputs.py`. 동일·별칭 경로, NaN/범위 초과, 유효 임계값 회귀 추가. | 작음 · 낮음. 정상 인자는 같은 판정 |
| CLIP 다운로드 없이 빠른 CLI 회귀 1개. `tests/test_predict_outputs.py`, `.github/workflows/tests.yml` | 가짜 판정만 주입하고 실제 인자·이미지 로딩·CSV·HTML 처리와 종료 코드를 확인한다. 계산 흐름의 골든은 기존 전체 실행에서 유지한다. | `python -m pytest -q tests/test_predict_outputs.py`, `python -m pytest -q`. CLIP 없는 기존 빠른 Actions에서도 새 테스트 실행을 확인. | 작음 · 없음 |

### 웹 호스팅 전에

| 무엇·어디 | 어떻게 | 확인 명령 | 작업량 · 판정 위험 |
|---|---|---|---|
| Spaces 추론 경계와 전용 환경. `src/predict.py:137–191`, `src/images.py:75–81`, `requirements.txt`, Space의 앱 진입점 | 기존 판정과 전처리를 재사용한다. 모델을 한 번 로드하고 업로드 크기·픽셀·형식, 요청별 임시 출력과 파일 정리, 명시적인 준비·실패 상태를 정한다. 앱 실행에 필요한 패키지만 고정하고 TV-13 모델 버전을 표시한다. | `python -m pytest -q tests/test_app.py`(추가 예정), `python app.py`(추가 예정), 콜드/웜 시작과 손상·대형 업로드·동시 요청 스모크, `FULL_TESTS=1 python -m pytest -q`. | 중간 · 중간. 전처리·판정이 달라질 수 있어 골든/T9 동일 필수 |
| 전체 예측 테스트 자동화. `.github/workflows/tests.yml:8–23`, `tests/test_project.py:136–171` | 빠른 작업은 유지하고, CLIP 버전을 고정·캐시한 릴리스/수동 전체 작업을 추가한다. Spaces에서 첫 다운로드와 오프라인 캐시를 둘 다 검증한다. | 새 Actions 작업의 `FULL_TESTS=1 python -m pytest -q`, `python tools/baseline_card.py --check`. 실행 시간·메모리는 기록하고 이 감사에서 미측정으로 둔다. | 중간 · 없음 |
| 과거 평가의 버전·출력 보호. `results/t4_test/score.py` 등 T4~T9 및 각 결과 기록, `results/user_photos_2/score.py`, `docs/training_and_evaluation.md` | 역사적 명령의 실행 가능 범위를 표시한다. 재사용이 필요한 개발 모드는 기준 체크포인트의 버전·해시를 명시적으로 검사하고, 리허설은 별도 경로만 사용한다. 이미 끝난 본 채점은 공식 기록과 함께 보존한다. | `python -m pytest -q tests/test_t9_score.py tests/test_baseline_record.py`, 다른 버전의 가짜 체크포인트를 초기에 거부하는 테스트, `python tools/baseline_card.py --check`. | 중간 · 낮음. 프로덕션 판정과 기존 파일은 동일 |

### 나중에

| 무엇·어디 | 어떻게 | 확인 명령 | 작업량 · 판정 위험 |
|---|---|---|---|
| 학습 진입점 분리와 저장 보호. `src/train.py:222–385`, `src/calibrate.py:48–52`, `src/train_clip_head.py:171–176`, `src/train_new_veg_head.py:111–114` | 준비·한 에폭 학습/검증·보정·산출물 저장 경계를 추출한다. 후보 출력은 임시 파일 확정 방식으로 바꾸고 기존 인자를 검증한다. | `python -m pytest -q tests/test_manifest_checks.py tests/test_project.py`, 가짜 작은 데이터와 임시 출력에 대한 1 epoch 저장·다시 로드 시험, 골든/T9 기록 비교. | 큼 · 중간. 학습 순서가 바뀌면 후보 가중치가 달라질 수 있음 |
| 보관 실험 의존성과 호환 옵션 정리. `experiments/README.md`, 위 4개 직접 경로 및 7개 외부 의존 파일, `src/decision.py:220–244`, `src/images.py:84–88`, 두 CLIP 학습 파일 | 실제로 유지할 재현 경로만 명시 입력으로 전환하고, 보관 스크립트는 해당 커밋·필수 입력을 적는다. 조각 검사와 옵션은 사용처와 함께 이전/제거한다. `exec` 의존은 명시적인 함수 호출로 바꾼다. | 위 `rg` 호출 검색, `python -m pytest -q`, `FULL_TESTS=1 python -m pytest -q`. 재현을 유지할 경로만 개발 자료로 스모크한다. | 큼 · 중간. 확정 전에 골든/T9 기록 동일 확인 |
| 새 데이터 평가의 독립성 강화. `dataset/tools/build_dataset_v3.py`, `src/train.py:66–73`, 다음 평가 규칙 | 이후 데이터 빌드에 촬영자 또는 검증한 세션 키를 도입하고, 새 블라인드 목록을 후보 선택과 분리한다. 기존 분할·봉인은 이력으로 보존한다. | `python -m pytest -q tests/test_manifest_checks.py tests/test_compare_dataset_builds.py tests/test_sweetpotato_split.py`, 임시 다음 버전에서 교차 그룹 0 확인. | 중간 · 높음. 새 학습 데이터는 다른 판정을 만들 수 있음 |
