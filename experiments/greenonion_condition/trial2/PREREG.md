# 대파 상태 학습 2차 시험 (개발 단계만, 사전 등록 — 1차 결과를 본 뒤 정함)

2026-11-28. 결과를 보기 전에 커밋한다. **최종 시험용은 열지 않는다.** 1차(`../trial1`) 결과를 보고 정한 두 번째 후보임을 밝힌다.

## 1차에서 본 것

- G1·G2 통과, G3 실패: 대파로 알아본 상한 대파 62장 중 9장만 "상함".
- 원인: 학습기가 `labels.UNTRAINED_CLASSES`(greenonion_rotten)를 헤드에 그대로 적어, 상한 대파를 학습한 뒤에도
  `decision.zero_shot_logits`가 15번 칸의 글 설명 점수를 0으로 둔다. 14번(정상 대파)은 점수를 받으므로 정상 쪽으로 기운다.

## 바꾸는 것 하나

- 학습할 때 greenonion_rotten을 "학습한 칸"으로 둔다(헤드의 `untrained_classes`에서 뺀다).
  15번 칸은 text16에 이미 있는 "a photo of rotten green onions." 점수를 받는다.
- 방법: 학습기 코드는 바꾸지 않고, `train15.py`가 학습기 모듈의 `UNTRAINED_CLASSES`를 빈 집합으로 바꾼 뒤 그대로 실행한다
  ("아님" 기준·τ·온도 보정이 바뀐 점수로 다시 계산된다).
- 자료·사진 목록·특징(`../trial1/photos.csv`, `features.npz`)·교차검증 접힘·결정 규칙·관문 G1·G2·G3은 1차와 같다.

## 미리 밝혀 둘 점

- 개발 자료를 두 번 본다. 최종 성능은 시험용 37장(정리 후)으로만 말한다.
- 이 시험으로 `model.pth`, `labels.py`, 앱은 바꾸지 않는다. 채택하려면 별도로 `labels.UNTRAINED_CLASSES`·`NO_CONDITION_SPECIES`와 테스트·문서를 함께 고친다.
