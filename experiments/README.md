# 실험 색인

실험 폴더 29개를 주제별로 한 줄씩 정리했다. 폴더마다 계획(PREREG)과 결과가 README에 있다.
채택된 것은 `model.pth`의 버전(TV)을 올렸다(맨 위 [README](../README.md)의 버전 표).

**스크립트를 다시 돌리려면:**
- 사진은 라이선스 때문에 저장소 밖 작업 폴더에 있다.
- 그 폴더를 환경변수 `DEPP_WORK`로 알려 준다(예: `DEPP_WORK=/data/depp_work python experiments/.../evaluate.py`).
- 사진 출처와 받는 방법은 [data_sources.md](../data_sources.md)에 있다.

## 판정 구조 (무엇이 지금 시스템을 만들었나)

| 폴더 | 한 것 | 결과 |
|---|---|---|
| [real_scene_threshold](real_scene_threshold/README.md) | 거르기 기준을 실제 장면 채소로 다시 맞춤 | 채택 TV-5 |
| [clip_classifier](clip_classifier/README.md) | CLIP 특징 분류기를 채소 모델과 함께 씀 | 채택 TV-6 |
| [not_five_class](not_five_class/README.md) | CLIP 분류기에 "다섯 채소가 아님"을 직접 가르침 | 채택 TV-7 |
| [clip_only_reject](clip_only_reject/README.md) | 거르기를 CLIP 기준 하나로 | 채택 TV-8 |
| [new_vegetables](new_vegetables/README.md) | 양파·고구마·대파 추가(8종), 글 설명 섞기 | 채택 TV-9, TV-10 |
| [watermark](watermark/README.md) | 워터마크 지름길 줄이기(가짜 워터마크로 재학습) | 채택 TV-11 |
| [zero_shot_reject](zero_shot_reject/README.md) | CLIP 글 설명으로 거르기 보강 | 채택 TV-12 (지금) |
| [fusion_weight](fusion_weight/README.md) | 두 모델을 합치는 비중 다시 정하기 | 채택 안 함 |
| [clip_gate](clip_gate/README.md) | CLIP 문지기("다섯 채소인가?"만 묻기) | 채택 안 함 |
| [tta](tta/README.md) | 뒤집기 평균(TTA) | 채택 안 함 |
| [posthoc_literature](posthoc_literature/README.md) | 논문 방법을 재학습 없이 적용 | 채택 안 함 |
| [improve_without_photos](improve_without_photos/README.md) | 글 설명 점수 섞기, 조각 보기, 판단 보류 | 2·3 채택 안 함, 판단 보류는 predict.py 기본값으로 반영 |
| [bigger_eye](bigger_eye/README.md) | CLIP 자리에 더 좋은 이미지 모델 넣기 (B급 2개, L급 1개) | 반영 안 함 (L1 개발 +20, T5로 다시 확인 예정) |
| [shape_and_parts](shape_and_parts/README.md) | Test 오류 두 종류 고치기: 종류는 CLIP 글로(A), 상한 조각이면 보류(B) | 둘 다 반영 안 함 (A −2, B는 오답 2장 잡고 맞는 답 16장 보류) |

## 학습 사진 늘리기·고치기

| 폴더 | 한 것 | 결과 |
|---|---|---|
| [10class_tuning](10class_tuning/README.md) | 10개 클래스 모델 추가 학습 | 채택 안 함 |
| [more_data](more_data/README.md) | 학습 데이터 크게 늘리기 | 채택 안 함 |
| [pepper_web](pepper_web/README.md) | 피망 웹 사진 추가 | 채택 안 함 |
| [partial_rot](partial_rot/README.md) | 일부만 상한 채소 잡기 | 추가 실험 |
| [field_tomato](field_tomato/README.md) | 밭 토마토 사진 추가 학습 | 추가 실험 |
| [rotten_phone](rotten_phone/README.md) | 휴대폰으로 찍은 정상·상한 사진 추가 | 채택 안 함 |
| [fix_carrot_cucumber](fix_carrot_cucumber/README.md) | 흙 묻은 당근·쪼그라든 오이 오답 고치기 | 채택 안 함 |
| [asian_cucumber](asian_cucumber/README.md) | 아시아 오이로 학습하기 | 채택 안 함 (사진은 이후 학습에 포함) |
| [clip_head_more_data](clip_head_more_data/README.md) | CLIP 분류기를 더 다양한 사진으로 재학습 | 채택 안 함 (1장 차이) |
| [weak_spots](weak_spots/README.md) | 약점 사진으로 CLIP 분류기 재학습 | 채택 안 함 |
| [t3_weak_spots](t3_weak_spots/README.md) | T3에서 보인 약점 고치기(후보 6개) | 채택 안 함 |
| [balanced_clip_head](balanced_clip_head/README.md) | 정상:상함 1:1로 사진 수 맞추기 | 채택 안 함 (점수 그대로) |

## 과일·다른 것 거르기

| 폴더 | 한 것 | 결과 |
|---|---|---|
| [fruit_other](fruit_other/README.md) | 과일을 "기타" 클래스로 학습 | 채택 안 함 |
| [fruit_gate](fruit_gate/README.md) | 과일 판별 모델을 앞에 따로 두기 | 채택 안 함 |
| [fruit_farm_tomato](fruit_farm_tomato/README.md) | 농가 토마토와 상한 과일 가르치기 | 채택 안 함 |

## 평가 점검

| 폴더 | 한 것 | 결과 |
|---|---|---|
| [session_overlap](session_overlap/README.md) | Train–Test 같은 촬영 세션 겹침 측정 | 측정 (Test 해석에 반영) |

## 실험들에서 반복해서 나온 교훈
- **웹 사진을 더 넣는 것은 거의 효과가 없었다:** more_data, pepper_web, weak_spots, t3_weak_spots, balanced_clip_head 모두 같았다.
- **채소 모델을 인터넷 사진으로 다시 학습하면 밭·휴대폰 사진이 떨어졌다(세 번):** 개선은 주로 CLIP 쪽 판정 구조에서 나왔다.
- **상한 과일을 "채소 아님"으로 가르치면 진짜 상한 채소도 같이 거른다(두 번).**
- **Test 300과 새 인터넷 241은 여러 번 썼다:** 실제 성능은 처음 보는 시험지(T3)로 본다([results/scorecard](../results/scorecard/README.md)).
