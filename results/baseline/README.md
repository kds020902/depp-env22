# 기준 모델 기록

| 파일 | 내용 |
|---|---|
| `tv13.json` | 지금 `model.pth`(TV-13)의 기록: 파일 해시, 분할, 시드, 설정, 판정 기준, 지표. `python tools/baseline_card.py`가 만들고 테스트가 확인한다 |
| `tv13_dev.json` | 개발 사진 1,935장에서 TV-13의 답. `python tools/compare_on_dev.py results/baseline/tv13_dev.json` |
| `tv12.json` | TV-12일 때의 기록(2026-11-22 ~ 11-23). 그때의 `model.pth` 해시가 들어 있다. 더 고치지 않는다 |
| `tv12_dev.json` | 개발 사진에서 TV-12의 답. `model.pth`에서 감자 재확인을 빼면 이 답이 그대로 나오는 것을 테스트가 확인한다 |

- TV-13은 TV-12와 가중치가 같다. 체크포인트에 `potato_check`가 더 있다(`experiments/potato_check`).
- TV-12의 가중치 파일은 커밋 `9ebb286` ~ TV-13 채택 커밋 직전의 `model.pth`다.
