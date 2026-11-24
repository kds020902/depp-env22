# 웹 배포 (Hugging Face Spaces)

웹 페이지는 `app.py`와 `web.css`로 구성됩니다. 사진을 올린 뒤 **판정하기**를 누르면 `images.open_image()`로 열어 `predict.Classifier`에 넘깁니다. 판정은 `python src/predict.py`와 같습니다.

## 내 컴퓨터에서 먼저 보기

```bash
python -m pip install -r requirements.txt gradio==6.32.0
python app.py          # http://127.0.0.1:7860
```

처음 실행할 때 CLIP 가중치(약 600MB)를 받습니다.

## Space에 올리기

1. Space에 올릴 폴더를 만듭니다. 새 폴더 이름을 주어야 합니다.

   ```bash
   python tools/build_space.py ../veg-space
   ```

   | 들어가는 것 | 출처 |
   |---|---|
   | `app.py`, `web.css`, `model.pth` | 저장소 그대로 |
   | `src/`의 8개 파일 | 예측에 필요한 것만 |
   | `README.md` | `space/README.md` (Space 설정이 맨 위에 있음) |
   | `requirements.txt` | `space/requirements.txt` (기록된 결과를 만든 버전으로 고정) |

2. huggingface.co에서 **New Space**를 만듭니다. SDK는 **Gradio**, 공개 범위는 **Public**입니다.
3. 만든 폴더의 파일을 Space의 **Files → Add file → Upload files**로 올립니다. `src` 폴더는 폴더째 끌어다 놓습니다.
4. **App** 탭에서 빌드가 끝나기를 기다립니다. 처음에는 패키지 설치와 CLIP 가중치 내려받기로 몇 분이 걸립니다.

토큰이나 비밀번호를 저장소나 대화에 적지 않습니다. 올리기는 로그인한 브라우저에서 직접 합니다.

## 올린 뒤 확인할 것

| 사진 | 기대하는 답 |
|---|---|
| `dataset/test/P_fresh/cucumber_fresh_0017.jpg` | 오이 정상, 100% |
| `dataset/test/N_rotten/cucumber_rotten_0142.jpg` | 오이 비정상, 99%, 빨간 지도 있음 |
| `dataset/test/P_fresh/potato_fresh_0201.jpg` | 감자 정상, 99% |
| 채소가 아닌 사진 (예: 고양이) | 학습된 데이터가 아닙니다 |

확률이 위와 1%p 넘게 다르면 패키지 버전이 다른 것입니다. `requirements.txt`의 버전이 그대로 설치됐는지 빌드 기록에서 확인합니다.

## 알아둘 점

- **요금제**: 2026-11-24에 읽은 Hugging Face 문서에는 Gradio Space를 만들려면 유료 요금제가 필요하고, 무료 개인 계정은 ZeroGPU로 도는 Gradio Space를 2개까지 둘 수 있다고 적혀 있습니다. 이미 다른 Space가 있으면 남은 자리를 먼저 확인합니다. ZeroGPU 하드웨어에서 이 페이지가 그대로 도는지는 확인하지 못했습니다.
- **자원**: 이 컨테이너(CPU)에서 잰 값은 시작 약 30초(가중치를 받은 뒤), 사진 한 장 0.2~0.7초, 메모리 최대 약 1.5GB입니다. 문서의 CPU Basic은 2 vCPU · 16GB입니다.
- **잠들기**: 무료 하드웨어의 Space는 한동안 쓰지 않으면 멈추고, 다시 열 때 시작 시간이 걸립니다. 발표 직전에 한 번 열어 둡니다.
- **사진**: 올린 사진은 서버의 임시 파일로만 있다가 약 10분 뒤 지워집니다. 20MB나 4천만 화소를 넘는 사진은 받지 않습니다.
- **동시 접속**: 한 번에 한 장씩 처리하고 나머지는 줄을 섭니다(최대 20명).
- **모델을 바꾸면** 폴더를 다시 만들어 다시 올립니다. Space의 파일을 직접 고치지 않습니다.
