# CLIP 문지기, 결과를 보기 전에 적음 (2026-10-11)

## 목표

- 사용자 목표는 두 가지다. 과일 이름을 맞힐 필요는 없다.
    - 오이·감자·토마토·피망·당근 사진은 놓치지 않는다.
    - 그 밖의 사진에는 "이 다섯 채소가 아닙니다"라고 답한다.
- 지금 채소 모델(B)은 다섯 채소만 봐서, 채소와 닮은 과일을 "처음 보는 것"으로 알아보지 못한다.
- 그래서 인터넷 사진 수억 장으로 학습한 CLIP(Radford et al. 2021)을 앞에 두고 "다섯 채소 중 하나인가?"만 묻는다.

## 판정 순서

1. CLIP 문지기가 "다섯 채소가 아니다"라고 보면 거른다.
2. 통과한 사진은 지금과 똑같이 B가 답한다. 실제 장면 채소로 맞춘 energy 기준(−1.962)도 그대로 쓴다.

B는 바꾸지 않으므로 채소 종류·상태 판정은 그대로다. 바뀌는 것은 문지기가 거르는 사진뿐이다.

## 후보 4개 (CLIP 2종 × 방법 2종)

**CLIP 모델**
- OpenAI ViT-B/16
- DataComp-XL ViT-B/16 (open_clip `datacomp_xl_s13b_b90k`)

**방법 Z: 글자로 묻기 (zero-shot, 학습 없음)**
- 사진과 각 단어 묶음의 유사도에 softmax(CLIP 기본 배율 100)를 한다.
- 다섯 채소 단어의 확률 합을 "다섯 채소일 확률"로 쓴다.
- 단어 묶음은 아래와 같이 지금 정하고 바꾸지 않는다.
    - **다섯 채소:** cucumber, potato, tomato, bell pepper, carrot
    - **과일 40개:** apple, pear, Asian pear, orange, mandarin, tangerine, lemon, lime, grapefruit, yuzu, persimmon, peach, nectarine, apricot, plum, cherry, strawberry, raspberry, blueberry, blackberry, grape, kiwi, banana, mango, papaya, pineapple, watermelon, melon, Korean melon, cantaloupe, pomegranate, fig, jujube, date, coconut, avocado, lychee, passion fruit, guava, dragon fruit
    - **다른 채소 26개:** onion, garlic, cabbage, lettuce, broccoli, cauliflower, eggplant, zucchini, pumpkin, squash, sweet potato, radish, daikon, beet, turnip, ginger, mushroom, corn, green beans, peas, spinach, celery, leek, chili pepper, okra, asparagus
    - **문장:** 위 단어들은 다섯 문장의 평균을 쓴다. 문장은 "a photo of a {}.", "a close-up photo of a {}.", "a photo of a fresh {}.", "a photo of a rotten {}.", "a photo of {}s."이다.
    - **그 밖 19개 ("a photo of {}." 한 문장):** a person, an animal, a vehicle, a building, a room, furniture, a plate of food, a dish, a pizza, a salad, a sandwich, a cake, a bottle, a toy, a plant, a flower, a tree, text, a landscape

**방법 P: 사진으로 가르치기 (선형 분류기)**
- CLIP 이미지 특징(L2 정규화)에 로지스틱 회귀를 학습한다. 클래스 가중치는 균형, C=1이며 조정하지 않는다.
- 학습 사진은 과일 문지기 2차의 학습 목록을 그대로 쓴다.
    - 채소 3,147장
    - 과일 2,667장
    - 확인용 사진의 복사본은 이미 뺀 목록이다.
- 점수는 "다섯 채소일 확률"이다.

## 기준값 (후보마다 같은 규칙)

- 기준값은 Valid 채소 300장과 LVIS 실제 장면 채소 192장을 합친 492장 중 99%가 통과하는 값이다.
- 즉 문지기가 실제 채소를 1% 넘게 거르지 않게 한다.

## 후보 고르기 (개발용 사진만 사용)

- LVIS 실제 장면 과일 192장 중 위 기준으로 거른 비율이 가장 높은 후보 하나를 고른다. 이 사진은 확인용 과일과 같은 사진이 아니고, 채소 모델과 방법 P 학습에도 쓰지 않았다.
- 비율이 같으면 AUROC로 고른다. AUROC는 Valid+LVIS 채소 492장과 개발용 과일(LVIS 과일 192, Valid 과일 30, Kaggle 과일 개발용 절반 200)을 가르는 정도다.
- 확인용 사진은 후보를 고른 뒤 한 번만 본다.

## 판정 규칙 (지금 방식과 비교)

지금 방식은 B와 기준 −1.962의 조합이다.

| | 지금 | 통과 기준 |
|---|---:|---:|
| 인터넷 채소 221장 맞게 남긴 수 | 158 | 153 이상 |
| 160장 시험지 맞게 남긴 수 | 146 | 142 이상 |
| 아시아 오이 37장 맞게 남긴 수 | 29 | 27 이상 |
| Valid 300장 맞게 남긴 수 | 278 | 272 이상 |
| 인터넷 과일 374장 중 거른 수 | 101 | **233 이상** (예전 Valid 기준 수준) |
| 일상 사진 400장 중 거른 수 | 311 | 311 이상 (문지기는 더 거르기만 하므로 저절로 통과) |

- 위를 모두 넘으면 Test 300장을 한 번 채점한다. Test 맞게 남긴 수가 269장 이상(지금 272장)이면 채택한다.
- 함께 적을 숫자: Kaggle 과일 채점용 절반, 데모 과일 예시 5장, 문지기가 거른 채소 사진 목록.

## 미리 밝혀 둘 점

- 확인용 사진(인터넷 채소·과일, 160장 시험지, 아시아 오이)은 이전 실험에서 여러 번 봤다. 이번에는 후보 4개 중 하나를 개발용 사진만으로 고르고 확인용 사진은 한 번만 보지만, 완전히 새 시험지는 아니다.
- 과일 단어 목록에는 감·귤·참외·대추처럼 한국에서 흔한 과일이 들어 있다. 확인용 과일 374장에도 이 과일들이 있다. 목록은 일반적인 과일 이름이고 사진을 보고 고르지 않았다.
