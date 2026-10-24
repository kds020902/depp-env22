"""Closest-thing hint for photos turned away as "학습된 데이터가 아닙니다" (optional; needs `pip install open_clip_torch`).

CLIP (open_clip ViT-B-16, DataComp-XL weights, about 600 MB, downloaded on first use) compares the photo with 90 words:
the five vegetables, 40 fruits, 26 other vegetables and 19 other things, and names the closest one. It only adds a hint;
which photos are turned away is still decided by the vegetable model. The word list is the one of experiments/clip_gate.
"""

from __future__ import annotations

import torch

ARCH, PRETRAINED = "ViT-B-16", "datacomp_xl_s13b_b90k"
TEMPLATES = ["a photo of a {}.", "a close-up photo of a {}.", "a photo of a fresh {}.", "a photo of a rotten {}.",
             "a photo of {}s."]
TRAINED = {"cucumber": "오이", "potato": "감자", "tomato": "토마토", "bell pepper": "피망", "carrot": "당근"}
FRUITS = {
    "apple": "사과", "pear": "서양배", "Asian pear": "배", "orange": "오렌지", "mandarin": "귤", "tangerine": "귤",
    "lemon": "레몬", "lime": "라임", "grapefruit": "자몽", "yuzu": "유자", "persimmon": "감", "peach": "복숭아",
    "nectarine": "천도복숭아", "apricot": "살구", "plum": "자두", "cherry": "체리", "strawberry": "딸기",
    "raspberry": "라즈베리", "blueberry": "블루베리", "blackberry": "블랙베리", "grape": "포도", "kiwi": "키위",
    "banana": "바나나", "mango": "망고", "papaya": "파파야", "pineapple": "파인애플", "watermelon": "수박",
    "melon": "멜론", "Korean melon": "참외", "cantaloupe": "캔털루프 멜론", "pomegranate": "석류", "fig": "무화과",
    "jujube": "대추", "date": "대추야자", "coconut": "코코넛", "avocado": "아보카도", "lychee": "리치",
    "passion fruit": "패션프루트", "guava": "구아바", "dragon fruit": "용과",
}
OTHER_VEGETABLES = {
    "onion": "양파", "garlic": "마늘", "cabbage": "양배추", "lettuce": "상추", "broccoli": "브로콜리",
    "cauliflower": "콜리플라워", "eggplant": "가지", "zucchini": "애호박", "pumpkin": "호박", "squash": "호박",
    "sweet potato": "고구마", "radish": "무", "daikon": "무", "beet": "비트", "turnip": "순무", "ginger": "생강",
    "mushroom": "버섯", "corn": "옥수수", "green beans": "껍질콩", "peas": "완두콩", "spinach": "시금치",
    "celery": "셀러리", "leek": "대파", "chili pepper": "고추", "okra": "오크라", "asparagus": "아스파라거스",
}
OTHER_THINGS = {
    "a person": "사람", "an animal": "동물", "a vehicle": "탈것", "a building": "건물", "a room": "실내",
    "furniture": "가구", "a plate of food": "음식 접시", "a dish": "요리", "a pizza": "피자", "a salad": "샐러드",
    "a sandwich": "샌드위치", "a cake": "케이크", "a bottle": "병", "a toy": "장난감", "a plant": "식물",
    "a flower": "꽃", "a tree": "나무", "text": "글자", "a landscape": "풍경",
}
PRODUCE = {**TRAINED, **FRUITS, **OTHER_VEGETABLES}
NAMES = list(PRODUCE) + list(OTHER_THINGS)
KOREAN = {**PRODUCE, **OTHER_THINGS}


def _final_consonant(word: str) -> int | None:
    code = ord(word[-1]) - 0xAC00
    return code % 28 if 0 <= code < 11172 else None


def _with_wa(word: str) -> str:
    """Attach 와/과 to a Korean word (와 after a vowel)."""
    final = _final_consonant(word)
    return word + ("와(과)" if final is None else "와" if final == 0 else "과")


def _with_ro(word: str) -> str:
    """Attach 으로/로 to a Korean word (로 after a vowel or ㄹ)."""
    final = _final_consonant(word)
    return word + ("(으)로" if final is None else "로" if final in (0, 8) else "으로")


class ClipHint:
    def __init__(self, device: str = "cpu"):
        import open_clip  # optional dependency
        self.device = device
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(ARCH, pretrained=PRETRAINED, device=device)
        self.model.eval()
        tokenizer = open_clip.get_tokenizer(ARCH)
        with torch.inference_mode():
            def encode(texts):
                return torch.nn.functional.normalize(self.model.encode_text(tokenizer(texts).to(device)), dim=-1)
            rows = [torch.nn.functional.normalize(encode([t.format(n) for t in TEMPLATES]).mean(0), dim=0)
                    for n in PRODUCE]
            rows += list(encode([f"a photo of {n}." for n in OTHER_THINGS]))
        self.text = torch.stack(rows)

    @torch.inference_mode()
    def closest(self, image) -> tuple[str, float]:
        """The closest of the 90 words (English) and its share of the softmax."""
        x = self.preprocess(image).unsqueeze(0).to(self.device)
        features = torch.nn.functional.normalize(self.model.encode_image(x), dim=-1)
        p = (100 * features @ self.text.T).softmax(1)[0]
        return NAMES[int(p.argmax())], float(p.max())

    def message(self, image) -> tuple[str, str]:
        """(closest word in Korean, hint sentence) for a photo the vegetable model turned away."""
        name, _ = self.closest(image)
        korean = KOREAN[name]
        if name in TRAINED:
            return korean, f"{_with_wa(korean)} 가장 비슷하지만 모델이 확신하지 못했습니다. 채소 하나만 가까이에서 다시 찍어 보세요."
        return korean, f"{_with_ro(korean)} 보입니다."
