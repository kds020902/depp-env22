"""Class, vegetable and condition names (class id = 2 x vegetable + condition)."""

# Class id = 2 x species + condition. New species are appended, so older 6-class checkpoints keep their ids.
CLASS_NAMES = [
    "cucumber_fresh", "cucumber_rotten", "potato_fresh",
    "potato_rotten", "tomato_fresh", "tomato_rotten",
    "bellpepper_fresh", "bellpepper_rotten", "carrot_fresh", "carrot_rotten",
    "onion_fresh", "onion_rotten", "sweetpotato_fresh", "sweetpotato_rotten", "greenonion_fresh", "greenonion_rotten",
]
SPECIES_NAMES = ["cucumber", "potato", "tomato", "bellpepper", "carrot", "onion", "sweetpotato", "greenonion"]
# No public photos of rotten green onions were found: the slot keeps class ids aligned but has no training photos,
# so green onions are recognised by kind only and their condition is not judged.
UNTRAINED_CLASSES = {"greenonion_rotten"}
NO_CONDITION_SPECIES = {"greenonion"}
CONDITION_NAMES = ["fresh", "rotten"]
UNKNOWN_MESSAGE = "학습된 데이터가 아닙니다"
KOREAN = {"cucumber": "오이", "potato": "감자", "tomato": "토마토", "bellpepper": "피망", "carrot": "당근",
          "onion": "양파", "sweetpotato": "고구마", "greenonion": "대파",
          "fresh": "정상", "rotten": "비정상", "not_judged": "상태 판정 안 함"}


def condition_of(index: int) -> str:
    """fresh / rotten for a class id; not_judged for a vegetable whose condition the model was not trained to judge."""
    return "not_judged" if SPECIES_NAMES[index // 2] in NO_CONDITION_SPECIES else CONDITION_NAMES[index % 2]
