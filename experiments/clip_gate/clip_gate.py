"""CLIP gate: asks only "is this one of the five vegetables (cucumber, potato, tomato, bell pepper, carrot)?".

    python experiments/clip_gate/clip_gate.py photo.jpg ...

Zero-shot, no training: the photo is compared with text for the five vegetables and for 85 other things (40 fruits,
26 other vegetables, 19 everyday things); the summed softmax probability of the five vegetables is the gate score.
A photo is turned away when the score is below THRESHOLD, the value that still passes 99% of the 300 Valid vegetables and
the 192 LVIS real-scene vegetables (experiments/real_scene_threshold/lvis_vegetables.csv).
Model: open_clip ViT-B-16, pretrained "datacomp_xl_s13b_b90k" (downloaded from the Hugging Face Hub on first use).
"""

import sys
from pathlib import Path

import open_clip
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from model_utils import open_image  # noqa: E402

ARCH, PRETRAINED = "ViT-B-16", "datacomp_xl_s13b_b90k"
THRESHOLD = 0.005768541246652603
POS = ["cucumber", "potato", "tomato", "bell pepper", "carrot"]
FRUIT = ["apple", "pear", "Asian pear", "orange", "mandarin", "tangerine", "lemon", "lime", "grapefruit", "yuzu", "persimmon",
         "peach", "nectarine", "apricot", "plum", "cherry", "strawberry", "raspberry", "blueberry", "blackberry", "grape", "kiwi",
         "banana", "mango", "papaya", "pineapple", "watermelon", "melon", "Korean melon", "cantaloupe", "pomegranate", "fig",
         "jujube", "date", "coconut", "avocado", "lychee", "passion fruit", "guava", "dragon fruit"]
VEG = ["onion", "garlic", "cabbage", "lettuce", "broccoli", "cauliflower", "eggplant", "zucchini", "pumpkin", "squash",
       "sweet potato", "radish", "daikon", "beet", "turnip", "ginger", "mushroom", "corn", "green beans", "peas", "spinach",
       "celery", "leek", "chili pepper", "okra", "asparagus"]
GEN = ["a person", "an animal", "a vehicle", "a building", "a room", "furniture", "a plate of food", "a dish", "a pizza",
       "a salad", "a sandwich", "a cake", "a bottle", "a toy", "a plant", "a flower", "a tree", "text", "a landscape"]
TEMPLATES = ["a photo of a {}.", "a close-up photo of a {}.", "a photo of a fresh {}.", "a photo of a rotten {}.", "a photo of {}s."]
NAMES = POS + FRUIT + VEG + GEN


class ClipGate:
    def __init__(self):
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(ARCH, pretrained=PRETRAINED)
        self.model.eval()
        tokenizer = open_clip.get_tokenizer(ARCH)
        with torch.inference_mode():
            def encode(texts):
                return torch.nn.functional.normalize(self.model.encode_text(tokenizer(texts)), dim=-1)
            rows = [torch.nn.functional.normalize(encode([t.format(c) for t in TEMPLATES]).mean(0), dim=0)
                    for c in POS + FRUIT + VEG]
            rows += list(encode([f"a photo of {g}." for g in GEN]))
        self.text = torch.stack(rows)

    @torch.inference_mode()
    def probabilities(self, images) -> torch.Tensor:
        x = torch.stack([self.preprocess(image) for image in images])
        features = torch.nn.functional.normalize(self.model.encode_image(x), dim=-1)
        return (100 * features @ self.text.T).softmax(1)

    def vegetable_score(self, images) -> torch.Tensor:
        return self.probabilities(images)[:, :len(POS)].sum(1)


if __name__ == "__main__":
    gate = ClipGate()
    for path in sys.argv[1:]:
        p = gate.probabilities([open_image(path)])[0]
        score, top = float(p[:len(POS)].sum()), NAMES[int(p.argmax())]
        print(f"{path}\tvegetable score {score:.4f}\tclosest: {top}\t{'pass' if score >= THRESHOLD else 'not one of the five vegetables'}")
