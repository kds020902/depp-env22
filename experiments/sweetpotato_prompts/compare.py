"""experiments/sweetpotato_prompts (PREREG.md): change only the potato / sweet potato text prompts of the new-vegetable head.

    python experiments/sweetpotato_prompts/compare.py

Every weight and threshold stays TV-12's; the four rows of new_vegetable_head["text_embeddings"] are replaced by the prompts
of prompts.json. Development photos only (tools/compare_on_dev.py); writes dev_results.json and prints the tables.
Needs the CLIP text encoder (open_clip, about 600 MB of weights).
"""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "tools"))
from clip_model import ClipModel  # noqa: E402
from compare_on_dev import AWAY, changed, development_sets, evaluate, table  # noqa: E402
from labels import CLASS_NAMES  # noqa: E402

FIVE = CLASS_NAMES[:10]


def embed(clip: ClipModel, sentences: list[str]) -> torch.Tensor:
    """As TV-12's rows were made (experiments/new_vegetables/v7/explore.py): normalise each sentence, mean, normalise."""
    import open_clip
    tokens = open_clip.get_tokenizer("ViT-B-16")(sentences)
    with torch.inference_mode():
        rows = torch.nn.functional.normalize(clip.model.encode_text(tokens), dim=-1)
    return torch.nn.functional.normalize(rows.mean(0), dim=0)


def with_prompts(tv12: dict, clip: ClipModel, prompts: dict[str, list[str]]) -> dict:
    checkpoint = copy.deepcopy(tv12)
    head = checkpoint["new_vegetable_head"]
    for class_name, sentences in prompts.items():
        i = CLASS_NAMES.index(class_name)
        head["text_embeddings"][i] = embed(clip, sentences)
        head["prompts"][i] = "|".join(sentences)
    return checkpoint


def root_counts(result: dict, sets: dict) -> dict:
    """The numbers of PREREG.md from one system's answers."""
    labels = [y for _, _, ys in sets.values() if ys is not None for y in ys]
    answers = [a for name, (_, _, ys) in sets.items() if ys is not None for a in result[name]["answers"]]
    pick = lambda cls: [a for y, a in zip(labels, answers) if y.startswith(cls)]
    kind = lambda group, species: sum(a.startswith(species + "_") for a in group)
    rotten, fresh, potato = pick("sweetpotato_rotten"), pick("sweetpotato_fresh"), pick("potato_")
    total = result["_all_labelled"]
    return {"rotten sweet potato -> sweet potato": [kind(rotten, "sweetpotato"), len(rotten)],
            "rotten sweet potato -> rotten sweet potato": [rotten.count("sweetpotato_rotten"), len(rotten)],
            "rotten sweet potato -> potato": [kind(rotten, "potato"), len(rotten)],
            "fresh sweet potato -> sweet potato": [kind(fresh, "sweetpotato"), len(fresh)],
            "potato -> sweet potato": [kind(potato, "sweetpotato"), len(potato)],
            "rotten called fresh (all labelled)": [total["rotten_called_fresh"], total["rotten_n"]],
            "five vegetables right": [sum(total["classes"][c]["right"] for c in FIVE), sum(total["classes"][c]["n"] for c in FIVE)],
            "held back": [total["held_back"], total["n"]], "shown answers right": [total["shown_right"], total["shown"]],
            "fruit 374 turned away": [result["fruit 374"]["answers"].count(AWAY), 374],
            "everyday things 400 turned away": [result["everyday things 400"]["answers"].count(AWAY), 400]}


def conditions(base: dict, c: dict) -> dict:
    first = lambda d, key: d[key][0]
    return {"rotten sweet potato -> sweet potato: +2 or more": first(c, "rotten sweet potato -> sweet potato") >= first(base, "rotten sweet potato -> sweet potato") + 2,
            "potato -> sweet potato: at most +1": first(c, "potato -> sweet potato") <= first(base, "potato -> sweet potato") + 1,
            "rotten called fresh: not more": first(c, "rotten called fresh (all labelled)") <= first(base, "rotten called fresh (all labelled)"),
            "five vegetables right: at least -2": first(c, "five vegetables right") >= first(base, "five vegetables right") - 2,
            "fruit turned away: at least -2": first(c, "fruit 374 turned away") >= first(base, "fruit 374 turned away") - 2,
            "everyday things turned away: at least -2": first(c, "everyday things 400 turned away") >= first(base, "everyday things 400 turned away") - 2}


def main():
    tv12 = torch.load(REPO / "model.pth", map_location="cpu", weights_only=True)
    clip, sets = ClipModel("cpu"), development_sets()
    head = tv12["new_vegetable_head"]
    # the stored rows must come back from the stored prompts, else this code does not embed text as TV-12 did
    cosines = [float(embed(clip, head["prompts"][i].split("|")) @ head["text_embeddings"][i]) for i in range(len(CLASS_NAMES))]
    assert min(cosines) > 0.999, f"TV-12's prompts do not reproduce its stored text embeddings: {cosines}"
    systems = {"TV-12": tv12, **{name: with_prompts(tv12, clip, prompts)
                                 for name, prompts in json.load(open(HERE / "prompts.json", encoding="utf-8")).items()}}
    results = {name: evaluate(checkpoint, sets) for name, checkpoint in systems.items()}
    counts = {name: root_counts(result, sets) for name, result in results.items()}
    checks = {name: conditions(counts["TV-12"], c) for name, c in counts.items() if name != "TV-12"}
    record = {"tv12_prompt_reproduction_min_cosine": round(min(cosines), 6), "counts": counts, "conditions": checks,
              "carried_forward": [name for name, c in checks.items() if all(c.values())],
              "answers_changed_from_TV-12": {name: changed(results["TV-12"], r, sets) for name, r in results.items() if name != "TV-12"},
              "answers": {name: {s: r[s]["answers"] for s in sets} for name, r in results.items()}}
    json.dump(record, open(HERE / "dev_results.json", "w"), indent=1, ensure_ascii=False)
    print(table(results), "\n")
    names = list(counts)
    print("| count | " + " | ".join(names) + " |\n|---|" + "---:|" * len(names))
    for key in counts["TV-12"]:
        print(f"| {key} | " + " | ".join(f"{counts[m][key][0]}/{counts[m][key][1]}" for m in names) + " |")
    for name, c in checks.items():
        print(name, "carried forward" if all(c.values()) else "not carried forward", {k: v for k, v in c.items() if not v})
    print("answers changed:", record["answers_changed_from_TV-12"])


if __name__ == "__main__":
    main()
