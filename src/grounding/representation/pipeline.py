"""Extended vocabulary and feature extraction using the project encoders."""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import torch
from tqdm import tqdm

from grounding.data.dataset import Nr3DDataset
from grounding._vendor.lasp.encoders.base import CategoryConcept, Far, Near
from grounding._vendor.lasp.encoders.ternary import Between as MiddleConcept
from grounding._vendor.lasp.encoders.unary import (
    AtTheCorner,
    Large,
    Low,
    OnTheFloor,
    Small,
    Tall,
)
from grounding._vendor.lasp.encoders.vertical import Above, Below
from grounding.representation.encoders.view import Behind, Front, Left, Right
from grounding._vendor.lasp.eval_helper import get_all_categories, get_all_relations

from grounding.representation.build import build_scene_features
from grounding.representation.encoders.against_wall import AgainstTheWall
from grounding.representation.encoders.isolation import Isolated
from grounding.representation.features import get_all_colors
from grounding.representation.vocabulary import Vocabulary


CONCEPTS = {
    "beside": Near,
    "near": Near,
    "next to": Near,
    "within": Near,
    "in": Near,
    "inside": Near,
    "close": Near,
    "closer": Near,
    "closest": Near,
    "far": Far,
    "farthest": Far,
    "opposite": Far,
    "furthest": Far,
    "corner": AtTheCorner,
    "against wall": AgainstTheWall,
    "above": Above,
    "on top": Above,
    "on the top": Above,
    "on": Above,
    "below": Below,
    "under": Below,
    "beneath": Below,
    "higher": Tall,
    "taller": Tall,
    "highest": Tall,
    "upper": Tall,
    "lower": Low,
    "lowest": Low,
    "smaller": Small,
    "shorter": Small,
    "smallest": Small,
    "larger": Large,
    "bigger": Large,
    "largest": Large,
    "longer": Large,
    "middle": MiddleConcept,
    "center": MiddleConcept,
    "between": MiddleConcept,
    "surrounded by": Near,
    "around": Near,
    "left": Left,
    "right": Right,
    "front": Front,
    "back": Behind,
    "behind": Behind,
    "cross": Far,
    "across": Far,
    "facing": Near,
    "with": Near,
    "attached": Near,
    "on the floor": OnTheFloor,
    "isolated": Isolated,
    "isolation": Isolated,
    "alone": Isolated,
    "by itself": Isolated,
    "lone": Isolated,
}

ARITIES = {
    Near: 2,
    Far: 2,
    AgainstTheWall: 1,
    Above: 2,
    Below: 2,
    Tall: 1,
    Low: 1,
    Small: 1,
    Large: 1,
    MiddleConcept: 3,
    Left: 2,
    Right: 2,
    Front: 2,
    Behind: 2,
    AtTheCorner: 1,
    OnTheFloor: 1,
    Isolated: 1,
}

VOCABULARY = Vocabulary.from_tables(CONCEPTS, ARITIES)
rel_num = VOCABULARY.rel_num
ALL_VALID_RELATIONS = VOCABULARY.names

def get_all_features(
    scan_id: str,
    concept_names: set[str],
    dataset: Nr3DDataset,
    label_type: str,
):
    scan_data = dataset.scans[scan_id]
    category_concept = CategoryConcept(
        scan_data,
        label_type=label_type,
    )

    return build_scene_features(
        scan_data,
        concept_names,
        vocabulary=VOCABULARY,
        category_concept=category_concept,
    )


def compute_all_features(args) -> None:
    dataset = Nr3DDataset(split=args.split, label_type=args.label)

    concept_names_per_scene = defaultdict(set)
    concepts_per_scene = defaultdict(dict)

    with Path(args.parses).open() as f:
        for line in tqdm(f):
            data = json.loads(line)
            json_obj = data["json_obj"]

            if json_obj is None:
                continue

            scan_id = data["scan_id"]
            concepts = concept_names_per_scene[scan_id]

            concepts.update(get_all_relations(json_obj))
            concepts.update(get_all_categories(json_obj))
            concepts.update(
                f"color:{color}"
                for color in get_all_colors(json_obj)
            )

    for scan_id, concept_names in tqdm(concept_names_per_scene.items()):
        concepts_per_scene[scan_id] = get_all_features(
            scan_id,
            concept_names,
            dataset,
            args.label,
        )

    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)

    torch.save(
        concepts_per_scene,
        output_path
        / f"nr3d_features_per_scene_{args.label}_label_extended.pth",
    )


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--output",
        required=True,
    )
    parser.add_argument(
        "--label",
        choices=["pred", "gt"],
        default="gt",
    )
    parser.add_argument(
        "--split",
        choices=["test", "dev", "train"],
        default="test",
        help="which annotation csv the scene loader reads: test (default) is the held-out split, dev the frozen dev draw written by scripts/prepare/build_dev_csv.py. Features and parses must be on the same split.",
    )
    parser.add_argument(
        "--parses",
        default="data/corpora/nr3d.jsonl",
        help="Nr3D parses used to determine the required concepts.",
    )

    parser.add_argument("--gap_scale", type=float)
    parser.add_argument("--distance_scale", type=float)
    parser.add_argument("--color_scale", type=float)

    args = parser.parse_args()

    from grounding import sensitivity

    sensitivity_args = {
        "gap_scale": "wall.gap_scale",
        "distance_scale": "isolation.distance_scale",
        "color_scale": "color.scale",
    }

    for flag, name in sensitivity_args.items():
        value = getattr(args, flag)
        if value is not None:
            sensitivity.set(name, value)

    print(f"sensitivity parameters: {sensitivity.snapshot()}")

    compute_all_features(args)


if __name__ == "__main__":
    main()
