import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from src.dataset.datasets import Nr3DDataset
from src.relation_encoders.base import CategoryConcept, Far, Near
from src.relation_encoders.ternary import Between as MiddleConcept
from src.relation_encoders.unary import (
    AgainstTheWall,
    AtTheCorner,
    Large,
    Low,
    OnTheFloor,
    Small,
    Tall,
)
from src.relation_encoders.vertical import Above, Below
from src.relation_encoders.view import Behind, Front, Left, Right
from grounding._vendor.lasp.eval_helper import get_all_categories, get_all_relations


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

concepts_classes = {
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
    "smaller": Small,
    "shorter": Small,
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
}

concept_num = {
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
}

rel_num = {
    name: concept_num[encoder] - 1
    for name, encoder in concepts_classes.items()
}

ALL_VALID_RELATIONS = concepts_classes.keys()


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

    object_locations = torch.from_numpy(
        np.vstack(scan_data["pred_locs"])
    ).to(DEVICE)

    room_points = None
    all_concepts = {}

    for concept_name in concept_names:
        if concept_name not in ALL_VALID_RELATIONS:
            all_concepts[concept_name] = category_concept.forward(concept_name)
            continue

        encoder = concepts_classes[concept_name]

        if rel_num[concept_name] == 0:
            if concept_name in {"corner", "against wall", "on the floor"}:
                if room_points is None:
                    room_points = torch.from_numpy(
                        scan_data["room_points"]
                    ).to(DEVICE)

                all_concepts[concept_name] = (
                    encoder(object_locations, room_points)
                    .forward()
                    .float()
                )
            else:
                all_concepts[concept_name] = (
                    encoder(object_locations, None)
                    .forward()
                    .float()
                )
        else:
            all_concepts[concept_name] = (
                encoder(object_locations)
                .forward()
                .float()
            )

    return all_concepts


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
        output_path / f"nr3d_features_per_scene_{args.label}_label.pth",
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

    args = parser.parse_args()
    compute_all_features(args)


if __name__ == "__main__":
    main()
