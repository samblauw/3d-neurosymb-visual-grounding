"""Build scene features from an engine's vocabulary and category encoder.

Thesis wall, isolation and colour features use additional scene geometry.
Only concepts requested by the supplied programs are built; colour keys use
``color:<name>``. Cache contents therefore depend on the parse input."""

from __future__ import annotations

import numpy as np
import torch

from grounding.representation.encoders.against_wall import AgainstTheWall
from grounding.representation.encoders.color import ColorConcept
from grounding.representation.encoders.isolation import Isolated
from grounding.representation.vocabulary import ROOM_CONCEPTS, Vocabulary

__all__ = ["build_scene_features"]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_scene_features(scan_data, concept_names, *, vocabulary: Vocabulary,
                         category_concept):
    """``{concept_name: tensor}`` for one scene.

    ``scan_data``        the scene dict (pred_locs, inst_labels, obj_ids, ...)
    ``concept_names``    the names this scene's parses mention
    ``vocabulary``       name -> encoder class, plus arities
    ``category_concept`` the CLIP category encoder, already constructed by the
                         caller (it is vendor code and takes a label_type)
    """
    all_concepts = {}
    object_locations = torch.from_numpy(np.vstack(scan_data["pred_locs"])).to(DEVICE)
    room_points = None          # loaded once, only if a room concept needs it
    color_concept = None        # constructed once, only if a colour is asked for

    for concept_name in concept_names:
        if concept_name.startswith("color:"):
            if color_concept is None:
                color_concept = ColorConcept(scan_data["gt_colors"], scan_data["obj_ids"])
            all_concepts[concept_name] = color_concept.forward(concept_name[6:]).float()
            continue

        if concept_name not in vocabulary.names:
            # Retired viewpoint keys are not category names. Ignore them when
            # reading historical programs instead of embedding their literal text.
            if "@" in concept_name:
                continue
            all_concepts[concept_name] = category_concept.forward(concept_name)
            continue

        encoder = vocabulary.encoder_for(concept_name)

        if vocabulary.rel_num[concept_name] != 0:
            # A relation: the encoder needs only the object boxes.
            all_concepts[concept_name] = encoder(object_locations).forward().float()
            continue

        # --- unary concepts ---------------------------------------------
        if encoder is Isolated:
            # Isolation is judged against objects of the SAME category, so this
            # encoder needs the per-object labels the others do not.
            all_concepts[concept_name] = (
                Isolated(object_locations, None, category_concept.pred_class_list)
                .forward().float())
        elif concept_name in ROOM_CONCEPTS:
            if room_points is None:
                room_points = torch.from_numpy(scan_data["room_points"]).to(DEVICE)
            if encoder is AgainstTheWall:
                # The annotated wall boxes, not the room outline: the outline
                # cannot tell a wall from the object standing in front of it.
                all_concepts[concept_name] = (
                    AgainstTheWall(object_locations, room_points,
                                   wall_boxes=scan_data.get("wall_boxes"))
                    .forward().float())
            else:
                all_concepts[concept_name] = (
                    encoder(object_locations, room_points).forward().float())
        else:
            # Upstream's unary signature: boxes, and an optional convex hull.
            all_concepts[concept_name] = (
                encoder(object_locations, None).forward().float())

    return all_concepts
