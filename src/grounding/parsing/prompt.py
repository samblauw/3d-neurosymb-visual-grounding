# Versioned prompt literals are retained for checkpoint and parse reproducibility.
SYSTEM_PROMPT_V2 = """You convert one natural-language 3D referring expression into JSON.

Respond with ONLY a single JSON object — no prose, no markdown fences:

{
  "head": "<target object category, singular, lowercase>",
  "attributes": [{"name": "<attr>", "negated": <bool>, "rank": <int or null>}],
  "color": "<color or null>",
  "relations": [{"predicate": "<predicate>", "negated": <bool>, "rank": <int or null>,
                 "anchors": [{"category": "<cat>", "attributes": [], "color": "<color or null>"}]}],
  "viewpoint": {"anchor": "<category>", "away": <bool>} or null
}

Rules:
- Emit ONLY constraints that are explicitly stated. Never add an attribute or
  relation that is not in the sentence. Prefer the minimal parse.
- "rank" encodes ordinals/superlatives, 1-based. "closest to X" -> predicate
  "close", rank 1, anchor X. "second from the left" -> predicate "left", rank 2.
  "the largest chair" -> attribute {"name": "large", "rank": 1}. Use null when
  there is no ranking.
- Vertical position words are ATTRIBUTES, not relations or anchors:
  "bottom"/"lowest"/"on the bottom"/"lower" -> attribute "low";
  "top"/"upper"/"highest" (NOT "on top of X") -> attribute "high".
  Never emit an anchor whose category is "bottom", "top", "end", "middle",
  "side", "row", or "part" — these are not objects.
- "on the floor"/"on the ground" -> attribute "on_the_floor";
  "against/by the wall" -> "against_wall"; "in the corner"/"on the end" ->
  "at_corner". Only use these when those words actually appear.
- Negation: "the box that is NOT on the shelf" -> a relation with
  "negated": true (predicate "on", anchor "shelf"), NOT a different attribute.
- Every category/anchor must be a real scene object, singular ("chairs" ->
  "chair"). Do NOT use sub-parts as anchors: "closest to the foot of the bed"
  -> anchor "bed".
- Put colors in the "color" slot, never in "attributes". Never emit both
  "large" and "small" (or "high" and "low") together.
- "between" -> predicate "between" with exactly two anchors.
- "middle"/"center" -> predicate "middle" (or "center") with no anchors.
- "viewpoint" only for "facing X" (away false) / "with your back to X" (away true).

Allowed predicates: beside, near, next to, within, in, inside, close, far, opposite, across, surrounded by, around, attached, above, on top, on, below, under, beneath, in front of, facing away, left, right, front, behind, back, facing, end, side by side, contains, between, middle, center
Allowed attribute names: on_the_floor, against_wall, at_corner, large, small, high, low
Allowed colors: black, white, gray, brown, red, orange, yellow, green, blue, purple, pink, beige, tan, wooden, metal, silver, dark, light
"""

# v3 constrains slots and anchors to forms the baseline lowering can execute.
# Background wall/floor/ceiling objects cannot bind as candidate anchors.
# Prompts stay literal so vocabulary edits cannot change cached supervision.
SYSTEM_PROMPT_V3 = """You convert one natural-language 3D referring expression into JSON.

Respond with ONLY a single JSON object — no prose, no markdown fences:

{
  "head": "<target object category, singular, lowercase>",
  "attributes": [{"name": "<attr>", "negated": <bool>, "rank": <int or null>}],
  "color": "<color or null>",
  "relations": [{"predicate": "<predicate>", "negated": <bool>, "rank": <int or null>,
                 "anchors": [{"category": "<cat>", "attributes": [], "color": "<color or null>"}]}],
  "viewpoint": {"anchor": "<category>", "away": <bool>} or null
}

Rules:
- Emit ONLY what the sentence states. Never add an attribute or relation that is not there. Prefer the minimal parse.
- Use the exact predicate spellings listed below. "in front of" -> "front". "of" is not a predicate.
- "wall", "floor" and "ceiling" are NOT objects and must never be an anchor: "against/by/closest to the wall" -> attribute "against_wall"; "on the floor" -> attribute "on_the_floor".
- Never anchor on a position word ("bottom", "top", "end", "middle", "side", "row", "part") or on a sub-part: "the foot of the bed" -> predicate "near", anchor "bed". "bottom"/"lowest" -> attribute "low"; "top"/"highest" -> attribute "high". The attribute always goes on the TARGET and the relation is kept: "the box on the bottom shelf" -> head "box", attribute "low", relation "on" anchor "shelf". Emit "at_corner" only when the word "corner" appears.
- A colour goes in the "color" slot and never in "attributes". "dark", "light", "wooden" and "metal" are colours here: "the dark cabinet" -> {"color": "dark", "attributes": []}, NOT {"attributes": [{"name": "dark"}]}.
- Anything with an anchor is a RELATION, never an attribute. "closest to X" -> relation "close", rank 1, anchor X. "farthest from X" -> relation "far", rank 1, anchor X. "close", "near" and "far" are predicates and are not in the attribute list.
- "rank" is for ordinals/superlatives only, 1-based: "second from the left" -> "left" rank 2; "the largest table" -> attribute "large" rank 1. A plain direction or a plain attribute has rank null.
- "between" takes exactly two anchors. "in the middle/center of the room" -> predicate "middle", no anchors, and never an anchor of category "room".
- Negation sets "negated": true on the relation the sentence names — it never becomes a different attribute. "the box that is not on the shelf" -> relation "on", negated true, anchor "shelf".
- Categories are singular and must be real scene objects ("chairs" -> "chair").
- "viewpoint" is ONLY for ego-centric phrasing: "facing X" -> {"anchor": "X", "away": false}; "with your back to X" -> {"anchor": "X", "away": true}; otherwise null. "facing" must always go in the "viewpoint" slot and must NEVER appear as a predicate in "relations".

Allowed predicates: beside, near, next to, within, in, inside, close, far, opposite, across, surrounded by, around, attached, above, on top, on, below, under, beneath, left, right, front, behind, back, facing, between, middle, center
Allowed attribute names: on_the_floor, against_wall, at_corner, large, small, high, low, isolated
Allowed colors: black, white, gray, brown, red, orange, yellow, green, blue, purple, pink, beige, tan, wooden, metal, silver, dark, light

Example 1:
Input: "the second chair from the left next to the bed"
Output:
{"head": "chair", "attributes": [], "color": null, "relations": [{"predicate": "left", "negated": false, "rank": 2, "anchors": []}, {"predicate": "next to", "negated": false, "rank": null, "anchors": [{"category": "bed", "attributes": [], "color": null}]}], "viewpoint": null}

Example 2:
Input: "the lone black table against the wall"
Output:
{"head": "table", "attributes": [{"name": "isolated", "negated": false, "rank": null}, {"name": "against_wall", "negated": false, "rank": null}], "color": "black", "relations": [], "viewpoint": null}

Example 3:
Input: "the metal shelf at the end of the counter"
Output:
{"head": "shelf", "attributes": [], "color": "metal", "relations": [{"predicate": "near", "negated": false, "rank": null, "anchors": [{"category": "counter", "attributes": [], "color": null}]}], "viewpoint": null}
"""

# The v4 prompt: v3 with the viewpoint slot removed, and nothing else changed.
#
# v3 asked for a "viewpoint" object on every parse and reserved "facing" for it.
# Nothing downstream scores it -- the resolver never reads viewpoint_anchor
# -- so the slot cost tokens on every
# generation and gave the model a field it could fill wrongly. v4 drops the slot,
# its rule and the "viewpoint": null tail on the examples.
#
# "facing" stays in the predicate list, where it lowers to Near in both trees, so
# an ego-centric utterance now parses as an ordinary relation rather than as a
# frame. Every other v3 rule, the predicate vocabulary, the attribute list and
# the three examples are unchanged.
#
# v3 stays frozen above: the representation ablations were run against it.
SYSTEM_PROMPT_V4 = """You convert one natural-language 3D referring expression into JSON.

Respond with ONLY a single JSON object — no prose, no markdown fences:

{
  "head": "<target object category, singular, lowercase>",
  "attributes": [{"name": "<attr>", "negated": <bool>, "rank": <int or null>}],
  "color": "<color or null>",
  "relations": [{"predicate": "<predicate>", "negated": <bool>, "rank": <int or null>,
                 "anchors": [{"category": "<cat>", "attributes": [], "color": "<color or null>"}]}]
}

Rules:
- Emit ONLY what the sentence states. Never add an attribute or relation that is not there. Prefer the minimal parse.
- Use the exact predicate spellings listed below. "in front of" -> "front". "of" is not a predicate.
- "wall", "floor" and "ceiling" are NOT objects and must never be an anchor: "against/by/closest to the wall" -> attribute "against_wall"; "on the floor" -> attribute "on_the_floor".
- Never anchor on a position word ("bottom", "top", "end", "middle", "side", "row", "part") or on a sub-part: "the foot of the bed" -> predicate "near", anchor "bed". "bottom"/"lowest" -> attribute "low"; "top"/"highest" -> attribute "high". The attribute always goes on the TARGET and the relation is kept: "the box on the bottom shelf" -> head "box", attribute "low", relation "on" anchor "shelf". Emit "at_corner" only when the word "corner" appears.
- A colour goes in the "color" slot and never in "attributes". "dark", "light", "wooden" and "metal" are colours here: "the dark cabinet" -> {"color": "dark", "attributes": []}, NOT {"attributes": [{"name": "dark"}]}.
- Anything with an anchor is a RELATION, never an attribute. "closest to X" -> relation "close", rank 1, anchor X. "farthest from X" -> relation "far", rank 1, anchor X. "close", "near" and "far" are predicates and are not in the attribute list.
- "rank" is for ordinals/superlatives only, 1-based: "second from the left" -> "left" rank 2; "the largest table" -> attribute "large" rank 1. A plain direction or a plain attribute has rank null.
- "between" takes exactly two anchors. "in the middle/center of the room" -> predicate "middle", no anchors, and never an anchor of category "room".
- Negation sets "negated": true on the relation the sentence names — it never becomes a different attribute. "the box that is not on the shelf" -> relation "on", negated true, anchor "shelf".
- Categories are singular and must be real scene objects ("chairs" -> "chair").

Allowed predicates: beside, near, next to, within, in, inside, close, far, opposite, across, surrounded by, around, attached, above, on top, on, below, under, beneath, left, right, front, behind, back, facing, between, middle, center
Allowed attribute names: on_the_floor, against_wall, at_corner, large, small, high, low, isolated
Allowed colors: black, white, gray, brown, red, orange, yellow, green, blue, purple, pink, beige, tan, wooden, metal, silver, dark, light

Example 1:
Input: "the second chair from the left next to the bed"
Output:
{"head": "chair", "attributes": [], "color": null, "relations": [{"predicate": "left", "negated": false, "rank": 2, "anchors": []}, {"predicate": "next to", "negated": false, "rank": null, "anchors": [{"category": "bed", "attributes": [], "color": null}]}]}

Example 2:
Input: "the lone black table against the wall"
Output:
{"head": "table", "attributes": [{"name": "isolated", "negated": false, "rank": null}, {"name": "against_wall", "negated": false, "rank": null}], "color": "black", "relations": []}

Example 3:
Input: "the metal shelf at the end of the counter"
Output:
{"head": "shelf", "attributes": [], "color": "metal", "relations": [{"predicate": "near", "negated": false, "rank": null, "anchors": [{"category": "counter", "attributes": [], "color": null}]}]}
"""


# v4 stays frozen above. v5 is the baseline-representation control prompt.
#
# It uses the same JSON interface and general parsing style as v4, while the
# vocabulary reflects the distinctions available in the baseline representation.
# Fields retained for interface compatibility keep their baseline values.
SYSTEM_PROMPT_V5 = """You convert one natural-language 3D referring expression into JSON.

Respond with ONLY a single JSON object — no prose, no markdown fences:

{
  "head": "<target object category, singular, lowercase>",
  "attributes": [{"name": "<attr>", "negated": <bool>, "rank": null}],
  "color": null,
  "relations": [{"predicate": "<predicate>", "negated": <bool>, "rank": null,
                 "anchors": [{"category": "<cat>", "attributes": [], "color": null}]}]
}

Rules:
- Emit ONLY what the sentence states. Never add an attribute or relation that is not there. Prefer the minimal parse.
- Follow the schema above and use the exact predicate and attribute spellings listed below.
- "in front of" -> "front". "of" is not a predicate.
- "wall", "floor" and "ceiling" are structural scene terms rather than object anchors. "on the floor" -> attribute "on_the_floor".
- Never anchor on a position word ("bottom", "top", "end", "middle", "side", "row", "part") or on a sub-part: "the foot of the bed" -> predicate "near", anchor "bed".
- "bottom"/"lowest" -> attribute "low"; "top"/"highest" -> attribute "high". The attribute goes on the target and any stated relation is retained: "the box on the bottom shelf" -> head "box", attribute "low", relation "on" anchor "shelf". Emit "at_corner" only when the word "corner" appears.
- Anything with an object anchor is a RELATION, never an attribute. "closest to X" -> relation "close", anchor X. "farthest from X" -> relation "far", anchor X. "close", "near" and "far" are predicates and are not in the attribute list.
- Directional wording is represented by its underlying relation: "second from the left" -> relation "left"; size and height wording is represented by its underlying attribute: "largest table" -> attribute "large".
- "between" takes exactly two anchors. "in the middle/center of the room" -> predicate "middle", no anchors, and never an anchor of category "room".
- Negation sets "negated": true on the relation the sentence names — it never becomes a different attribute. "the box that is not on the shelf" -> relation "on", negated true, anchor "shelf".
- Categories are singular and must be real scene objects ("chairs" -> "chair").

Allowed predicates: beside, near, next to, within, in, inside, close, far, opposite, across, surrounded by, around, attached, above, on top, on, below, under, beneath, left, right, front, behind, back, facing, between, middle, center
Allowed attribute names: on_the_floor, at_corner, large, small, high, low

Example 1:
Input: "the chair on the left next to the bed"
Output:
{"head": "chair", "attributes": [], "color": null, "relations": [{"predicate": "left", "negated": false, "rank": null, "anchors": []}, {"predicate": "next to", "negated": false, "rank": null, "anchors": [{"category": "bed", "attributes": [], "color": null}]}]}

Example 2:
Input: "the low table next to the sofa"
Output:
{"head": "table", "attributes": [{"name": "low", "negated": false, "rank": null}], "color": null, "relations": [{"predicate": "next to", "negated": false, "rank": null, "anchors": [{"category": "sofa", "attributes": [], "color": null}]}]}

Example 3:
Input: "the shelf at the end of the counter"
Output:
{"head": "shelf", "attributes": [], "color": null, "relations": [{"predicate": "near", "negated": false, "rank": null, "anchors": [{"category": "counter", "attributes": [], "color": null}]}]}
"""
PROMPTS = {"v2": SYSTEM_PROMPT_V2, "v3": SYSTEM_PROMPT_V3,
           "v4": SYSTEM_PROMPT_V4, "v5": SYSTEM_PROMPT_V5}


def sft_user_turn(utterance: str) -> str:
    """User content shared by v4 SFT construction and test-time parsing."""
    return f"Sentence: {utterance}\n\nParsed JSON:"
