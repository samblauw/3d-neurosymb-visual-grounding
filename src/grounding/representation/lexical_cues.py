"""Lexical cue tables that select Nr3D train utterances before parsing.

The regex only chooses candidates; parsing decides semantics, negation and
sole-feature eligibility. Used by experiments/representation/lexical_probe.py,
build_stage1_probes.py and the reference-frame row selection.

Cues must be capable of reaching probe_feature.py's sole test:

  isolation    one relation, named in ISOLATION
  againstwall  one relation, "against wall"
  viewdep      one relation in VIEWDEP, not rank >= 2
  ordinal      one relation carrying rank >= 2   -- so only RANKED laterals
               ("second from the left"); "leftmost" lowers to rank 1 and
               "middle" to no rank, and neither can ever qualify
  color        NO relations at all, and a coloured node -- so the cue is just
               that a colour word appears

Negation is preserved for the parser; current sole checks match only relation
names. Cue tables were ported from archive/dev-lexical and define the sample.
"""
from __future__ import annotations

import re

from grounding.representation.encoders.color import LUMINANCE, REFERENCE, SYNONYM

__all__ = ["CUES", "EXCLUDE", "DEFAULT_KINDS", "cue_of"]

_SIDE = r"left|right"
_RANK = (r"second|2nd|third|3rd|fourth|4th|fifth|5th|sixth|6th|"
         r"seventh|7th|eighth|8th|ninth|9th|tenth|10th")
_COLOUR_WORD = "|".join(sorted((re.escape(c) for c in
                                set(REFERENCE) | set(SYNONYM) | set(LUMINANCE)),
                               key=len, reverse=True))

# (name, kind, pattern) per feature. `kind` is the sub-population a cue belongs
# to, recorded so a later split by phrasing needs no second pass.
CUES: dict[str, list[tuple[str, str, str]]] = {
    # kind is what the phrase measures distance to: peer -- same-category
    # objects, named ("the other chairs"); any -- the surroundings, whatever
    # they are ("nothing next to it"); both -- underspecified. Recorded rather
    # than assumed because it is the question the encoder has to answer, and
    # "by itself" -- which says neither -- is the commonest phrasing.
    "isolation": [
        ("by itself",      "both", r"\bby it ?self\b|\bby them ?selves\b"),
        # it'?s: the corpus writes "on it's own" six times, the same phrase with
        # the possessive misspelled. Matching it is a spelling fix, not a
        # widening.
        ("on its own",     "both", r"\bon it'?s own\b|\bon their own\b|\ball on it'?s own\b"),
        ("alone",          "both", r"\b(?:standing |sitting |sits |stands |all )?alone\b"),
        ("lone",           "both", r"\blone(?:ly|some)?\b|\bsolitary\b|\bsolo\b"),
        ("isolated",       "both", r"\bisolated\b|\bin isolation\b"),
        ("away from",      "peer", r"\baway from (?:the |all (?:the )?)?(?:other|rest|remaining)"),
        ("separated",      "peer", r"\b(?:separated|separate|apart|set apart|removed|detached)"
                                   r"\s+from\b|\bset apart\b|\boff to (?:the )?(?:side|itself)\b"),
        ("furthest from",  "peer", r"\b(?:furthest|farthest|far(?:thest)?) (?:away )?from "
                                   r"(?:the )?(?:other|rest|group)"),
        # Lateral only. "a desk with nothing ON it" is an occupancy statement
        # about the desk's surface, not a claim that it stands apart.
        ("nothing around", "any",  r"\bnothing (?:is )?(?:around|next to|beside|near)\b"),
        ("not near any",   "peer", r"\bnot (?:near|next to|beside|touching|close to) any\b"),
    ],
    # clean_program's _AGAINST_WALL, split into the two claims it conflates. The
    # encoder scores ADJACENCY -- how close the box sits to a wall plane -- and
    # only the first group asserts that. "on the wall" and "wall-mounted" say a
    # thing is FIXED TO a wall surface, which every picture, window and door in
    # the room satisfies, so the cue cannot discriminate within its own class.
    #
    # Hence --kind adjacency by default. The mounted group is still listed and
    # still selectable (--kind adjacency,mounted) because excluding it is a
    # claim about what the ENCODER can use, not about what the sentence means.
    #
    # NB this deliberately no longer matches clean_program's gate, which still
    # lets "on the wall" assert against_wall. The probe stops selecting those
    # rows; the parser keeps emitting them. Narrowing the gate itself would strip
    # the attribute from many against-wall programs -- a representation change to
    # test on its own, not a side effect of a probe.
    "againstwall": [
        ("adjacency", "adjacency", r"\b(?:up\s+)?against\s+(?:the|a)\s+wall|"
                                   r"\balong\s+the\s+wall|\bflush\s+against|"
                                   r"\bpushed\s+against|\battached\s+to\s+the\s+wall|"
                                   r"\btouching\s+the\s+wall|\bnext\s+to\s+the\s+wall|"
                                   r"\bby\s+the\s+wall"),
        ("mounted",   "mounted",   r"\bon\s+the\s+wall\b|\bwall\s*-?\s*mounted\b"),
    ],
    # Ranked laterals only -- see the header. The rank and the side are captured
    # so the pair is reportable without re-parsing the sentence.
    "ordinal": [
        ("from side",  "lateral", rf"\b(?P<rank>{_RANK})\b[^.;,]{{0,40}}?\bfrom\s+"
                                  rf"(?:the\s+|your\s+|our\s+)?(?:far\s+)?(?P<side>{_SIDE})\b"),
        ("counting",   "lateral", rf"\bcount(?:ing)?\s+(?:in\s+)?from\s+(?:the\s+)?"
                                  rf"(?P<side>{_SIDE})\b[^.;]{{0,30}}?\b(?P<rank>{_RANK})\b"),
        ("side first", "lateral", rf"\bfrom\s+(?:the\s+)?(?P<side>{_SIDE})\b[^.;,]{{0,30}}?"
                                  rf"\b(?P<rank>{_RANK})\b"),
        ("on side",    "lateral", rf"\b(?P<rank>{_RANK})\b[^.;,]{{0,40}}?\bon\s+"
                                  rf"(?:the\s+|your\s+|our\s+)?(?P<side>{_SIDE})\b"),
        ("to side",    "lateral", rf"\b(?P<rank>{_RANK})\b[^.;,]{{0,40}}?\bto\s+"
                                  rf"(?:the\s+|your\s+|our\s+)?(?P<side>{_SIDE})\b"),
    ],
    "color": [("colour word", "colour", rf"\b(?:{_COLOUR_WORD})\b")],
    # An utterance that STATES an observer frame. Split by what the frame gives
    # the resolver, because the routing table keys on exactly that: `stance`
    # names a position, `gaze` a look target, `back` a target to face away from,
    # `entry` a doorway walked through. `deictic` says the frame is the reader's
    # own and supplies no anchor at all -- it is kept because "on your left" is
    # a frame claim the representation provably cannot serve, and counting those
    # rows is how the abstention split stays honest.
    #
    # These are NOT the parse. v4 removed the parser's viewpoint slot, so a
    # frame reaches the experiment only through the separate diagnostic pass;
    # this table decides which utterances that pass is run on, and nothing else.
    "refframe": [
        ("stance",  "stance",  r"\bstand(?:ing|s)?\s+(?:at|in|on|by|near)\b|"
                               r"\bsit(?:ting|s)?\s+(?:at|in|on)\b|"
                               r"\bif you (?:are |were )?(?:stand|sit)"),
        ("gaze",    "gaze",    r"\bfac(?:e|ing|ed)\b|\blook(?:ing)?\s+(?:at|toward|towards|into|in)\b|"
                               r"\bstar(?:e|ing)\s+at\b|\btoward(?:s)?\b"),
        ("back",    "back",    r"\bback\s+(?:is\s+)?to\b|\bback\s+turned\b|"
                               r"\bwith\s+your\s+back\b"),
        ("entry",   "entry",   r"\bwalk(?:ing)?\s+(?:in|into|through)\b|"
                               r"\benter(?:ing|ed)?\b|\bcom(?:e|ing)\s+(?:in|through)\b|"
                               r"\bas\s+you\s+(?:walk|enter|come)\b"),
        ("deictic", "deictic", r"\bon\s+your\s+(?:left|right)\b|\byour\s+(?:left|right)\b|"
                               r"\bbehind\s+you\b|\bin\s+front\s+of\s+you\b"),
    ],
    "viewdep": [
        ("lateral", "lateral", r"\b(?:to|on|at) the (?:far )?(?:left|right)\b|"
                               r"\b(?:left|right)(?:most|-hand)?\b"),
        ("frontal", "frontal", r"\bin front of\b|\bbehind\b|\bat the back of\b"),
    ],
}

# Matches that mean the cue is about a different axis or a different claim.
EXCLUDE: dict[str, str] = {
    # "second shelf from the top, on the right" -- vertical rank, wrong axis.
    "ordinal": r"\bfrom\s+(?:the\s+)?(?:top|bottom|up|down|floor|ceiling)\b",
}

# NOT cues, for isolation: "the only door", "the single chair" are UNIQUENESS
# claims. The Isolated encoder scores distance to neighbours, so a unique object
# satisfies it only by accident, and the parser still emits `isolated` for some
# of them.

#: Kinds selected when --kind is not given. Only where the full set would put
#: rows in the population that the feature's encoder provably cannot use.
DEFAULT_KINDS: dict[str, set[str]] = {"againstwall": {"adjacency"}}

_COMPILED = {f: [(n, k, re.compile(p, re.I)) for n, k, p in cues]
             for f, cues in CUES.items()}
_EXCLUDE = {f: re.compile(p, re.I) for f, p in EXCLUDE.items()}


def cue_of(feature: str, utterance: str,
           kinds: set[str] | None = None) -> tuple[str | None, str | None, int, int]:
    """The earliest cue in the utterance: (name, kind, start, end), or all-empty.

    Earliest rather than first-listed so the reported cue is the one the sentence
    leads with, whatever order the table happens to be in.

    `kinds` narrows which cues are eligible BEFORE the earliest is chosen, not
    after. Filtering afterwards would drop "the picture on the wall, against the
    far wall" for leading with a mounted phrase, even though it carries an
    adjacency one too.
    """
    bad = _EXCLUDE.get(feature)
    best: tuple[str | None, str | None, int, int] = (None, None, -1, -1)
    for name, kind, pat in _COMPILED[feature]:
        if kinds and kind not in kinds:
            continue
        m = pat.search(utterance)
        if not m or (bad is not None and bad.search(utterance[:m.end()])):
            continue
        if best[2] < 0 or m.start() < best[2]:
            best = (name, kind, m.start(), m.end())
    return best
