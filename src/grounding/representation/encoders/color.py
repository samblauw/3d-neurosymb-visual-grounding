# Score each object against a named colour, using the GT per-instance colour GMM.
#
# instance_id_to_gmm_color gives, per instance, a 3-component Gaussian mixture
# over RGB in [-1, 1]: {"weights": [w1,w2,w3], "means": [[r,g,b] x3]}. Scoring
# against the mixture rather than a single average matters -- a white chair with
# a black seat averages to grey and would match neither.

from __future__ import annotations

import numpy as np
import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Reference RGB in [0,1] for serialized colour names.
REFERENCE = {
    "black": (0.05, 0.05, 0.05),
    "white": (0.95, 0.95, 0.95),
    "gray": (0.50, 0.50, 0.50),
    "grey": (0.50, 0.50, 0.50),
    "brown": (0.55, 0.35, 0.17),
    "red": (0.75, 0.12, 0.12),
    "orange": (0.93, 0.53, 0.12),
    "yellow": (0.93, 0.88, 0.20),
    "green": (0.22, 0.58, 0.24),
    "blue": (0.20, 0.35, 0.78),
    "purple": (0.50, 0.20, 0.60),
    "pink": (0.94, 0.60, 0.70),
    "beige": (0.90, 0.85, 0.70),
    "tan": (0.82, 0.71, 0.55),
    "silver": (0.75, 0.75, 0.78),
    # material words that behave like colours in these utterances
    "metal": (0.70, 0.70, 0.72),
    "wooden": (0.60, 0.42, 0.25),
    "wood": (0.60, 0.42, 0.25),
}

# "dark" and "light" are luminance, not hue, so they get their own path.
LUMINANCE = {"dark", "light", "bright"}

# Decay of the RGB distance term. 0.35 puts a mid-grey at ~0.3 against white,
# so a wrong colour is penalised without being annihilated.
SCALE = 0.35


_LUM_W = np.array([0.2126, 0.7152, 0.0722])

# Colours the parses use that are not in the vocabulary above.
SYNONYM = {
    "maroon": "red", "gold": "yellow", "golden": "yellow", "cream": "beige",
    "ivory": "beige", "navy": "blue", "teal": "blue", "turquoise": "blue",
    "olive": "green", "lime": "green", "violet": "purple", "lavender": "purple",
    "charcoal": "black", "steel": "metal", "chrome": "silver",
    "burgundy": "red", "cardboard": "brown", "darker": "dark", "lighter": "light",
}
# "light blue" / "dark brown" / "darker" -- a modifier on a base colour. The base
# carries the match; the modifier is dropped rather than guessed at, since
# lightening a reference RGB is not the same as what a speaker means by it.
_MOD = ("light ", "dark ", "darker ", "lighter ", "bright ", "pale ", "deep ")


# The keys forward() knows how to turn into a score.
_SCORABLE = frozenset(REFERENCE) | LUMINANCE


def _canonical(c: str):
    """One colour string -> a key this encoder can score, or None to stay neutral.

    Words with no single reference RGB -- "colorful", "rainbow", "transparent"
    -- match nothing here and fall through to None, which forward() reads as
    "stay neutral".
    """
    if not c:
        return None
    if c in _SCORABLE:
        return c
    c = SYNONYM.get(c, c)
    if c in _SCORABLE:
        return c
    for m in _MOD:
        if c.startswith(m):
            return _canonical(c[len(m):])
    # "blackish" -> "black"
    if c.endswith("ish"):
        return _canonical(c[:-3])
    # "black and white": a multi-colour object. Take the first named colour --
    # scoring against one of them is better than neutral, and the GMM already
    # represents an object as a mixture so a partial match still scores.
    if " and " in c:
        return _canonical(c.split(" and ")[0])
    # "purple pink", "blue grey": take the first token that names a colour.
    if " " in c:
        for tok in c.split():
            got = _canonical(tok)
            if got:
                return got
    return None


class ColorConcept:
    def __init__(self, gmm_colors: list, obj_ids: list) -> None:
        # gmm_colors is indexed by instance id; obj_ids selects the candidates
        # in the order the rest of the pipeline uses them.
        self.weights = []
        self.means = []
        for i in obj_ids:
            g = gmm_colors[i]
            w = np.asarray(g["weights"], dtype=np.float64)
            m = (np.asarray(g["means"], dtype=np.float64) + 1.0) / 2.0   # -> [0,1]
            self.weights.append(w / max(w.sum(), 1e-9))
            self.means.append(np.clip(m, 0.0, 1.0))

    def forward(self, color: str) -> torch.Tensor:
        c = _canonical((color or "").strip().lower())
        n = len(self.weights)
        # Unknown or multi-colour word: stay neutral rather than guess. 1.0 is
        # the identity for a product and sits above any category floor.
        if c is None:
            return torch.ones(n, dtype=torch.float32, device=DEVICE)
        if c in LUMINANCE:
            out = [self._luminance(i, c) for i in range(n)]
        else:
            ref = np.asarray(REFERENCE[c])
            out = [self._match(i, ref) for i in range(n)]
        return torch.as_tensor(out, dtype=torch.float32, device=DEVICE)

    # Expected similarity under the mixture: sum_k w_k * exp(-||mean_k - ref||/SCALE)
    def _match(self, i: int, ref: np.ndarray) -> float:
        d = np.linalg.norm(self.means[i] - ref[None, :], axis=1)
        return float((self.weights[i] * np.exp(-d / SCALE)).sum())

    def _luminance(self, i: int, word: str) -> float:
        lum = float((self.weights[i] * (self.means[i] @ _LUM_W)).sum())
        return 1.0 - lum if word == "dark" else lum
