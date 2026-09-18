"""Map serialized concept names to engine-supplied encoder classes.

Arity includes the target; rel_num stores the number of anchors (arity minus
one). The table retains baseline concepts and thesis extensions without
importing either engine."""

from __future__ import annotations

from dataclasses import dataclass

__all__ = ["Vocabulary", "ROOM_CONCEPTS", "THESIS_PREDICATES"]

#: Unary concepts whose encoder takes the room extent as a second argument.
#: Upstream reads seven scalars out of the point cloud for these; everything
#: else is computed from the object boxes alone.
ROOM_CONCEPTS = frozenset({"corner", "against wall", "on the floor"})

#: Predicate names this thesis adds to upstream's vocabulary, and why.
#: These are *names*, not classes -- what each binds to is the harness' business.
THESIS_PREDICATES: dict[str, str] = {
    # Without these an ordinal of smallest/lowest lowers to a relation_name with
    # no encoder, normalise_predicate returns None, and the constraint is
    # dropped silently. Upstream has only the 'largest'/'highest' ends.
    "smallest": "the small end of the size axis",
    "lowest": "the low end of the height axis",
    # Upstream has no isolation encoder at all, so every phrasing below lowers
    # to a predicate the baseline drops.
    "isolated": "isolation",
    "isolation": "isolation",
    "alone": "isolation",
    "by itself": "isolation",
    "lone": "isolation",
}


@dataclass(frozen=True)
class Vocabulary:
    """A predicate table, with the arities derived once.

    ``classes``     name -> encoder class (opaque to this package; only called)
    ``arity``       encoder class -> number of objects it relates
    ``rel_num``     name -> number of ANCHORS it takes (arity - 1)
    """

    classes: dict[str, type]
    arity: dict[type, int]
    rel_num: dict[str, int]

    @classmethod
    def from_tables(cls, classes: dict[str, type], arity: dict[type, int]) -> "Vocabulary":
        missing = {c.__name__ for c in classes.values() if c not in arity}
        if missing:
            raise ValueError(f"no arity for encoder classes: {sorted(missing)}")
        # rel_num is the ANCHOR count, one less than the arity: a unary concept
        # ("against wall") relates 1 object and takes 0 anchors; "left" relates 2
        # and takes 1; "between" relates 3 and takes 2. The resolver switches on
        # this value, so the -1 is load-bearing.
        rel_num = {name: arity[c] - 1 for name, c in classes.items()}
        return cls(classes=dict(classes), arity=dict(arity), rel_num=rel_num)

    @property
    def names(self):
        """Every valid predicate name. Kept as the dict keys view that the
        harness has always exposed as ALL_VALID_RELATIONS."""
        return self.classes.keys()

    def encoder_for(self, name: str) -> type:
        return self.classes[name]

    def is_a(self, name: str, encoder: type) -> bool:
        """Does `name` bind to this exact encoder class?

        How the builder recognises the encoders that need non-standard
        arguments, without hard-coding predicate names -- ``against wall`` and
        ``isolated`` are each reachable under several spellings.
        """
        return self.classes.get(name) is encoder
