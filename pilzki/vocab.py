"""Trait vocabulary the classifier predicts.

Mirrors `partChoices` / `groupTags` in pilzdaten/internal/web/static/js/config.js.
Only traits a forager can see on an ordinary photo in the forest and confirm
by looking are listed. Left out on purpose:
- not visible at all: smell, taste, edibility, spore print, microscopy,
  colour change, milk, season, rarity;
- needs touching, cutting, digging or waiting: slimy, hard, flesh colour,
  hollow stem, rooting stem, hygrophanous;
- too fine for a typical photo: gill attachment, gill spacing/forking/edge,
  snakeskin pattern.
The species head could still "guess" such tags via the species, but the user
could not verify them, so they are never prefilled.

`exclusive` groups allow a single tag in the web UI (a new cap colour replaces
the old one), so prefill picks at most one tag per exclusive group.
`requires` names a tag that must be likely (p >= REQUIRE_P) before the group is
prefilled at all, e.g. gill colour only makes sense for a cap-and-stem mushroom.
"""

from __future__ import annotations

from dataclasses import dataclass

COLORS = ["weiß", "gelb", "braun", "schwarz", "rot", "orange", "blau", "gruen", "rosa", "grau"]
REQUIRE_P = 0.5


@dataclass(frozen=True)
class Group:
    name: str
    part: str  # part of the mushroom drawing in the web UI (config.js partNames)
    exclusive: bool
    tags: tuple[str, ...]
    requires: str | None = None


GROUPS: tuple[Group, ...] = (
    Group("gruppe", "body", True, ("hutpilz", "kugel", "hut_koralle", "konsistenz_gallertartig", "f", "sonstige_gattung")),
    Group("hut_farbe", "cap", True, tuple(COLORS)),
    Group("hut_form", "cap", True, ("spitz", "schirm", "trichter", "sonstige_form"), requires="hutpilz"),
    Group("hut_oberflaeche", "cap", False, ("flocken", "hut_gerieft")),
    Group("unterseite", "underside", True, ("lamellen", "roehren", "stachel", "leisten"), requires="hutpilz"),
    Group("unterseite_farbe", "underside", True, tuple(f"lr_{c}" for c in COLORS), requires="hutpilz"),
    Group("stiel_farbe", "stem", True, tuple(f"stiel_{c}" for c in COLORS), requires="hutpilz"),
    Group("stiel_details", "stem", False, ("ring", "knollig", "netz"), requires="hutpilz"),
    Group("untergrund", "growth", True, ("auf_boden", "auf_wiese", "holz")),
    Group("wuchs", "growth", False, ("bueschelig",)),
)

TAGS: tuple[str, ...] = tuple(tag for group in GROUPS for tag in group.tags)
TAG_GROUP: dict[str, str] = {tag: group.name for group in GROUPS for tag in group.tags}

# Tags with a fruiting layer under a cap; mirrors hatTags in internal/catalog/field.go.
HAT_TAGS = frozenset({"lamellen", "roehren", "stachel", "leisten"})

assert len(TAGS) == len(set(TAGS)), "a tag may belong to one group only"


def as_json() -> list[dict]:
    return [
        {"name": g.name, "part": g.part, "exclusive": g.exclusive, "requires": g.requires, "tags": list(g.tags)}
        for g in GROUPS
    ]
