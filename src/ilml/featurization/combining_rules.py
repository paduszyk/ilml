from __future__ import annotations

from collections.abc import Callable

from ilml.chemistry import IonicLiquid

type CombiningRule = Callable[[IonicLiquid, float, float], float]

combining_rules: dict[str, CombiningRule | None] = {"concatenate": None}


def register(func: CombiningRule) -> CombiningRule:
    combining_rules.update({func.__name__: func})

    return func


@register
def sum(
    ionic_liquid: IonicLiquid,
    cation_feature: float,
    anion_feature: float,
) -> float:
    return cation_feature + anion_feature


@register
def min_abs(
    ionic_liquid: IonicLiquid,
    cation_feature: float,
    anion_feature: float,
) -> float:
    return float(min(abs(cation_feature), abs(anion_feature)))


@register
def max_abs(
    ionic_liquid: IonicLiquid,
    cation_feature: float,
    anion_feature: float,
) -> float:
    return float(max(abs(cation_feature), abs(anion_feature)))


@register
def mean(
    ionic_liquid: IonicLiquid,
    cation_feature: float,
    anion_feature: float,
) -> float:
    return 0.5 * (cation_feature + anion_feature)


@register
def mean_atom_count(
    ionic_liquid: IonicLiquid,
    cation_feature: float,
    anion_feature: float,
) -> float:
    return (
        (ionic_liquid.cation.atom_count * cation_feature)
        + (ionic_liquid.anion.atom_count * anion_feature)
    ) / ionic_liquid.atom_count


@register
def mean_molecular_weight(
    ionic_liquid: IonicLiquid,
    cation_feature: float,
    anion_feature: float,
) -> float:
    return (
        (ionic_liquid.cation.molecular_weight * cation_feature)
        + (ionic_liquid.anion.molecular_weight * anion_feature)
    ) / ionic_liquid.molecular_weight
