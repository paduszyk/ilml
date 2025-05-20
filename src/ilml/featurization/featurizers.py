from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from rdkit.Chem.Descriptors import CalcMolDescriptors

from ilml.memory import cache

from .combining_rules import CombiningRule

if TYPE_CHECKING:
    from ilml.chemistry import Ion, IonicLiquid

rdkit_calc_mol_descriptors = cache(CalcMolDescriptors)

ion_featurizers: dict[str, IonFeaturizer] = {}


def register(name: str) -> Callable[[type[IonFeaturizer]], type[IonFeaturizer]]:
    def decorator(ion_featurizer_class: type[IonFeaturizer]) -> type[IonFeaturizer]:
        ion_featurizers.update({name: ion_featurizer_class()})

        return ion_featurizer_class

    return decorator


class IonFeaturizer(ABC):
    def __call__(self, ion: Ion) -> dict[str, float]:
        features = self._featurize(ion)

        for name, value in features.items():
            try:
                features[name] = float(value)
            except (TypeError, ValueError):
                features[name] = None

        return {name: value for name, value in features.items() if value is not None}

    @abstractmethod
    def _featurize(self, ion: Ion) -> dict[str, Any]:
        pass


@register("RDKit")
class RDKitIonFeaturizer(IonFeaturizer):
    def _featurize(self, ion: Ion) -> dict[str, Any]:
        return rdkit_calc_mol_descriptors(ion.rdkit_mol)  # type: ignore[no-any-return]


class IonicLiquidFeaturizer:
    def __init__(
        self,
        ion_featurizer: IonFeaturizer,
        combining_rule: CombiningRule | None = None,
    ) -> None:
        self.ion_featurizer = ion_featurizer
        self.combining_rule = combining_rule

    def __call__(self, ionic_liquid: IonicLiquid) -> dict[str, float]:
        cation_features = self.ion_featurizer(ionic_liquid.cation)
        anion_features = self.ion_featurizer(ionic_liquid.anion)

        if (combining_rule := self.combining_rule) is not None:
            feature_names = set(cation_features) | set(anion_features)

            ionic_liquid_features = {}
            for name in feature_names:
                if not (
                    (cation_feature := cation_features[name]) is not None
                    and (anion_feature := anion_features.get(name)) is not None
                ):
                    continue

                ionic_liquid_features[name] = combining_rule(
                    ionic_liquid,
                    cation_feature,
                    anion_feature,
                )
        else:
            ionic_liquid_features = {
                **{f"{name}_cation": value for name, value in cation_features.items()},
                **{f"{name}_anion": value for name, value in anion_features.items()},
            }

        return ionic_liquid_features
