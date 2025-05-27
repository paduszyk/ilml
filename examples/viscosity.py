from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import ilthermopy as ilt
import numpy as np
import pandas as pd
from tqdm import tqdm

from ilml.featurization import IonicLiquidFeaturizer, ion_featurizers, combining_rules
from ilml.chemistry import IonicLiquid
from ilml.dataset import ILMLDataset
from ilml.exceptions import ILTEntryProcessingError

if TYPE_CHECKING:
    from ilthermopy.data_structs import Entry as ILTEntry

DATAFILES_DIR = Path(__file__).parent / "datafiles"


class Dataset(ILMLDataset):
    @staticmethod
    def get_ilt_entry_ids() -> list[str]:
        # Search for all pure IL entries with viscosity data.
        search = ilt.Search(prop="Viscosity", n_compounds=1)

        # Select only entries with 1 phase, liquid phase, and binary ILs SMILES.
        search = search[
            (search["num_phases"] == 1)
            & (search["phases"] == "Liquid")
            & search["cmp1_smiles"].apply(lambda smiles: len(smiles.split(".")) == 2)
        ]

        return list(search["id"])

    @staticmethod
    def process_ilt_entry(ilt_entry: ILTEntry) -> ILTEntry:
        # Update column names of the entry's data.
        data = ilt_entry.data.copy().rename(columns=ilt_entry.header)

        # Check the chemical composition of the IL associated with the entry.
        allowed_elements = {
            *("C", "H", "N", "O", "P", "S"),
            *("F", "Cl", "Br", "I"),
        }

        ionic_liquid_smiles = ilt_entry.components[0].smiles
        ionic_liquid = IonicLiquid.from_smiles(ionic_liquid_smiles)

        if disallowed_elements := (ionic_liquid.element_set - allowed_elements):
            msg = (
                f"entry {ilt_entry.id!r} contains disallowed elements: "
                f"{', '.join(disallowed_elements)}"
            )

            raise ILTEntryProcessingError(msg)

        # Check the data header and perform preliminary transformations.
        try:
            data = pd.DataFrame(
                {
                    "T_K": data["Temperature, K"],
                    "p_kPa": data.get("Pressure, kPa", 101),
                    "log10_viscosity_mPa_s": np.log10(data["Viscosity, Pa&#8226;s => Liquid"] * 1000),
                },
            )  # fmt: skip
        except KeyError as exc:
            msg = f"missing required columns in data for entry {ilt_entry.id!r}"

            raise ILTEntryProcessingError(msg) from exc

        # Check temperature range.
        # Filter out data on temperatures other than (298 +/- 1) K.
        if (data := data.loc[data["T_K"].between(297, 299, inclusive="both"), :]).empty:
            msg = f"entry {ilt_entry.id!r} contains no data at (298 +/- 1) K"

            raise ILTEntryProcessingError(msg)

        # Check pressure range.
        # The dataset should only contain atmospheric pressure data.
        if (
            data["p_kPa"].nunique() > 1
            or not data["p_kPa"].between(100, 102, inclusive="left").all()
        ):
            msg = f"entry {ilt_entry.id!r} contains non-atmospheric pressure data"

            raise ILTEntryProcessingError(msg)

        # Since the data are at constant temperature and pressure, the respective
        # columns can be safely dropped.
        data = data.drop(columns=["T_K", "p_kPa"])

        # Reset data index.
        data = data.reset_index(drop=True)

        # Finally reassign data to the entry.
        ilt_entry.data = data

        return ilt_entry

    def get_data(self) -> pd.DataFrame:
        data = super().get_data()

        ionic_liquid_smiles = pd.Series(index=data.index, dtype="string")
        cation_family = pd.Series(index=data.index, dtype="string")
        anion_family = pd.Series(index=data.index, dtype="string")

        for entry_id in data.index.unique():
            entry = self.get_entry(entry_id)

            ionic_liquid_smiles.loc[entry_id] = entry.ionic_liquid.smiles
            cation_family.loc[entry_id] = entry.ionic_liquid.cation.chemical_family
            anion_family.loc[entry_id] = entry.ionic_liquid.anion.chemical_family

        data.insert(0, "ionic_liquid_smiles", ionic_liquid_smiles)
        data.insert(1, "cation_family", cation_family)
        data.insert(2, "anion_family", anion_family)

        data.reset_index(inplace=True, drop=False)
        data.set_index(
            keys=[
                "entry_id",
                "ionic_liquid_smiles",
                "cation_family",
                "anion_family",
            ],
            inplace=True,
        )

        return data


if __name__ == "__main__":
    # Instance the dataset.
    dataset = Dataset()

    # Get the data.
    data = dataset.get_data()

    # For each ionic liquid get entry IDs.
    entry_ids = pd.DataFrame.from_dict(
        {
            ionic_liquid_smiles: ",".join(
                data.xs(ionic_liquid_smiles, level="ionic_liquid_smiles")
                .index.get_level_values("entry_id")
                .unique()
                .to_list()
            )
            for ionic_liquid_smiles in data.index.get_level_values(
                "ionic_liquid_smiles"
            ).unique()
        },
        orient="index",
        columns=["entry_ids"],
    )

    # Group the data by ionic liquid and average.
    data = data.groupby(
        level=[
            "ionic_liquid_smiles",
            "cation_family",
            "anion_family",
        ]
    ).mean()

    data = data.join(entry_ids, on="ionic_liquid_smiles")

    # Featurize.
    if not DATAFILES_DIR.exists():
        DATAFILES_DIR.mkdir(parents=True, exist_ok=True)

    for ion_featurizer_name, ion_featurizer in ion_featurizers.items():
        for combining_rule_name, combining_rule in combining_rules.items():
            # Create a new design table for each combination of ion featurizer and
            # combining rule.
            design_table = data.copy()

            # Featurize the dataset.
            featurize = IonicLiquidFeaturizer(
                ion_featurizer=ion_featurizer,
                combining_rule=combining_rule,
            )

            ionic_liquid_features = {}
            for ionic_liquid_smiles in tqdm(
                data.index.get_level_values("ionic_liquid_smiles"),
                desc="Featurizing",
            ):
                ionic_liquid = IonicLiquid.from_smiles(ionic_liquid_smiles)

                ionic_liquid_features[ionic_liquid_smiles] = featurize(ionic_liquid)

            # Convert the features to a DataFrame.
            features = pd.DataFrame.from_dict(ionic_liquid_features, orient="index")

            # Merge the features with the data.
            design_table = data.join(features, on="ionic_liquid_smiles").dropna(axis=1)

            print(design_table.shape)

            # Save the design table to a CSV file.
            design_table.to_csv(
                DATAFILES_DIR
                / f"viscosity_{ion_featurizer_name}_{combining_rule_name}.csv",
                index=True,
            )
