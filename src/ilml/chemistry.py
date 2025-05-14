from __future__ import annotations

from dataclasses import dataclass, field
from typing import ClassVar, Self

from rdkit import Chem
from rdkit.Chem import Descriptors

from .exceptions import InvalidChargeError


@dataclass
class Ion:
    smiles: str
    rdkit_mol: Chem.Mol = field(repr=False)

    family_smarts_patterns: ClassVar[dict[str, list[str]]] = {}

    def __post_init__(self) -> None:
        if self.charge == 0:
            msg = "ions must have a non-zero charge"

            raise InvalidChargeError(msg)

    @property
    def element_set(self) -> set[str]:
        return {atom.GetSymbol() for atom in self.rdkit_mol.GetAtoms()}  # type: ignore[no-untyped-call]

    @property
    def chemical_family(self) -> str | None:
        for family_name, smarts_patterns in self.family_smarts_patterns.items():
            for smart_pattern in smarts_patterns:
                if self.rdkit_mol.HasSubstructMatch(Chem.MolFromSmarts(smart_pattern)):
                    return family_name

        return None

    @property
    def charge(self) -> int:
        return Chem.GetFormalCharge(self.rdkit_mol)

    @property
    def atom_count(self) -> int:
        return self.rdkit_mol.GetNumAtoms()

    @property
    def molecular_weight(self) -> float:
        return Descriptors.MolWt(self.rdkit_mol)  # type: ignore[attr-defined, no-any-return]

    @classmethod
    def from_smiles(cls, smiles: str) -> Self:
        rdkit_molecule = Chem.MolFromSmiles(smiles)
        smiles = Chem.MolToSmiles(rdkit_molecule, isomericSmiles=False)

        return cls(smiles, rdkit_molecule)


@dataclass
class Cation(Ion):
    family_smarts_patterns: ClassVar[dict[str, list[str]]] = {
        "imidazolium": [
            "c1c[n+]cn1",
        ],
        "pyrazolium": [
            "c1cn[n+]c1",
        ],
        "triazolium": [
            "c1nnc[n+]1",
            "c1c[n+]nn1",
            "c1n[n+]cn1",
        ],
        "thiazolium": [
            "c1csc[n+]1",
        ],
        "quinlolinium": [
            "c1ccc2[n+]cccc2c1",
            "c1ccc2c[n+]ccc2c1",
        ],
        "pyridinium": [
            "c1cc[n+]cc1",
        ],
        "pyrrolidinium": [
            "[#6]-1-[#6]-[#6]-[#7+]-[#6]-1",
        ],
        "piperidinium": [
            "[#6]-1-[#6]-[#6]-[#7+]-[#6]-[#6]-1",
        ],
        "piperazinium": [
            "[#6]-1-[#6]-[#7]-[#6]-[#6]-[#7]-1",
        ],
        "morpholinium": [
            "[#6]-1-[#6]-[#8]-[#6]-[#6]-[#7+]-1",
        ],
        "phosphonium": [
            "[#15+]",
        ],
        "guanidinium": [
            "[#7]-[#6](-[#7])=[#7+]",
        ],
        "amidium": [
            "[!#1!#6][#6]=[#7+]",
        ],
        "ammonium": [
            "[#7+]",
        ],
        "cyclopropenium": [
            "[#7]-[#6]1=,:[#6](-[#7])[#6+]1-[#7]",
        ],
        "cyclic sulfonium": [
            "[#6]-1-[#6]-[#6]-[#16+]-[#6]-1",
            "[#6]-1-[#6]-[#6]-[#16+]-[#6]-[#6]-1",
        ],
        "sulfonium": [
            "[#16+]",
        ],
    }

    def __post_init__(self) -> None:
        super().__post_init__()

        if (charge := self.charge) < 0:
            msg = f"cations must have a positive charge, got {charge:+d}"

            raise InvalidChargeError(msg)


@dataclass
class Anion(Ion):
    family_smarts_patterns: ClassVar[dict[str, list[str]]] = {
        "bistriflamides": [
            "O=S(=O)[#7-]S(=O)=O",
            "[#7-]S(=O)=O",
        ],
        "cyclic amides": [
            "c1c[n-]cn1",
            "c1cn[n-]c1",
            "c1nnn[n-]1",
            "c1c[n-]nn1",
            "c1nc[n-]n1",
            "c1cc[n-]c1",
        ],
        "methanides": [
            "[#6-]",
        ],
        "borates": [
            "[#5-]",
        ],
        "phosphates": [
            "[#15-]",
        ],
        "inorganics": [
            "[F,Cl,Br,I;-]",
            "[#8-]-[#7+](-[#8-])=O",
            "[#16-]C#N",
            "[#8-][Cl+3]([#8-])([#8-])[#8-]",
            "[#7-]=[N+]=[#7-]",
        ],
        "sulfates": [
            "[#8]S([#8-])(=O)=O",
        ],
        "sulfonates": [
            "[#8-]S(=O)=O",
        ],
        "organic phosphates": [
            "[#8]P([#8])([#8-])=O",
            "[#8]-[#15](-[#8-])=O",
            "[#8-]-[#15]=O",
        ],
        "carboxylates": [
            "[#8-]-[#6]=O",
        ],
        "phenolates": [
            "[#8-]-c:1:*:*:*:*:*:1",
        ],
        "carboanions": [
            "[#8-]-[#6]=[#6]",
        ],
        "amides": [
            "[#7-]",
        ],
    }

    def __post_init__(self) -> None:
        super().__post_init__()

        if (charge := self.charge) > 0:
            msg = f"anions must have a negative charge, got {charge:+d}"

            raise InvalidChargeError(msg)


@dataclass
class IonicLiquid:
    cation: Cation
    anion: Anion

    @property
    def smiles(self) -> str:
        return f"{self.cation.smiles}.{self.anion.smiles}"

    @property
    def element_set(self) -> set[str]:
        return self.cation.element_set | self.anion.element_set

    @property
    def atom_count(self) -> int:
        return self.cation.atom_count + self.anion.atom_count

    @property
    def molecular_weight(self) -> float:
        return self.cation.molecular_weight + self.anion.molecular_weight

    @classmethod
    def from_smiles(cls, smiles: str) -> Self:
        left_smiles, right_smiles = smiles.split(".")

        left_ion = Ion.from_smiles(left_smiles)
        right_ion = Ion.from_smiles(right_smiles)

        if left_ion.charge < 0:
            left_ion, right_ion = right_ion, left_ion

        cation = Cation(left_ion.smiles, left_ion.rdkit_mol)
        anion = Anion(right_ion.smiles, right_ion.rdkit_mol)

        return cls(cation, anion)
