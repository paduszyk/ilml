from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from .chemistry import IonicLiquid

if TYPE_CHECKING:
    import pandas as pd
    from ilthermopy.data_structs import Entry as ILTEntry


@dataclass
class ILMLEntry:
    ilt_entry: ILTEntry = field(repr=False)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(id={self.id!r})"

    @property
    def id(self) -> str:
        return self.ilt_entry.id  # type: ignore[no-any-return]

    @property
    def data(self) -> pd.DataFrame:
        return self.ilt_entry.data

    @property
    def ionic_liquid(self) -> IonicLiquid:
        return IonicLiquid.from_smiles(self.ilt_entry.components[0].smiles)
