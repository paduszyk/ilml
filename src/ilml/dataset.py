from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import ilthermopy as ilt
import pandas as pd
from tqdm import tqdm

from .entry import ILMLEntry
from .exceptions import EntriesNotFoundError, ILTEntryProcessingError
from .memory import cache

if TYPE_CHECKING:
    from ilthermopy.data_structs import Entry as ILTEntry

ilt_get_entry = cache(ilt.GetEntry)


@dataclass
class ILMLDataset(ABC):
    entries: list[ILMLEntry] = field(default_factory=list, repr=False)

    def __post_init__(self) -> None:
        self._populate()

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(entry_count={len(self.entries)})"

    @staticmethod
    @abstractmethod
    def get_ilt_entry_ids() -> list[str]:
        pass

    @staticmethod
    @abstractmethod
    def process_ilt_entry(ilt_entry: ILTEntry) -> ILMLEntry:
        pass

    def get_entry(self, entry_id: str) -> ILMLEntry:
        for entry in self.entries:
            if entry.id == entry_id:
                return entry

        msg = f"could not find entry with ID {entry_id!r}"

        raise LookupError(msg)

    def get_data(self) -> pd.DataFrame:
        if not (entries := self.entries):
            msg = "dataset is empty"

            raise EntriesNotFoundError(msg)

        return pd.concat(
            {entry.id: entry.data for entry in entries},
            names=[
                "entry_id",
                "datapoint_id",
            ],
        ).reset_index("datapoint_id", drop=True)

    def get_references(self) -> pd.DataFrame:
        references = {}

        for entry in self.entries:
            ref = entry.ilt_entry.ref

            references[entry.id] = ref.full

        references_df = pd.DataFrame.from_dict(
            references,
            orient="index",
            columns=["reference"],
        )
        references_df.index.name = "entry_id"

        return references

    def _populate(self) -> None:
        ilt_entry_ids = self.get_ilt_entry_ids()

        for ilt_entry_id in tqdm(ilt_entry_ids):
            ilt_entry = ilt_get_entry(ilt_entry_id)

            try:
                ilt_entry = self.process_ilt_entry(ilt_entry)
            except ILTEntryProcessingError:
                continue

            entry = ILMLEntry(ilt_entry)

            self.entries.append(entry)
