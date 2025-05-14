from __future__ import annotations


class ILMLError(Exception):
    pass


class ILMLEntryError(ILMLError):
    pass


class ILMLDatasetError(ILMLError):
    pass


class EntriesNotFoundError(ILMLDatasetError):
    pass


class ILTEntryProcessingError(ILMLDatasetError):
    pass


class ChemistryError(ILMLError):
    pass


class IonError(ChemistryError):
    pass


class InvalidChargeError(IonError):
    pass
