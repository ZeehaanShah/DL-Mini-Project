"""Electricity demand forecasting models and baselines for BDG2."""
from .baselines import (
    DemandPersistenceBaseline,
    DemandSeasonalNaiveBaseline,
    DemandWeeklySeasonalNaiveBaseline,
)
from .sequence_models import DemandLSTM

__all__ = [
    "DemandPersistenceBaseline",
    "DemandSeasonalNaiveBaseline",
    "DemandWeeklySeasonalNaiveBaseline",
    "DemandLSTM",
]
