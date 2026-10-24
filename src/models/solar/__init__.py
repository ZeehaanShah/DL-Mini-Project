"""Solar PV nowcasting models and baselines for SKIPP'D."""
from .baselines import (
    GlobalMeanBaseline,
    GlobalMedianBaseline,
    HourOfDayClimatologyBaseline,
    ConditionalPersistenceBaseline,
)
from .resnet import ResNet18Nowcaster

__all__ = [
    "GlobalMeanBaseline",
    "GlobalMedianBaseline",
    "HourOfDayClimatologyBaseline",
    "ConditionalPersistenceBaseline",
    "ResNet18Nowcaster",
]
