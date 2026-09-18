"""Shared benchmark boxes and split metadata."""

from grounding.data.benchmark import BOXES_DIR, load_boxes
from grounding.data.splits import (SPLITS_DEV_PATH, SPLITS_PATH,
                                   load_splits)

__all__ = ["BOXES_DIR", "SPLITS_PATH", "SPLITS_DEV_PATH", "load_boxes",
           "load_splits"]
