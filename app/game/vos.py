"""Game value objects."""

from enum import IntEnum


class TileMergingAction(IntEnum):
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3
