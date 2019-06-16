# -*- coding: utf-8 -*-

import enum

class RunningType(enum.IntEnum):
    train=0
    valid=1
    test=2
    predict=3
