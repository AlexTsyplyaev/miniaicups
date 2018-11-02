import os
import random


def toint(value, default=None):
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


TICKS_TO_DEADLINE = toint(os.getenv('TICKS_TO_DEADLINE'), 600)
MAX_EXECUTION_TIME = REQUEST_MAX_TIME = 2147483646 # max C long -1
MAX_TICK_COUNT = toint(os.getenv('MAX_TICK_COUNT'), 20000)
SEED = toint(os.getenv('SEED'), random.randint(0, 2**128))
REST_TICKS = toint(os.getenv('REST_TICKS'), 90)

random.seed(SEED)
