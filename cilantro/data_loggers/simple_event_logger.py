"""
    A simple event logger - provides no persistence.
"""
import collections

import pandas as pd
import numpy as np
# Local
from cilantro.data_loggers.base_data_logger import BaseDataLogger


class SimpleEventLogger(object):
    def __init__(self, max_len=1000):
        """
        Constructor.
        """
        self.events = collections.deque(maxlen=max_len)
        super(SimpleEventLogger, self).__init__()

    def log_event(self, event):
        """
        Do nothing, just store event in memory and flush the list if exceeds size.
        """
        self.events.append(event)
