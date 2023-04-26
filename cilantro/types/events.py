"""
    Defines events used in Cilantro
    -- romilbhardwaj
    -- kirthevasank
"""

import time
from enum import Enum


class EventTypes(Enum):
    """ Event types. """
    MISC = 0
    UTILITY_UPDATE = 1
    APP_ADDED = 2
    APP_REMOVED = 3
    NODE_ADDED = 4
    NODE_REMOVED = 5
    ALLOC_TIMEOUT = 6
    SIGNIFICANT_LOAD_CHANGE = 7


class BaseEvent:
    """ Basic event abstraction. """

    def __init__(self,
                 timestamp: float = None,
                 event_type: EventTypes = EventTypes.MISC):
        """
        Base event.
        :param timestamp: When the event was created (float)
        :param event_type: Enum of EventTypes
        """
        if timestamp is None:
            timestamp = time.time()
        self.timestamp = timestamp
        self.event_type = event_type

    def __repr__(self):
        """ representation. """
        return f"BaseEvent, type:{self.event_type}"


class AllocExpirationEvent(BaseEvent):
    """ Allocation timeout event. """

    def __init__(self,
                 allocation_informer: object = None,
                 timestamp: float = None,
                 event_type: EventTypes = EventTypes.ALLOC_TIMEOUT):
        """
        Allocation timeout event event. Used in child classes (AppAddEvent)
        :param app_path: Hierarchy path to the app
        :param timestamp:
        :param event_type:
        """
        self._allocation_informer = allocation_informer
        super().__init__(timestamp, event_type)

    def inform_event_source_of_allocation(self):
        """ Informs the event source of the allocation. """
        self._allocation_informer()
        self._allocation_informer = None # In case, there is a memory leak.
        # TODO (@Romilb): Check if this is a better way to do this.

    def __repr__(self):
        """ representation. """
        return f"Allocation timeout, type:{self.event_type}"


class UtilityUpdateEvent(BaseEvent):
    """ Utility update event. """

    def __init__(self,
                 app_path: str = None,
                 load: float = 1,
                 reward: float = 1,
                 alloc: float = 1,
                 sigma: float = 1,
                 event_start_time: float = 1,
                 event_end_time: float = 1,
                 timestamp: float = None,
                 event_type: EventTypes = EventTypes.UTILITY_UPDATE,
                 debug: str = ""):
        """
        Event to report utility updates.
        :param app_path: Hierarchy path to the app
        :param load: Load reported by app
        :param reward: Reward reported by app
        :param alloc: Allocation reported by app
        :param sigma: Sigma reported by app
        :param timestamp:
        :param event_type:
        :param debug: Debug string optionally sent by client.
        """
        self.app_path = app_path
        self.load = load
        self.reward = reward
        self.alloc = alloc
        self.sigma = sigma
        self.event_start_time = event_start_time
        self.event_end_time = event_end_time
        self.debug = debug
        super().__init__(timestamp, event_type)

    def __repr__(self):
        """ representation. """
        return f"UtilityUpdateEvent, app: {self.app_path}, load: {self.load}, " + \
               f"reward: {self.reward}," + \
               f"alloc: {self.alloc}, sigma: {self.sigma}, type:{self.event_type}, debug: {self.debug}"

    def __dict__(self):
        """ dictionary. """
        return {'timestamp': self.timestamp,
                'load': self.load,
                'reward': self.reward,
                'alloc': self.alloc,
                'sigma': self.sigma,
                'event_start_time': self.event_start_time,
                'event_end_time': self.event_end_time,
                'debug': self.debug
        }


class AppUpdateEvent(BaseEvent):
    """ App Update event. """

    def __init__(self,
                 app_path: str = None,
                 timestamp: float = None,
                 event_type: EventTypes = EventTypes.APP_ADDED):
        """
        App update base event. Used in child classes (AppAddEvent)
        :param app_path: Hierarchy path to the app
        :param timestamp:
        :param event_type:
        """
        self.app_path = app_path
        super().__init__(timestamp, event_type)

    def __repr__(self):
        """ representation. """
        return f"AppUpdateEvent, app: {self.app_path}, type:{self.event_type}"


class AppAddEvent(AppUpdateEvent):
    """ App Add event. """

    def __init__(self,
                 app_path: str = None,
                 app_threshold: float = None,
                 app_weight: float = None,
                 app_unit_demand: float = None,
                 timestamp: float = None,
                 event_type: EventTypes = EventTypes.APP_ADDED):
        """
        App update base event. Used in child classes (AppAddEvent)
        :param app_path: Hierarchy path to the app
        :param app_threshold: Threshold (SLO) for the app
        :param app_weight: Weight in the hierarchy wrt. siblings.
        :param timestamp:
        :param event_type:
        """
        self.app_threshold = app_threshold
        self.app_weight = app_weight
        self.app_unit_demand = app_unit_demand
        super().__init__(app_path, timestamp, event_type)

    def __repr__(self):
        """ representation. """
        return f"AppAddEvent, app: {self.app_path}, threshold: {self.app_threshold}," \
               f"weight: {self.app_weight}, type:{self.event_type}"
