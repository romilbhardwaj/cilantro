import asyncio
import time
from typing import List

from cilantro.backends.base_event_source import BaseEventSource
from cilantro.backends.base_framework_manager import BaseFrameworkManager
from cilantro.types.events import UtilityUpdateEvent, AppUpdateEvent, EventTypes, AppAddEvent


class DummyEventSource(BaseEventSource):
    def __init__(self, output_queue, sleep_time, app_name='dummy'):
        """
        Generates events at a fixed frequency
        :param sleep_time: time to sleep
        """
        self.sleep_time = sleep_time
        self.app_name = app_name
        self.last_event_time = time.time()
        super(DummyEventSource, self).__init__(output_queue)


    async def event_generator(self):
        """
        Long running loop that generates events indefinitely
        :return:
        """
        while True:
            now = time.time()
            event = UtilityUpdateEvent(app_path=self.app_name, event_end_time=now, event_start_time=self.last_event_time)
            self.last_event_time = now
            await self.output_queue.put(event)
            await asyncio.sleep(self.sleep_time)

class DummyFrameworkManager(BaseFrameworkManager):
    def __init__(self,
                 event_queue: asyncio.Queue,
                 default_jobs: List[str],
                 cluster_resources: float = 1,
                 alloc_granularity: float = 1):
        """
        Dummy framework manager for testing without running kubernetes.
        Adds default jobs as AppUpdateEvents at the start to simulate app discovery.
        :param event_queue: Event queue to populate with initial app update events.
        :param cluster_resources: Resource count to return whenever queried
        :param default_jobs: List of jobs to emulate in the start.
        """
        self.event_queue = event_queue
        self.default_jobs = default_jobs
        self.cluster_resources = cluster_resources
        self.alloc_granularity = alloc_granularity
        super(DummyFrameworkManager, self).__init__()

        self._generate_init_events()

    def _generate_init_events(self):
        for j in self.default_jobs:
            e = AppAddEvent(app_path=j,
                            app_threshold=1,
                            app_weight=1,
                            timestamp=time.time(),
                            event_type=EventTypes.APP_ADDED)
            self.event_queue.put_nowait(e)

    def apply_allocation(self, allocation):
        # Do nothing, just check if the returned keys are valid.
        for j in allocation.keys():
            assert j in self.default_jobs

    def get_cluster_resources(self,
                              resource_label: str = 'cpu'):
        return self.cluster_resources

    def get_alloc_granularity(self,
                              resource_label: str = 'cpu'):
        return self.alloc_granularity
