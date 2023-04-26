"""
    An event source for allocation timeouts.
    -- romilbhardwaj
    -- kirthevasank
"""

import asyncio
import time
# Local
from cilantro.backends.base_event_source import BaseEventSource
from cilantro.types.events import AllocExpirationEvent


class AllocExpirationEventSource(BaseEventSource):
    """ An allocation timeout event source. """

    def __init__(self, output_queue, timeout_time, sleep_time=None):
        """ Constructor. """
        self.timeout_time = timeout_time
        self.sleep_time = sleep_time if sleep_time else timeout_time/3
        self.last_alloc_or_event_gen_time = time.time() - self.timeout_time
        super().__init__(output_queue)

    # TODO (@Romilb): should the following be async def set_last_alloc_time(...) etc.?
    def set_last_alloc_time(self, last_alloc_time):
        """ Set the last allocation time. """
        if last_alloc_time > self.last_alloc_or_event_gen_time:
            self.last_alloc_or_event_gen_time = last_alloc_time

    def allocation_made_by_scheduler(self):
        """ Informs this event source that an allocation was made by the scheduler. """
        self.set_last_alloc_time(time.time())

    async def event_generator(self):
        """ Generates the event. """
        while True:
            curr_time = time.time()
            if curr_time >= self.last_alloc_or_event_gen_time + self.timeout_time:
                event = AllocExpirationEvent(allocation_informer=self.allocation_made_by_scheduler)
                # TODO (@Romilb): Sending a handle to allocation_made_by_scheduler to
                # CilantroScheduler since I want to be able to inform the event source that an
                # allocation was made (this ensures it waits for sometime before generating another
                # event).
                self.last_alloc_or_event_gen_time = curr_time
                await self.output_queue.put(event)
            await asyncio.sleep(self.sleep_time)

