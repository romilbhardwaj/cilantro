import asyncio


class BaseEventSource(object):
    def __init__(self, output_queue: asyncio.Queue):
        '''
        Base event source class
        :param output_queue: asyncio.Queue object passed by the main loop. This is where the events will be inserted.
        '''
        self.output_queue = output_queue

    async def event_generator(self):
        '''
        Long running loop that generates events indefinitely
        :return:
        '''
        pass