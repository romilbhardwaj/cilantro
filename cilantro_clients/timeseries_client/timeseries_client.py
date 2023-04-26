import argparse
import logging
import os
import time
import grpc
import pandas
from cilantro.backends.grpc.protogen import utility_update_pb2_grpc, utility_update_pb2

logger = logging.getLogger(__name__)

class TimeseriesClient(object):
    def __init__(self,
                 client_id: str,
                 csv_file: str,
                 ip: str,
                 port: int,
                 roll_over: bool = True):
        '''
        Dummy client for the cilantro scheduler that reads a CSV file and
        sends data as utility updates to the cilantro scheduler.
        :para
        :param csv_file: path to csv file. CSV format is [time, load, reward]
        :param ip: IP of GRPC server to publish samples at
        :param port: GRPC server port
        :param roll_over: Loops infinitely if set true
        '''
        self.client_id = client_id
        self.ip = ip
        self.port = port
        self.csv_file = csv_file
        self.roll_over = roll_over

        self.df = pandas.read_csv(csv_file)
        self.max_time = max(self.df['time'])
        self.last_get_time = 0  # Time stamp of last sample get call
        self.start_time = time.time()

    def set_start_time(self):
        self.start_time = time.time()

    def get_samples_range(self, start_time, end_time):
        idxs = self.df['time'].between(start_time, end_time)
        return self.df[idxs]

    def get_samples(self):
        now = time.time()
        start_time = self.last_get_time-self.start_time
        end_time = now - self.start_time
        if self.roll_over:
            # Assuming get_samples is called more frequently than max_time, roll over condition is a simple %
            start_time %= self.max_time
            end_time %= self.max_time
        if not self.roll_over and (end_time > self.max_time):
            raise StopIteration('All values have been iterated over.')
        samples = self.get_samples_range(start_time=start_time,
                                         end_time=end_time)
        self.last_get_time = now
        return samples

    def publish_samples(self, samples):
        with grpc.insecure_channel(f'{self.ip}:{self.port}') as channel:
            stub = utility_update_pb2_grpc.UtilityMessagingStub(channel)
            for index, sample in samples.iterrows():
                load = float(sample['load'])
                utility = float(sample['reward'])
                logger.info(f"Publishing load: {load}, reward: {utility}")
                ret = stub.PublishUtility(utility_update_pb2.UtilityMessage(app_id=self.client_id,
                                                                            utility=utility,
                                                                            load=load))

    def run_loop(self):
        while True:
            try:
                samples = self.get_samples()
            except StopIteration as e:
                logger.info(f"{str(e)}")
                break
            if len(samples) > 0:
                self.publish_samples(samples)
            time.sleep(0.1)

if __name__ == '__main__':
    def parseargs():
        parser = argparse.ArgumentParser(description='Timeseries client that generates and sends utility messages from a csv.')
        parser.add_argument('--port', '-p', type=int, default=None, help='GRPC Port')
        parser.add_argument('--ip', '-i', type=str, default=None, help='GRPC IP address')
        parser.add_argument('--client-id', '-c', type=str, default="TSClient",
                            help='Name to be used for publishing utility messages.')
        parser.add_argument('--csv-file', '-f', type=str, default="",
                            help='Path to CSV file.')
        parser.add_argument('--roll-over', '-r', action="store_true", default=False,
                            help='If specified, rolls over the CSV file to produce infinite stream.')
        args = parser.parse_args()
        return args

    logging.basicConfig(level=logging.DEBUG)
    args = parseargs()

    if args.ip is None:
        # If ip is not specified, assume running inside k8s cluster and find the service
        args.ip = os.environ['CILANTRO_SERVICE_SERVICE_HOST']

    if args.port is None:
        args.port = os.environ['CILANTRO_SERVICE_SERVICE_PORT']

    ts = TimeseriesClient(client_id=args.client_id,
                          csv_file=args.csv_file,
                          ip=args.ip,
                          port=args.port,
                          roll_over=args.roll_over)
    ts.set_start_time()
    ts.run_loop()