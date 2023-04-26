"""
    wrk2 output parser.
    Parses the output from wrk2, the http load generator used by deathstarbench
    applications (such as hotel reservation).

    To produce example output: ./wrk -R 100 -D exp -t 16 -c 16 -d 30 -L -s \
    ./scripts/hotel-reservation/mixed-workload_type_1.lua \
    http://frontend.default.svc.cluster.local:5000 > output.txt

    First line is the vector of resource allocations
    Second line is the target qps
    event_start_time: and event_end_time: are added in the file by client
    Following lines are the output from wrk

    This client embeds a lot of data in the debug str as json. Example:
    {
      "runtime": 29.98,
      "throughput": 2976.15,
      "num_operations": 89213,
      "avg_latency": 38.749,
      "stddev_latency": 48.459,
      "p50": 18.29,
      "p90": 112.51,
      "p99": 190.08,
      "p999": 244.74,
      "p9999": 294.4,
      "p100": 331.52,
      "event_start_time": 1649991801.0285008,
      "event_end_time": 1649991831.0650449,
      "target_qps": 3000,
      "load": 89940,
      "allocs": {
        "root--consul": 8,
        "root--frontend": 8,
        "root--memcached-profile": 8,
        "root--memcached-rate": 8,
        "root--memcached-reserve": 8,
        "root--mongodb-profile": 8,
        "root--mongodb-rate": 8,
        "root--mongodb-recommendation": 8,
        "root--mongodb-reservation": 8,
        "root--profile": 8,
        "root--search": 8
      }
    }
    -- romilbhardwaj
"""

import json
from argparse import ArgumentParser
import logging
import re
from typing import Dict

from cilantro_clients.data_sources.log_parsers.base_log_parser import BaseLogParser

def convert_pxx_to_ms(lat_str: str):
    # Converts a pxx str which has units in the end (e.g 100ms or 8.2s) to ms float
    if 'us' in lat_str:
        lat_ms = float(lat_str[:-2])/1000
    elif 'ms' in lat_str:
        lat_ms = float(lat_str[:-2])
    elif 's' in lat_str:
        lat_ms = float(lat_str[:-1])*1000
    else:
        raise Exception(f"Unknown pxx format - got {lat_str}")
    return lat_ms


# Patterns are of the form {metricname: [regex, type to cast]}
WRK_PATTERN_DICT = {
    'runtime': [r" requests in ([0-9]+\.?[0-9]*|\.[0-9]+)", float],
    'throughput': [r"Requests\/sec:\s*([0-9]+\.?[0-9]*|\.[0-9]+)", float],
    'num_operations': [r"([0-9]+) requests in ", float],
    'avg_latency': [r"Mean\s*=\s*([0-9]+\.?[0-9]*|\.[0-9]+)", float],
    'stddev_latency': [r"StdDeviation\s*=\s*([0-9]+\.?[0-9]*|\.[0-9]+)", float],
    'p50': [r"50.000%\s*([0-9]+(?:\.?[0-9]*|\.[0-9])+[A-Za-z]+)", convert_pxx_to_ms],
    'p90': [r"90.000\%\s*([0-9]+(?:\.?[0-9]*|\.[0-9])+[A-Za-z]+)", convert_pxx_to_ms],
    'p99': [r"99.000\%\s*([0-9]+(?:\.?[0-9]*|\.[0-9])+[A-Za-z]+)", convert_pxx_to_ms],
    'p999': [r"99.900\%\s*([0-9]+(?:\.?[0-9]*|\.[0-9])+[A-Za-z]+)", convert_pxx_to_ms],
    'p9999': [r"99.990\%\s*([0-9]+(?:\.?[0-9]*|\.[0-9])+[A-Za-z]+)", convert_pxx_to_ms],
    'p100': [r"100.000\%\s*([0-9]+(?:\.?[0-9]*|\.[0-9])+[A-Za-z]+)", convert_pxx_to_ms],
    'event_start_time': [r"event\_start\_time:([0-9]+\.?[0-9]*|\.[0-9]+)", float],
    'event_end_time': [r"event\_end\_time:([0-9]+\.?[0-9]*|\.[0-9]+)", float]
    }

logger = logging.getLogger(__name__)


class WrkLogParser(BaseLogParser):
    """ Parser for wrk. """

    def __init__(self):
        """ Constructor. """
        super().__init__()

    def get_data(self, log_file) -> Dict:
        """
        This method returns a dict with data by parsing the log_file
        This method also computes the utilities from the raw metrics.
        :return:
        """
        # pylint: disable=not-callable
        data = {}
        with open(log_file, 'r') as f:
            log_data = f.read()
        # Extract first line
        lines = log_data.split('\n')
        allocs = json.loads(lines[0])
        target_qps = float(lines[1])
        # Find metrics -----------------------------------------------------------------------
        for metric, [pattern, t] in WRK_PATTERN_DICT.items():
            match = re.findall(pattern, log_data)
            if match:
                data[metric] = t(match[0])
            else:
                data[metric] = -1
                err_str = f"Metric {metric} not found in log {log_file}."
                logger.info(err_str)
        data['target_qps'] = target_qps
        data['load'] = target_qps*data['runtime']
        data['allocs'] = allocs
        # Return -----------------------------------------------------------------------------
        # print(data)
        print_str = ""
        for key, value in data.items():
            if isinstance(value, float):
                # Format the float to have 3 decimal places
                value_str = "{:.3f}".format(value)
                print_str += f"{key}: {value_str}; "
            else:
                print_str += f"{key}: {value}; "

        ret = {'load': data['num_operations'],
               'reward': data['avg_latency'],
               'alloc': -1,  # Allocs are sent in debug_str
               'sigma': data['stddev_latency'],
               'event_start_time': data['event_start_time'],
               'event_end_time': data['event_end_time'],
               'debug': json.dumps(data)}
        return ret

    @classmethod
    def add_args_to_parser(cls,
                           parser: ArgumentParser):
        """
        Adds class specific arugments to the given argparser.
        Useful to quickly add args to driver script for different sources.
        :param parser: argparse object
        :return: None
        """
        pass

