"""
    YCSB log parser.
    To produce example output: ./bin/ycsb run basic -P workloads/workloada -p \
        measurementtype=hdrhistogram -p hdrhistogram.percentiles=99,95
    -- romilbhardwaj
    -- kirthevasank
"""

from argparse import ArgumentParser
import logging
import re
from typing import Dict

from cilantro_clients.data_sources.log_parsers.base_log_parser import BaseLogParser
from cilantro_clients.data_sources.metric_extraction_utils import \
    latency_metrics_from_quantile_histogram

# Patterns are of the form {metricname: [regex, type to cast]}
YCSB_PATTERN_DICT = {
    'runtime': [r"\[OVERALL], RunTime\(ms\), ([0-9]+\.?[0-9]*|\.[0-9]+)", float],
    'throughput': [r"\[OVERALL], Throughput\(ops\/sec\), ([0-9]+\.?[0-9]*|\.[0-9]+)", float],
    'num_operations': [r"\[UPDATE], Operations, ([0-9]+)", float],
    'avg_latency': [r"\[UPDATE], AverageLatency\(us\), ([0-9]+)", float],
    'min_latency': [r"\[UPDATE], MinLatency\(us\), ([0-9]+)", float],
    'max_latency': [r"\[UPDATE], MaxLatency\(us\), ([0-9]+)", float],
    'alloc': [r'^([0-9]+)', int], # Alloc is written in the first line of the file
    }

HISTOGRAM_PATTERNS_DICT = {k: [rf"\[UPDATE], {k}..PercentileLatency\(us\), ([0-9]+)", float]
                           for k in range(1,100)}   # Percentile to value map

SIGMA_MULTIPLIER = 1

logger = logging.getLogger(__name__)


class YCSBLogParser(BaseLogParser):
    """ Parser for YCSB. """

    def __init__(self, slo_latency):
        """ Constructor. """
        super().__init__()
        self.slo_latency = slo_latency

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
        # Find metrics -----------------------------------------------------------------------
        for metric, [pattern, t] in YCSB_PATTERN_DICT.items():
            match = re.findall(pattern, log_data)
            if match:
                data[metric] = t(match[0])
            else:
                data[metric] = -1
                err_str = f"Metric {metric} not found in log {log_file}."
                logger.info(err_str)
        # Parse the histogram ----------------------------------------------------------------
        histogram = {}
        for pxx, [pattern, t] in HISTOGRAM_PATTERNS_DICT.items():
            match = re.findall(pattern, log_data)
            if match:
                histogram[pxx/100] = t(match[0])
            else:
                raise LookupError(f"Histogram P{pxx} not found in log {log_file}.")
        # Add the 0th and 100th percentile
        histogram[0] = data['min_latency']
        histogram[1.0] = data['max_latency']
        reward, sigma = latency_metrics_from_quantile_histogram(
            num_events=data['num_operations'], slo_latency=self.slo_latency,
            quantile_histogram=histogram)
        # Return -----------------------------------------------------------------------------
        debug_str = 'runtime: %0.3f, throughput: %0.3f, avg_lat: %0.3f, slo_lat:%0.4f, %s'%(
            data['runtime'], data['throughput'], data['avg_latency'], self.slo_latency,
            str(histogram))
        ret = {'load': data['num_operations'],
               'reward': reward,
               'alloc': data['alloc'],
               'sigma': SIGMA_MULTIPLIER * sigma, # elongate it a bit
               'debug': debug_str,
              }
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
        parser.add_argument('--slo-latency', '-lat', type=int,
                            help='The latency for the SLO.')
        pass

