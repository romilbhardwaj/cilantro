"""
    Ray log parser.
    -- romilbhardwaj
    -- kirthevasank
"""

from argparse import ArgumentParser
import json
import logging
import numpy as np
# Cilantro
from cilantro_clients.data_sources.log_parsers.base_log_parser import BaseLogParser
from cilantro_clients.data_sources.metric_extraction_utils import latency_metrics_from_e2e_latencies


LATENCY_SIGMA_MULTIPLIER = 10
NUM_LAST_LATENCY_JOBS_TO_IGNORE = 10 # Because those dispatched at the end don't finish on time.
MAX_THROUGHPUT_COEFF = 1


logger = logging.getLogger(__name__)


class CrayLogParser(BaseLogParser):
    """ Parser for YCSB. """

    def __init__(self, slo_type, slo_latency=-1, max_throughput=-1):
        """ Constructor. """
        super().__init__()
        self.slo_type = slo_type
        if slo_type == 'latency':
            assert slo_latency > 0
            self.slo_latency = slo_latency
        if slo_type == 'throughput':
            self.max_throughput = max_throughput

    @classmethod
    def _add_common_metrics(cls, data, result_dict):
        """ Adds common metrics. """
        num_succ_completions = data['tasks_completed_count']
        event_duration = result_dict['event_end_time'] - result_dict['event_start_time']
        result_dict['throughput'] = num_succ_completions / event_duration
        complete_latencies = [elem['latency'] for elem in data['task_metrics']]
        result_dict['avg_completed_latency'] = np.mean(complete_latencies)
        job_runtimes = [elem['job_runtime'] for elem in data['task_metrics']]
        result_dict['avg_job_runtime'] = np.mean(job_runtimes)
        if data['load_round_total'] < 0:
            result_dict['load'] = 1 # No load for such workloads
        else:
            result_dict['load'] = data['load_round_total'] / event_duration # Load in QPS

    def _add_latency_metrics_to_dict(self, data, result_dict):
        """ Computes latency metrics. """
        # Compute e2e latencies  --------------------------------------------------------
        last_enqueue_time = data['round_end_time'] - self.slo_latency
        completed_latencies_to_consider = [elem['latency'] for elem in data['task_metrics']
                                           if elem['enqueue_time'] < last_enqueue_time]
        incomplete_latencies_to_consider = [1e6 for elem in data['incomplete_task_metrics']
                                            if elem['enqueue_time'] < last_enqueue_time]
        e2e_latencies = completed_latencies_to_consider + incomplete_latencies_to_consider
        # Compute reward and sigma ------------------------------------------------------
        reward, sigma = latency_metrics_from_e2e_latencies(e2e_latencies, self.slo_latency)
        result_dict['reward'] = reward
        if sigma is None:
            result_dict['sigma'] = None
        else:
            result_dict['sigma'] = min(0.5, LATENCY_SIGMA_MULTIPLIER * sigma)

    def _add_throughput_metrics_to_dict(self, data, result_dict):
        """ Computes throughput metrics. """
        reward = result_dict['throughput']
        if self.max_throughput > 0:
            max_throughput = self.max_throughput
        else:
            # Estimate this using the processing times
            job_runtimes = [elem['job_runtime'] for elem in data['task_metrics']]
            est_mean_runtime = np.mean(job_runtimes)
            alloc = result_dict['alloc']
            max_throughput = MAX_THROUGHPUT_COEFF * alloc / est_mean_runtime
        # Make sure its not larger than the observed throughput
        if max_throughput < reward:
            logger.info('max_throughput=%0.4f, but observed=%0.4f', max_throughput, reward)
            max_throughput = 2 * reward
        # compute sigma ---------------------------------------
        sigma = 0.5 * max_throughput
        result_dict['reward'] = reward
        result_dict['sigma'] = sigma
        result_dict['load'] = 1 # No load for throughput workloads

    def get_data(self, log_file):
        """
        This method returns a dict with data by parsing the output json file.
        """
        with open(log_file) as file_handle:
            data = json.load(file_handle)
            file_handle.close()
        ret = {}
        # Check if the loads are consistent throughout -------------------------------------
        if ('alloc_round_end' in data) and data['alloc'] != data['alloc_round_end']:
            logger.info('Returning None since allocation changed during this round.')
            return None
        debug_str = ''
        # Add some initial data -------------------------------------------------------------
        for key, value in data.items():
            if key == 'alloc':
                ret['alloc'] = value
            elif key == 'round_start_time':
                ret['event_start_time'] = value
            elif key == 'round_end_time':
                ret['event_end_time'] = value
            elif key in ['task_metrics', 'load_round_total_normalized', 'incomplete_task_metrics']:
                pass
            else:
                debug_str += '%s: %s, '%(str(key), str(value))
        self._add_common_metrics(data, ret)
        debug_str = 'throughput: %0.3f, avg-comp-lat: %0.3f, avg-job-runtime: %0.3f, %s'%(
            ret['throughput'], ret['avg_completed_latency'], ret['avg_job_runtime'], debug_str)
        ret['debug'] = debug_str
        # Compute reward and sigma ----------------------------------------------------------
        if self.slo_type == 'latency':
            self._add_latency_metrics_to_dict(data, ret)
            if ret['reward'] is None:
                logger.info('Reward computation returned None. Likely because there were 0 events')
                return None
        elif self.slo_type == 'throughput':
            self._add_throughput_metrics_to_dict(data, ret)
        else:
            raise NotImplementedError('Not impemented slo_type=%s yet!')
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
        parser.add_argument('--slo-type', '-slotype', type=str,
                            help='Type of SLO: latency, througput, or deadline')
        parser.add_argument('--slo-latency', '-slolat', type=float, default=-1.0,
                            help='If latency SLO, specify the latency in seconds.')
        parser.add_argument('--max-throughput', '-maxtp', type=int, default=-1,
                            help='If throughput, the maximum throughput.')
        # N.B: @Romil, we don't need to specify the performance goal here (such as 0.95 for latency
        # or 1000 QPS for throughput).
        pass

