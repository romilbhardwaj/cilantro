## A simple demo with a synthetic workload on a kind k8s cluster


### Running by computing oracular information from synthetic workload

First build the docker image.
```
$ cd ../../
$ ./docker_build_kind.sh
$ cd -
```
Then, to run the scheduler where the oracular data is computed directly from the synthetic workload,
execute the following command.
```
$ ./start_cilantro_driver.sh
```
You may change the policy by changing the `POLICY` variable in `cilantro_driver.py` (TODO: change
this to command line --kirthevasank).
To copy the saved results at any stage of the experiment, you should copy over the `results`
directory.
See  [`copy_data_from_pods_examples`](copy_data_from_pods_examples) file for a command you may use.
To stop the experiment, simply delete the cluster.
```
$ kubectl delte deployments.apps --all
```

### Running by loading oracular information from profiled data

To run this demo, we first need to profile the workloads and process the profiled data. There are
two types of workloads used in this environment with three applications (with two of them having the
same workload type). The two workload types are `dummy1` and `dummy2`. To profile the `dummy1`,
first ensure that the args in `config_profiling_driver.yaml` is set to `dummy1` and then run
```
$ ./start_profiling_driver.sh
```
When you profile a workload type, the script runs indefinitely, so you have to stop it at some
point. Usually, 3 rounds of allocations are enough for each allocation quantity we are profiling.
See the [`README.md`](../../README.md) of the maiin repo on how to view the logs to monitor the
current status of the profiling script.

After the job has been profiled for a while, before you delte the pods, make sure you copy the
profiled data.
See  [`copy_data_from_pods_examples`](copy_data_from_pods_examples) file for a command you may use.
You will need to replace `cilantroscheduler-xxx` with the correct pod
name, which can be obtained via the command `kubectl get pods`. Finally, delete the cluster.
```
$ kubectl delte deployments.apps --all
```

Once you do this for `dummy1`, repeat the same steps for `dummy2`.

After these steps, the directory `profiling_data_logs` in the current directory (in your local
machine) will contain the
profiled data. We now need to process the profiled data which can be done via the following
command. If you want to visualise the profiled data, this can be done by passing the `--to-plot 1` flag.
```
$ python ../../cilantro/profiling/process_profile_data.py --logs-dir profiling_data_logs
```
Once this is done, you should rebuild the docker file so that the profiled data is available when we
run the next time. This can be done via,
```
$ cd ../../
$ ./docker_build_kind.sh
$ cd -
```

Finally, you can run the scheduler by loading the profiled data as follows.
```
$ ./start_cilantro_from_profiling.sh
```
To copy the saved results at any stage of the experiment, you should copy over the `results`
directory.
See  [`copy_data_from_pods`](copy_data_from_pods) file for a command you may use.
To stop the experiment, simply delete the cluster.
```
$ kubectl delte deployments.apps --all
```

