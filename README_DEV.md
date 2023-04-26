# Developer guide for Cilantro

This document is a collection of notes for developers interested in developing and extending cilantro.

## Non-kubernetes installation
This installation allows you to run some components of cilantro locally. Install and run demo by following these commands.
```
$ pip install numpy, statsmodels, pylint    # dependencies
$ git clone https://github.com/romilbhardwaj/cilantro.git    # download repo
$ cd cilantro
$ pip install -e .   # install cilantro
$ cd demos/demo1
$ python demo1.py
```

This is the full installation of Cilantro to run E2E. We first install kubernetes on your local machine using kind, then run cilantro and finally run a test workload to make sure everything is working.  

### Setting up Kind and Kubectl
[Kind](https://kind.sigs.k8s.io/) is a tool to emulate a kubernetes cluster in your local machine by running kubernetes nodes as docker containers.
1. Install kind with on linux with (See [guide](https://kind.sigs.k8s.io/docs/user/quick-start/) for Mac):
```
curl -Lo ./kind https://kind.sigs.k8s.io/dl/v0.11.1/kind-linux-amd64
chmod +x ./kind
mv ./kind /some-dir-in-your-PATH/kind
```
2. Install `kubectl`. Kubectl is the command line utility to control and administer a kubernetes cluster.
```
sudo apt-get update
sudo apt-get install -y apt-transport-https ca-certificates curl
sudo curl -fsSLo /usr/share/keyrings/kubernetes-archive-keyring.gpg https://packages.cloud.google.com/apt/doc/apt-key.gpg
echo "deb [signed-by=/usr/share/keyrings/kubernetes-archive-keyring.gpg] https://apt.kubernetes.io/ kubernetes-xenial main" | sudo tee /etc/apt/sources.list.d/kubernetes.list
sudo apt-get update
sudo apt-get install -y kubectl
```
3. *Highly recommended* - setup [autocomplete for kubectl](https://kubernetes.io/docs/tasks/tools/included/optional-kubectl-configs-bash-linux/). This allows you to autocomplete kubectl commands by pressing tab.

## Running Cilantro outside of Kubernetes cluster

### Running Cilantro Scheduler
1. Run `./docker_build_.sh` in cilantro root to ensure that docker has the latest cilantro image compiled from source. This command takes the current snapshot of code and compiles it into a docker image and loads it into your kind cluster.
1. Go to `cd cilantro/k8s/`
2. The first time you setup the cluster, you must create permissions for cilantro to view and modify kubernetes state. Do so by running `kubectl apply -f auth_default_user.yaml`. This needs to be run only once per kubernetes cluster.
3. We can now launch cilantroscheduler. Run `kubectl apply -f cilantro.yaml`.

### Checking/debugging cilantro status
1. To get the list of running pods in kubernetes, run `kubectl get pods`.
2. To check the status of the Cilantro Scheduler pod (and identify any failure causes), run `kubectl describe pod cilantroscheduler<tab>`. Press tab to autocomplete the pod name.
3. To check the output logs of cilantro, run `kubectl logs -f cil<tab>` or check the kubernetes dashboard (see below). You should see something like `08-30 00:37:49 | DEBUG | cilantro.scheduler.cilantroscheduler || Waiting for event`

### Workload structure in cilantro
Once cilantro is running, it is looking for workloads to allocate resources to. Workloads in cilantro system have three main components:

* Workload servers, which are scaled by cilantro
* Workload client, which generates the queries (load) for the workload servers
* Cilantro client, which reads the workload client and/or cluster state to compute utility metrics and report them to cilantro scheduler.

### Running a test workload (proportional-to-allocation)
For the purposes of testing, we have a dummy workload which runs nginx servers, and a corresponding cilantro client which publishes the utility/load metrics as a function of the resource allocation to this deployment. You can find the functions in `cilantro_clients/drivers/k8s_to_grpc_driver.py`.

Since the workload is dummy, there are no workload clients to generate load - the metrics reported by cilantro client are a direct function of the allocation.

In these instructions, we first we launch the workload, then we launch the cilantro client which will report the workload's metrics.

1. Navigate to `cd k8s/workloads/test`
2. Launch the workload by running `kubectl create -f nginx-deployment.yaml`. Also have a look at this file - it starts with a resource allocation of 1 and defines the app_weight, slo threshold and unit demand as kubernetes labels.
3. Now that the workload is running, you'll notice the `cilantroscheduler` logs will show that it detected a new application and it will change it's resource allocation from 1 to <max_resources> since it is the only running app.
4. Now we must start reporting metrics to the cilantroscheduler by creating a cilantroclient for this workload. Do so by running `kubectl create -f nginx-cilantroclient-deployment.yaml`. Soon it will start publishing metrics, which will again reflect in cilantro logs.

### [Recommended] Setting up the Dashboard
![Kubernetes Dashboard](https://d33wubrfki0l68.cloudfront.net/349824f68836152722dab89465835e604719caea/6e0b7/images/docs/ui-dashboard.png)

The kubernetes dashboard allows you to monitor the cluster status and running workloads without having to bother with `kubectl`. To set up the dashboard:

1. We need to create the dashboard deployment. Do so by running `kubectl apply -f dashboard.yaml` in `cilantro/k8s`
2. In a new terminal window, run `kubectl proxy`
3. Open your browser and goto `http://localhost:8001/api/v1/namespaces/kubernetes-dashboard/services/https:kubernetes-dashboard:/proxy/#/workloads?namespace=default`
4. If prompted for login, press the skip button.
5. Spend some time exploring the dashboard and learning about it. It's a super useful tool - you can view the resource allocations (replica counts) and pod logs.

### Clean up
* To stop a running workload, run `kubectl delete -f <yaml_path>` or `kubectl delete deployment.apps <name>`.
* To stop all workloads and the cilantro scheduler, run `kubectl delete deployments.apps --all`
* To kill the cluster, run `kind delete cluster`
* Its useful to create a alias for quick clean up: ```alias kubeclean "kubectl delete jobs,daemonsets,replicasets,services,deployments,pods,rc,statefulset --all --grace-period=0 --force"```. Then you can simply run `kubeclean` to cleanup your kubernetes cluster.
