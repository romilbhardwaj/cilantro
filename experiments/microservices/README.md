# Cilantro Microservices Experiment

This experiment borrows the hotel-reservation benchmark from [Deathstarbench](https://github.com/delimitrou/DeathStarBench) to demonstrate
the application of cilantro to a microservices environment. Cilantro can optimize the 
end-to-end performance of the application by individually optimizing the resource 
allocation of each microservice. See Section 6.2 in the paper for more details.

## Pre-requisites

* Kubectl - [Linux](https://kubernetes.io/docs/tasks/tools/install-kubectl-linux/) or [MacOS](https://kubernetes.io/docs/tasks/tools/install-kubectl-macos/).
* A kubernetes cluster. 
  * For matching the results in our paper, we recommend using [eksctl](https://github.com/weaveworks/eksctl/blob/main/README.md#for-unix). It uses Amazon EKS to provision a Kubernetes cluster on AWS.
    * **Note** - this requires an AWS account, and we estimate an expense of ~$300 ($0.45 per m5.2xlarge instance hour x 20 instances x 8 hours x 4 benchmarks) for running the microservices experiments.
  * You can also use your own local Kubernetes cluster. Please make sure it has 200 vCPUs and the kubeconfig file is located at `$HOME/.kube/config`.
  * If you do not have access to cloud or a kubernetes cluster, you can run a Kubernetes cluster on your local machine with [kind](https://kind.sigs.k8s.io/docs/user/quick-start/#installation). This is useful for verifying functionality and debugging.
    * **Note** - We recommend a machine with at least 16 CPU cores and 32 GB RAM for running the experiments on `kind`. 
    * **Note** - Because kind is not representative of a real cluster, produced results will not be accurate or representative of Cilantro's performance.

## Provisioning your Kubernetes cluster for the experiment

If you are using a pre-existing kubernetes cluster, you can skip this section. If you want to provision a cluster on the cloud (AWS EKS) or on your local machine (`kind`), follow the instructions below.

### AWS EKS cluster (recommended)
1. Install [eksctl](https://github.com/weaveworks/eksctl/blob/main/README.md#for-unix). In summary, run:
   ```sh
   curl --silent --location "https://github.com/weaveworks/eksctl/releases/latest/download/eksctl_$(uname -s)_amd64.tar.gz" | tar xz -C /tmp
   sudo mv /tmp/eksctl /usr/local/bin
   eksctl version  # To test
   ```
2. Run the following command to create a cluster with 20 m5.2xlarge instances and pre-pull the container images for Cilantro.
   ```sh
   ./starters/create_eks_cluster.sh eks.kubeconfig
   ```
   This will take ~10 minutes.
3. Verify the cluster has been created by running `kubectl get nodes`. You should see 20 nodes in the cluster.

### Local Kubernetes cluster (`kind`)
If you do not have access cloud or a kubernetes cluster, you can use kind to provision a cluster on your local machine. This is useful for verifying functionality and debugging. However, we do not recommend running the experiments on kind as it is not representative of a real cluster.

0. Make sure your docker daemon is running and has [resource limits](https://docs.docker.com/desktop/settings/mac/#resources) set correctly.
1. Install [kind](https://kind.sigs.k8s.io/docs/user/quick-start/#installation). In summary, run:
   ```sh
   curl -Lo ./kind https://kind.sigs.k8s.io/dl/v0.11.1/kind-linux-amd64
   chmod +x ./kind
   sudo mv ./kind /usr/local/bin/kind
   kind version  # To test
   ```
2. Run the following command to create a cluster
   ```sh
   ./starters/create_kind_cluster.sh
   ```
3. Verify the cluster has been created by running `kubectl get nodes`. You should see 1 node in the cluster.
 

## Running Microservices and launching Cilantro
1. Create hotel-reservation microservices (Cilantro won't do this for you automatically like in cluster sharing experiment):
   ```
   ./starters/launch_hotelres.sh 
   ```
   You can check the status of pods by running `kubectl get pods`. Wait till all pods are running:
   ```sh
   (base) ➜  ~ watch kubectl get pods
   NAME                                            READY   STATUS    RESTARTS      AGE
   root--consul-c9fc844d4-vt9kb                    1/1     Running   0             75s
   root--frontend-6f9d6c86b7-jhckd                 1/1     Running   0             75s
   root--geo-664b55bc56-6wm99                      1/1     Running   1 (24s ago)   75s
   root--jaeger-85669ffcdd-gz6dt                   1/1     Running   0             75s
   root--memcached-profile-5bdf886f66-lrhhb        1/1     Running   0             75s
   root--memcached-rate-5b6f6b57c6-82xr9           1/1     Running   0             74s
   root--memcached-reserve-6d6c7bb678-xtd8q        1/1     Running   0             73s
   root--mongodb-geo-bfb8ff665-4t5qh               1/1     Running   0             75s
   root--mongodb-profile-57f4959847-cdmkt          1/1     Running   0             74s
   root--mongodb-rate-5b6d9898c6-f5689             1/1     Running   0             74s
   root--mongodb-recommendation-7dbc6cd97c-cthxh   1/1     Running   0             74s
   root--mongodb-reservation-57ffbfd557-c9vcw      1/1     Running   0             73s
   root--mongodb-user-6bc8cc7c85-9mq9x             1/1     Running   0             72s
   root--profile-7945dbbdc4-gz2kj                  1/1     Running   0             74s
   root--rate-65df547f8d-2kn46                     1/1     Running   0             74s
   root--recommendation-845d49d566-vf6ss           1/1     Running   0             73s
   root--reservation-69bff9d4d9-php86              1/1     Running   0             73s
   root--search-786bcbf57b-mrhw2                   1/1     Running   0             72s
   root--user-7c4b899595-zng5g                     1/1     Running   0             72s
   ```
2. After the microservices are running, launch cilantro scheduler and hr-client (which starts sending queries to the microservices). You can set the qps scale (load) in `./starters/hotel-res/cilantro-hr-client.yaml` by changing the --wrk-qps arg)
   ```sh
   # Set the policy to run - ucbopt, propfair, msile, msevoopt
   POLICY=propfair
   # If running on EKS
   ./starters/launch_cilantro_driver.sh ~/.kube/config $POLICY
   # If running on kind (local cluster)
   ./starters/launch_cilantro_driver_kind.sh ~/.kube/config $POLICY
   ```
3. [Optional] To view cilantro's logs, run:
   ```sh
   ./starters/view_logs.sh
   ```
4. [Optional] To check the status of the cluster, you can access the dashboard in your browser after running `kubectl proxy`. Press skip when asked for credentials.
   ```sh
   ./starters/kubeproxy.sh
   # In browser, open: http://localhost:8001/api/v1/namespaces/kubernetes-dashboard/services/https:kubernetes-dashboard:/proxy/#/persistentvolumeclaim?namespace=_all
   ```
5. We recommend running each baseline for at least 8 hours. At the end of the experiment, to fetch the results, run:
   ```sh
   ./starters/fetch_results.sh
   ```
   This will run a script that periodically fetches logs from Cilantro and saves them in `workdirs_eks` directory. You can stop it once it has fetched results once. 
6. After the experiment has run and results have been fetched, you can clean up the cluster (i.e., delete all pods and services) by running:
   ```sh
   ./starters/clean_cluster.sh
   ```
7. Repeat steps 1-6 for each baseline you want to run by changing `POLICY` variable. Here's each policy and their description.
   * `propfair` - Proportional Fairness, always equally allocates the resources among microservices.
   * `ucbopt` - Cilantro's upper confidence bound policy
   * `msile` - Epsilon-greedy with e=1/3.
   * `msevoopt` - Evolutionary algorithm
7After the experiment is done, clean up your cluster by running:
   ```
   # If running EKS
    ./starters/delete_eks_cluster.sh
   # If running kind cluster
    ./starters/delete_kind_cluster.sh
   ```
8. To plot the results, run:
   ```sh
    python plot_results.py
    ```
   You should see a plot like this:

   <img src="https://raw.githubusercontent.com/romilbhardwaj/cilantro/main/experiments/microservices/workdirs_eks/ms_exp.png"  width="300">

