# Cilantro Cluster Sharing Experiment

This experiment evaluates Cilantro’s multi-tenant policies (Section 4.1.2 in the paper) 
on a 1000 CPU cluster shared by 20 users. Please refer to Section 6.1 in the paper for more details.

## Pre-requisites

* Kubectl - [Linux](https://kubernetes.io/docs/tasks/tools/install-kubectl-linux/) or [MacOS](https://kubernetes.io/docs/tasks/tools/install-kubectl-macos/).
* A kubernetes cluster. 
  * For matching the results in our paper, we recommend using [eksctl](https://github.com/weaveworks/eksctl/blob/main/README.md#for-unix). It uses Amazon EKS to provision a Kubernetes cluster on AWS.
    * **💸 NOTE 💸** - this requires an AWS account, and we estimate an expense of at least ~$441 ($0.22 per m5.xlarge instance hour x 251 instances x 8 hours) per policy. For running all policies included in the paper, **we estimate a total cost of $6615**.
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
2. Run the following command to create a cluster with 251 m5.xlarge instances and pre-pull the container images for Cilantro.
   ```sh
   ./starters/create_eks_cluster.sh eks.kubeconfig
   ```
   This will take ~10 minutes.
3. Verify the cluster has been created by running `kubectl get nodes`. You should see 251 nodes in the cluster.

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
 

## Running the cluster sharing experiment and launching Cilantro
1. Launch the cluster sharing experiment by running the commands below, launch cilantro scheduler and hr-client (which starts sending queries to the microservices). You can set the qps scale (load) in `./starters/hotel-res/cilantro-hr-client.yaml` by changing the --wrk-qps arg)
   ```sh
   # Set the policy to run - propfair, mmf, mmflearn, utilwelforacle, utilwelflearn, evoutil, egalwelforacle, egalwelflearn, evoegal, greedyegal, minerva, ernest, quasar, parties, multincadddec
   POLICY=propfair
   # If running on EKS
   ./starters/launch_cilantro_driver.sh ~/.kube/config $POLICY
   # If running on kind (local cluster)
   ./starters/launch_cilantro_driver_kind.sh ~/.kube/config $POLICY
   ```
   You can check the status of pods by running `kubectl get pods`. You may first see `cilantroscheduler` starting, and after a bit cilantro scheduler will create the workload pods. Wait till all pods are running:
   ```sh
    (base) ➜  ~ k get pods
    NAME                                  READY   STATUS              RESTARTS   AGE
    cilantroscheduler-v2nsc               1/1     Running             0          46s
    root--db0j1-5c54cd6cd4-fgp4t          0/1     ContainerCreating   0          39s
    root--db0j1-client-7878998b77-68p5c   0/2     ContainerCreating   0          39s
    root--db0j1-head-6bb647f5d-n8kw9      0/1     ContainerCreating   0          39s
    root--db0j2-5d8dddb676-58qvf          0/1     ContainerCreating   0          39s
    root--db0j2-client-76589779d8-fm446   0/2     ContainerCreating   0          39s
    root--db0j2-head-66c9b97c9b-9wtgj     0/1     ContainerCreating   0          39s
    root--db0j3-7bbbcc76c6-6gqcg          0/1     ContainerCreating   0          39s
    root--db0j3-client-869cb95c87-v22p4   0/2     ContainerCreating   0          39s
    root--db0j3-head-757fd5c5d6-6t5gf     0/1     ContainerCreating   0          39s
    root--db1j1-85499bb98d-2ml22          0/1     ContainerCreating   0          38s
    root--db1j1-client-7fc99ccb44-9kbmr   0/2     ContainerCreating   0          38s
    root--db1j1-head-5f6f8dfd68-cf65d     0/1     ContainerCreating   0          39s
    root--db1j2-585c698b68-l2ptm          0/1     ContainerCreating   0          38s
    root--db1j2-client-69f79fbf97-8g249   0/2     ContainerCreating   0          38s
    root--db1j2-head-8c76fbf86-9g29n      0/1     ContainerCreating   0          38s
    root--db1j3-56cddb95d8-dpvlk          0/1     ContainerCreating   0          37s
    root--db1j3-client-f9dffb8bb-bv2ph    0/2     ContainerCreating   0          37s
    root--db1j3-head-d7fcbd8b4-k62ng      0/1     ContainerCreating   0          37s
    root--db1j4-5bcff66d7b-j8x98          0/1     ContainerCreating   0          36s
    root--db1j4-client-6c5fc88c67-q4wqm   0/2     ContainerCreating   0          36s
    root--db1j4-head-5d6d49bc6b-j28hn     0/1     ContainerCreating   0          36s
    root--db1j5-64688cc856-88mlq          0/1     ContainerCreating   0          36s
    root--db1j5-client-5cc9f6446d-jc5zp   0/2     ContainerCreating   0          36s
    root--db1j5-head-848fd6fc6d-vx87m     0/1     ContainerCreating   0          36s
    root--db1j6-69d8c6884d-rjpbx          0/1     ContainerCreating   0          35s
    root--db1j6-client-7bf9d4f8b5-xw4r8   0/2     ContainerCreating   0          35s
    root--db1j6-head-77bc9dccc8-j9bz8     0/1     ContainerCreating   0          36s
    root--db1j7-6c748fd5bd-jms2x          0/1     ContainerCreating   0          35s
    root--db1j7-client-7c5cd7dd58-nwk6h   0/2     ContainerCreating   0          35s
    root--db1j7-head-6595dcd56-nj4w4      0/1     ContainerCreating   0          35s
    root--mltj1-7c7766b76-gsvpn           0/1     ContainerCreating   0          34s
    root--mltj1-client-8d6c44c6d-cggqz    0/2     ContainerCreating   0          34s
    root--mltj1-head-699df77b84-pzgfh     0/1     ContainerCreating   0          34s
    root--mltj2-85489bd46d-tb4m5          0/1     ContainerCreating   0          34s
    root--mltj2-client-5d66b4fdbf-2g5z6   0/2     ContainerCreating   0          33s
    root--mltj2-head-9656988cd-rrm7q      0/1     ContainerCreating   0          34s
    root--mltj3-7dc7fbbbd-blgxm           0/1     ContainerCreating   0          33s
    root--mltj3-client-597b56cc9d-vwv95   0/2     ContainerCreating   0          33s
    root--mltj3-head-5b644fb7b-lwl4t      0/1     ContainerCreating   0          33s
    root--mltj4-9dc55685b-8mnwd           0/1     ContainerCreating   0          32s
    root--mltj4-client-58bdb769cf-wbbh5   0/2     ContainerCreating   0          32s
    root--mltj4-head-d45f7d578-v852k      0/1     ContainerCreating   0          33s
    root--mltj5-55fc784b4d-85wp6          0/1     ContainerCreating   0          32s
    root--mltj5-client-764b97667b-pwl2g   0/2     ContainerCreating   0          32s
    root--mltj5-head-5876f84cbd-mpmmv     0/1     ContainerCreating   0          32s
    root--mltj6-6bf4fbc544-ncbvc          0/1     ContainerCreating   0          31s
    root--mltj6-client-749cc984f-46tg4    0/2     ContainerCreating   0          31s
    root--mltj6-head-7d7447fcd6-lvgnk     0/1     ContainerCreating   0          31s
    root--mltj7-cdd6d4696-jrkwx           0/1     ContainerCreating   0          31s
    root--mltj7-client-76f9996fcd-hlcjv   0/2     ContainerCreating   0          30s
    root--mltj7-head-6f4477498d-jc5b2     0/1     ContainerCreating   0          31s
    root--prsj1-786f55fc64-2wsbf          0/1     ContainerCreating   0          30s
    root--prsj1-client-cb87f9f45-2k7vb    0/2     ContainerCreating   0          30s
    root--prsj1-head-67bb6dc47b-l85ck     0/1     ContainerCreating   0          30s
    root--prsj2-7967b7554d-vsrgs          0/1     ContainerCreating   0          29s
    root--prsj2-client-5f66f4ccfb-924dj   0/2     ContainerCreating   0          29s
    root--prsj2-head-78f646f96b-r4hjb     0/1     ContainerCreating   0          30s
    root--prsj3-5ffdb75776-grf7n          0/1     ContainerCreating   0          29s
    root--prsj3-client-7fbc57cd79-lshnh   0/2     ContainerCreating   0          29s
    root--prsj3-head-766985fd76-84bpz     0/1     ContainerCreating   0          29s   
   ```
   * **NOTE** - If you're running on kind, your local cluster may crash due to lack of resources. The error may look like `Unable to connect to the server: net/http: TLS handshake timeout`. If that happens, try increasing the CPUs and memory allocated to your Docker runtime by opening the Docker dashboard and going to Preferences -> Resources -> Advanced.
2. After all pods are running, the experiment has started. To stream cilantro logs, run:
   ```sh
   ./starters/view_logs.sh
   # Ctrl-C to stop streaming logs
   ```
   You should see an output like this periodically, which shows the resource allocation chosen by Cilantro for the round:
   ```
   propfair (193.0):: res-loss=0.737, fair_viol:(sum=0.526, max=0.667, mean=0.667), util_welf=0.191, egal_welf=0.063, avg_rew=14.453,  cost=20.000
    - root--db0j1: util=0.275, rew=0.248 alloc=1.000, load=20.188, ud=0.312, time=219.541, root--db0j2: util=0.366, rew=0.337 alloc=1.000, load=20.170, ud=0.312, time=218.445, root--db0j3: util=0.543, rew=0.525 alloc=1.000, load=20.192, ud=0.320, time=220.583, root--db1j1: util=0.197, rew=0.178 alloc=1.000, load=42.779, ud=2.146, time=207.205, root--db1j2: util=0.188, rew=0.169 alloc=1.000, load=42.461, ud=2.146, time=218.293, root--db1j3: util=0.188, rew=0.179 alloc=1.000, load=42.825, ud=3.463, time=206.910, root--db1j4: util=0.157, rew=0.150 alloc=1.000, load=42.800, ud=3.463, time=205.373, root--db1j5: util=0.158, rew=0.150 alloc=1.000, load=42.542, ud=3.463, time=216.728, root--db1j6: util=0.138, rew=0.137 alloc=1.000, load=42.652, ud=3.868, time=210.586, root--db1j7: util=0.169, rew=0.167 alloc=1.000, load=42.576, ud=3.868, time=215.426, root--mltj1: util=0.094, rew=37.798 alloc=1.000, load=1.000, ud=99.887, time=217.949, root--mltj2: util=0.094, rew=37.648 alloc=1.000, load=1.000, ud=99.887, time=205.986, root--mltj3: util=0.081, rew=36.635 alloc=1.000, load=1.000, ud=119.934, time=209.857, root--mltj4: util=0.100, rew=44.878 alloc=1.000, load=1.000, ud=119.934, time=220.620, root--mltj5: util=0.087, rew=43.543 alloc=1.000, load=1.000, ud=150.000, time=221.484, root--mltj6: util=0.081, rew=40.393 alloc=1.000, load=1.000, ud=150.000, time=217.390, root--mltj7: util=0.090, rew=45.238 alloc=1.000, load=1.000, ud=150.000, time=216.144, root--prsj1: util=0.234, rew=0.211 alloc=1.000, load=40.801, ud=0.316, time=207.715, root--prsj2: util=0.282, rew=0.254 alloc=1.000, load=40.698, ud=0.316, time=207.283, root--prsj3: util=0.244, rew=0.232 alloc=1.000, load=40.877, ud=0.570, time=202.461
   ```
3. [Optional] To check the status of the cluster, you can access the dashboard in your browser after running `kubectl proxy`. Press skip when asked for credentials.
   ```sh
   ./starters/kubeproxy.sh
   # In browser, open: http://localhost:8001/api/v1/namespaces/kubernetes-dashboard/services/https:kubernetes-dashboard:/proxy/#/persistentvolumeclaim?namespace=_all
   ```
4. We recommend running each baseline for at least 8 hours. At the end of the experiment, to fetch the results, run:
   ```sh
   ./starters/fetch_results.sh
   ```
   This will run a script that periodically fetches logs from Cilantro and saves them in `workdirs_eks` directory. You can stop it after it has fetched results once. 
5. Repeat steps 2-4 for each baseline you want to run by changing `POLICY` variable. Each policy should be run for at least 8 hours. Here's each policy and their description.
   * `propfair` - Proportional Fairness (`Resource-fair` in the paper), always equally allocates the resources among microservices.
   * `mmf` - No-Justified Complaints (NJC) resource allocation with oracular information (`Oracle-NJC` in the paper).
   * `mmflearn` - Cilantro's learned NJC resource allocation policy (`Cilantro-NJC` in the paper).
   * `evoutil` - Evolutionary algorithm for maximizing social welfare (`EvoAlg-SW` in the paper).
   * `utilwelflearn` - Cilantro's learned policy for maximizing social welfare (`Cilantro-SW` in the paper).
   * `utilwelforacle` - Policy for maximizing social welfare with oracular information (`Oracle-SW` in the paper).
   * `greedyegal` - Greedy policy for maximizing egalitarian welfare (`Greedy-EW` in the paper).
   * `evoegal` - Evolutionary algorithm for maximizing egalitarian welfare (`EvoAlg-EW` in the paper).
   * `egalwelflearn` - Cilantro's learned policy for maximizing egalitarian welfare (`Cilantro-EW` in the paper).
   * `egalwelforacle` - Policy for maximizing egalitarian welfare with oracular information (`Oracle-EW` in the paper).
   * `multincadddec` - Multiplicative Increase Additive Decrease (`MIAD` in the paper).
   * `minerva` - Our adaptation of the policy presented in [Minerva](https://dl.acm.org/doi/10.1145/3341302.3342077) (`Minerva` in the paper).
   * `ernest` - Our adaptation of the policy presented in [Ernest](https://dl.acm.org/doi/10.5555/2930611.2930635) (`Ernest` in the paper).
   * `quasar` - Our adaptation of the policy presented in [Quasar](https://dl.acm.org/doi/10.1145/2541940.2541941) (`Quasar` in the paper).
   * `parties` - Our adaptation of the policy presented in [PARTIES](https://dl.acm.org/doi/10.1145/3297858.3304005) (`PARTIES` in the paper).
   
6. After the experiment is done, clean up your cluster by running:
   ```
   # If running EKS
    ./starters/delete_eks_cluster.sh
   # If running kind cluster
    ./starters/delete_kind_cluster.sh
   ```
7. To plot the results fetched by the `fetch_results.sh` script, run:
   ```sh
    python plot_results.py
    ```

