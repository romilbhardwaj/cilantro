# Cilantro EKS Guide

To run experiments, we use [Amazon EKS](https://aws.amazon.com/eks), a managed kubneretes service from Amazon.
It takes care of spinning up VMs, installing k8s on them and configuring them to work in the same cluster.
Like we use Kind on our local machines to setup a cluster, we will use EKS to setup a real cluster in the cloud
(i.e., kind:local::EKS:cloud).

## Setting up your docker build to push to ECR
Till now, the docker images you build using `docker_build_kind.sh` have been building and storing the images locally.

To run on the cloud, we also need to push our images to the cloud so our remote kubernetes cluster can fetch those images.
Amazon ECR (Elastic container registry) is a remote repository for built docker images. Now onwards, we will build images locally, but then push them to ECR so they can be fetched from anywhere.
The new script we will use is `docker_build_ecr.sh`

Before using `docker_build_ecr.sh` you will need to setup your docker auth to work with ECR. To do so, retrieve an authentication token and authenticate your Docker client to your registry. Run the following command:

```
aws ecr-public get-login-password --region us-east-1 | docker login --username AWS --password-stdin public.ecr.aws/a9w6z7w5
```

Notes:
* You might need to setup aws-cli before this. See instructions on [installing](https://docs.aws.amazon.com/cli/latest/userguide/install-cliv2-linux.html) and [configuring](https://docs.aws.amazon.com/cli/latest/userguide/cli-configure-quickstart.html).
* If you're pushing to the official Cilantro image repo, you might need permissions on the ECR repository. Contact romil.bhardwaj@berkeley.edu to get access. 
* `a9w6z7w5` is our ECR alias for now. romilb has applied to get the name `cilantro`, but it is subject to AWS approval. 
* You MUST use `--region us-east-1` irrespective of the region your repo is actually in.
* Since you are pushing to remote repository, make sure the image works! Think of this push as a push to production - it will also affect anyone else running experiments on EKS. 

Test your setup by going to root and running `./docker_build_ecr.sh.` The push should be successful.


## Installing eksctl
Like kubectl, EKS has a command line tool to manage your EKS cluster called `eksctl`. We will primarily use this to setup our cluster.
Check the guide [here](https://docs.aws.amazon.com/eks/latest/userguide/eksctl.html). In summary:
```
curl --silent --location "https://github.com/weaveworks/eksctl/releases/latest/download/eksctl_$(uname -s)_amd64.tar.gz" | tar xz -C /tmp
sudo mv /tmp/eksctl /usr/local/bin
eksctl version  # To test
```

## Creating a cluster
eksctl defines the the cluster specification as a YAML file, which can be executed with `eksctl create cluster -f <yaml>`.
To setup our test cluster, run
```
eksctl create cluster -f eks-test-cluster.yaml
``` 
Running this takes ~10-20 min. You can track the status in the [cloudformation dashboard](https://us-west-2.console.aws.amazon.com/cloudformation/home?region=us-west-2#/stacks?filteringStatus=active&filteringText=&viewNested=true&hideStacks=false).  
 
Once setup, you should be able to run kubectl to interact with your cluster and create workloads.
For example, try getting the list of nodes:
```
kubectl get nodes

NAME                                          STATUS   ROLES    AGE     VERSION
ip-192-168-43-53.us-west-2.compute.internal   Ready    <none>   2m33s   v1.20.7-eks-135321
ip-192-168-65-73.us-west-2.compute.internal   Ready    <none>   2m30s   v1.20.7-eks-135321
```

As a test, try deploying the kubernetes dashboard.
```
cd ../k8s/
kubectl apply -f dashboard.yaml
kubectl proxy
# Now open http://localhost:8001/api/v1/namespaces/kubernetes-dashboard/services/https:kubernetes-dashboard:/proxy/#/workloads?namespace=default.
``` 

## Running cilantro
After creating the cluster, you should be able to use the usual kubectl commands.

🚨🚨🚨 Note that you **MUST** change the image in the yamls to `image: public.ecr.aws/cilantro/cilantro:latest` 🚨🚨🚨 

For an example, see `k8s/cilantro_eks.yaml`

## Clean up
Don't leave your cluster running for long - it's expensive! To remove a cluster, run:
```
eksctl delete cluster -f eks-test-cluster.yaml
```