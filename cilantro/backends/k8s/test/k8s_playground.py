from cilantro.backends.k8s.kubernetes_manager import KubernetesManager, load_k8s_config
from kubernetes import client
import logging

if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    load_k8s_config()
    core = client.CoreV1Api()
    apps = client.AppsV1Api()

