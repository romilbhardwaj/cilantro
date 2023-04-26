from cilantro.backends.k8s.kubernetes_manager import KubernetesManager
import logging

if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    k = KubernetesManager(event_queue=None)
    k.list_pods()
    print(k.get_cluster_resources())
    nodes = k.coreapi.list_node()