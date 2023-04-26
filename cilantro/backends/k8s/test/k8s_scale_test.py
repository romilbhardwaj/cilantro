import argparse

from cilantro.backends.k8s.kubernetes_manager import KubernetesManager
import logging

# Run this script only after launching the nginx deployment. Double check the name!

def parseargs():
    parser = argparse.ArgumentParser(description='Script to scale deployments up and down.')
    parser.add_argument('--replicas', '-r', type=int, default=2, help='Target replicas')
    parser.add_argument('--name', '-n', type=str, default='nginx', help='Deployment name to scale')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    args = parseargs()
    logging.info(args)
    k = KubernetesManager()
    k.scale_deployment(args.name, args.replicas)
