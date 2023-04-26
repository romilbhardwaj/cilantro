"""
    Setup for module cilantro.
    -- romilbhardwaj
    -- kirthevasank
"""

from setuptools import setup

setup(name='cilantro',
      version='0.1',
      description='Cilantro Kubernetes Driver',
      author='romilb,kirthevasank',
      packages=['cilantro', 'cilantro_clients'],
      zip_safe=False)
