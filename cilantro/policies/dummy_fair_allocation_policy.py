from typing import List, Dict

from cilantro.policies.base_policy import BasePolicy


class DummyFairAllocationPolicy(BasePolicy):
    def __init__(self,
                 total_resources: int,
                 jobs: List[str]):
        self.resource_quantity = total_resources
        self.jobs = jobs
        super(DummyFairAllocationPolicy, self).__init__(total_resources, jobs)

    def get_resource_allocation(self) -> Dict[str, float]:
        fair_share = int(self.total_resources/len(self.jobs)) if len(self.jobs) > 0 else 0
        return {j: fair_share for j in self.jobs}