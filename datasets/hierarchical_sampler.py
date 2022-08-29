import math

import torch
import torch.distributed as dist
from torch.utils.data.sampler import Sampler

from datasets.textfile_hierarchical_dataset import TextfileHierarchicalDataset


class HierarchicalPKSampler(Sampler):
    """Each batch loads P samples from K classes at a certain level of the taxonomy"""
    def __init__(self, P, K, data_source: TextfileHierarchicalDataset):
        super(HierarchicalPKSampler, self).__init__(data_source)

        self.data_source = data_source
        self.P = P
        self.K = K
        self.generator = torch.Generator()

    def _make_indices_batches(self, classes_to_sample, samples_dict, seed):
        self.generator.manual_seed(seed)
        subsets = []
        for c_id in classes_to_sample:
            # for each class, permute its elements and divide them in subsets of P samples of a single class
            class_samples_permuted = [samples_dict[c_id][i] for i in torch.randperm(
                len(samples_dict[c_id]), generator=self.generator)]
            j = 0
            while j < len(class_samples_permuted):
                if j + self.P > len(class_samples_permuted):
                    subsets.append(class_samples_permuted[j:])
                else:
                    subsets.append(class_samples_permuted[j:j + self.P])
                j += self.P

        # Permute subsets
        subsets = [subsets[i] for i in torch.randperm(len(subsets), generator=self.generator)]

        i = 0
        self.indices_list = []
        while i < len(subsets):
            if i + self.K > len(subsets):
                self.indices_list.append(sum(subsets[i:], []))
            else:
                self.indices_list.append(sum(subsets[i:i + self.K], []))
            i += self.K

    def new_sampling(self, seed, lvl):
        """Needs to be called each epoch to generate a different sampling."""
        classes_to_sample = self.data_source.labels_to_indices[lvl].keys()
        self._make_indices_batches(classes_to_sample, self.data_source.labels_to_indices[lvl], seed)

    def __iter__(self):
        if self.indices_list:
            return iter(self.indices_list)
        else:
            raise RuntimeError("sampler.new_sampling(seed, lvl) must be called")


class DistributedMixin:
    """Distributes batches of a sampler among the GPUs"""
    def __init__(self, *args, **kwargs):
        super(DistributedMixin, self).__init__(*args, **kwargs)

        if not dist.is_available():
            raise RuntimeError("Requires distributed package to be available")
        self.num_replicas = dist.get_world_size()
        self.rank = dist.get_rank()
        self.newsamplingerrormsg = "sampler.new_sampling(seed, lvl) must be called"

    def _assign_local_batches_to_rank(self):
        """Distributes samples across devices."""
        self.num_batches = int(math.ceil(len(self.indices_list) * 1. / self.num_replicas))
        # There can be empty batches on the GPUs of larger rank
        i = self.rank * self.num_batches
        self.local_batches = self.indices_list[i:i + self.num_batches]

    def __iter__(self):
        if self.indices_list:
            return iter(self.local_batches)
        else:
            raise RuntimeError(self.newsamplingerrormsg)


class DistributedHierarchicalPKSampler(DistributedMixin, HierarchicalPKSampler):
    """Distributes batches of HierarchicalPKSampler among the GPUs"""
    def __init__(self, *args, **kwargs):
        super(DistributedHierarchicalPKSampler, self).__init__(*args, **kwargs)

    def new_sampling(self, seed, lvl):
        super(DistributedHierarchicalPKSampler, self).new_sampling(seed, lvl)
        super(DistributedHierarchicalPKSampler, self)._assign_local_batches_to_rank()


class ClassSampler(Sampler):
    """Samples from a selected class of a selected level of the hierarchy"""
    def __init__(self, data_source: TextfileHierarchicalDataset, batch_size):
        super(ClassSampler, self).__init__(data_source)
        self.batch_size = batch_size
        self.data_source = data_source

    def get_dataset_classes_for_a_level(self, lvl):
        return list(self.data_source.labels_to_indices[lvl].keys())

    def get_all_lvls_classes(self):
        c = []
        for l in range(self.data_source.num_lvls):
            c += self.get_dataset_classes_for_a_level(l)
        return c

    def new_sampling(self, class_id, lvl):
        class_samples = self.data_source.labels_to_indices[lvl][class_id]
        self.indices_list = []
        i = 0
        while i < len(class_samples):
            if i + self.batch_size > len(class_samples):
                self.indices_list.append(class_samples[i:])
            else:
                self.indices_list.append(class_samples[i:i + self.batch_size])
            i += self.batch_size

    def __iter__(self):
        if self.indices_list:
            return iter(self.indices_list)
        else:
            raise RuntimeError("sampler.new_sampling(class_id, lvl) must be called")


class DistributedClassSampler(DistributedMixin, ClassSampler):
    """Distributes batches of ClassSampler among the GPUs"""
    def __init__(self, *args, **kwargs):
        super(DistributedClassSampler, self).__init__(*args, **kwargs)
        self.newsamplingerrormsg = "sampler.new_sampling(class_id, lvl) must be called"

    def new_sampling(self, class_id, lvl):
        super(DistributedClassSampler, self).new_sampling(class_id, lvl)
        super(DistributedClassSampler, self)._assign_local_batches_to_rank()

