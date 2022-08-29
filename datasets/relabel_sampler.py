import torch
from torch.utils.data.sampler import Sampler

from datasets.textfile_hierarchical_dataset import TextfileHierarchicalDataset
from datasets.hierarchical_sampler import DistributedMixin
from utils import custom_logging as logging
from utils.relabel import RelabelSamples


class DistributedMixinWithLabels(DistributedMixin):
    def _assign_local_batches_to_rank(self):
        super(DistributedMixinWithLabels, self)._assign_local_batches_to_rank()
        i = self.rank * self.num_batches
        self.local_labels = iter(self.labels_list[i:i+self.num_batches])


class DistributedRelabelPKSampler(DistributedMixinWithLabels, Sampler):
    """
    Each batch loads P samples from K classes (that can be repeated) at a certain level of the taxonomy.
    Every epoch a new sampling is generated. Sampling is replicated on every device and batches are distributed
    across devices."""
    def __init__(self, relabel_rate, P, K, data_source: TextfileHierarchicalDataset, relabel_root):
        super(DistributedRelabelPKSampler, self).__init__(data_source=data_source)
        self.data_source = data_source
        self.P = P
        self.K = K
        self.generator = torch.Generator()
        self.newsamplingerrormsg = "sampler.new_sampling(epoch) must be called"

        self.sample_relabel = RelabelSamples(data_source, relabel_rate, relabel_root)

    def _relabel(self, seed):
        """Relabel samples from leaf to parent classes."""
        self.sample_relabel.new_epoch_relabel(seed)

        # flattened dict of classes and list of images (no levels!):
        relabeled_samples_dict = self.sample_relabel.relabeled_samples
        classes_to_sample = list(relabeled_samples_dict.keys())

        # check this is the list of all known classes (leaf + known parents)
        if self.sample_relabel.relabel_root:
            assertion = len(classes_to_sample) == len(self.data_source.id_to_name)
        else:
            assertion = len(classes_to_sample)+1 == len(self.data_source.id_to_name)
        assert assertion, logging.log_error(
            '{} does not match {}'.format(len(classes_to_sample), len(self.data_source.id_to_name)))  # +1 bc of root)
        return classes_to_sample, relabeled_samples_dict

    def _make_indices_batches(self, classes_to_sample, samples_dict, seed):
        """Same as textfile_hierarchical_dataset.HierarchicalPKSampler implementation but adding labels."""
        self.generator.manual_seed(seed)
        subsets = []
        labels = []
        for c_id in classes_to_sample:
            # for each class, permute its elements and divide them in subsets of P samples of a single class
            class_samples_permuted = [samples_dict[c_id][i] for i in torch.randperm(
                len(samples_dict[c_id]), generator=self.generator)]
            j = 0
            while j < len(class_samples_permuted):
                if j + self.P > len(class_samples_permuted):
                    subsets.append(class_samples_permuted[j:])
                    labels.append([c_id] * len(class_samples_permuted[j:]))
                else:
                    subsets.append(class_samples_permuted[j:j + self.P])
                    labels.append([c_id] * len(class_samples_permuted[j:j + self.P]))
                j += self.P

        # Permute subsets
        assert len(subsets) == len(labels)
        shuffle_idx = torch.randperm(len(subsets), generator=self.generator)
        subsets = [subsets[i] for i in shuffle_idx]
        labels = [labels[i] for i in shuffle_idx]

        i = 0
        self.indices_list = []
        self.labels_list = []
        while i < len(subsets):
            if i + self.K > len(subsets):
                self.indices_list.append(sum(subsets[i:], []))
                self.labels_list.append(sum(labels[i:], []))
            else:
                self.indices_list.append(sum(subsets[i:i + self.K], []))
                self.labels_list.append(sum(labels[i:i + self.K], []))
            i += self.K

    def new_sampling(self, epoch):
        """ Needs to be called each epoch to generate a different sampling."""
        logging.log_debug('Sampling epoch {}'.format(epoch), only_rank0=False)
        classes_to_sample, samples_dict = self._relabel(epoch)
        self._make_indices_batches(classes_to_sample, samples_dict, epoch)

        super(DistributedRelabelPKSampler, self)._assign_local_batches_to_rank()

