import torch

from utils import custom_logging as logging


class RelabelFeatures:
    """ Relabels features to parent class. Relabel is different on each device, because we don't share batches across
    devices, each device trains a new epoch with different data"""
    def __init__(self, train_data, dataset_train, relabel_rate, relabel_root, device, n_devices):
        self.train_data = train_data
        self.dataset_train = dataset_train
        self.relabel_rate = relabel_rate
        self.relabel_root = relabel_root
        self.device = device
        self.n_devices = n_devices

        self.leaf_ids = train_data['labels'].unique()
        self.generator = torch.Generator()

    def new_epoch_relabel(self, epoch):
        seed = self.n_devices*epoch + self.device
        self.generator.manual_seed(seed)
        self._relabel_feat()

        relabeled_data = {'feat': self.train_data['feat'], 'labels': self.relabels}
        return relabeled_data

    def _relabel_to_parent(self, c_id, unvisited):
        """Relabels relabel_rate% of labels to parent class """
        p_id = self.dataset_train.parents[c_id][0]
        if (not self.relabel_root) and (p_id == self.dataset_train.root):  # Not relabel as root
            return unvisited

        idx = torch.nonzero(self.relabels == c_id, as_tuple=False)
        # shuffling seed depends on the epoch and device to make sure relabeling is new at each epoch
        shuffled_idx = idx[torch.randperm(idx.size(0), generator=self.generator)]
        num_relabeled_samples = int(self.relabel_rate * 0.01 * idx.size(0) + 1)
        self.relabels[shuffled_idx[:num_relabeled_samples]] = p_id
        logging.log_debug('Relabeled {} samples of {} from {} to {} on device {}'.format(
            num_relabeled_samples, idx.size(0), self.dataset_train.id_to_name[c_id],
            self.dataset_train.id_to_name[p_id], self.device), only_rank0=False)
        if p_id not in unvisited:
            unvisited.append(p_id)

        return unvisited

    def _relabel_feat(self):
        unvisited = list(self.leaf_ids.numpy())
        self.relabels = self.train_data['labels'].clone().detach()

        # relabel leaves first, then super classes
        while unvisited:
            c_id = unvisited[0]
            unvisited = unvisited[1:]
            if c_id != self.dataset_train.root:
                unvisited = self._relabel_to_parent(c_id, unvisited)


class RelabelSamples:
    """Relabels samples randomly for each epoch. Relabel is done recursively from leaf classes to parent classes, until
    we reach the root class. """
    def __init__(self, dataset_train, relabel_rate, relabel_root):
        self.dataset_train = dataset_train
        self.relabel_rate = relabel_rate
        self.relabel_root = relabel_root

        self.leaf_ids = dataset_train.labels_to_indices[0].keys()
        self.train_samples = {c_id: samples_list for (c_id, samples_list) in dataset_train.labels_to_indices[0].items()}
        self.generator = torch.Generator()

    def new_epoch_relabel(self, seed):
        self.generator.manual_seed(seed)
        self._relabel_feat()

    def _relabel_to_parent(self, c_id, unvisited):
        """Relabels relabel_rate% of labels to parent class """
        p_id = self.dataset_train.parents[c_id][0]
        if (not self.relabel_root) and (p_id == self.dataset_train.root):  # not relabel as root
            return unvisited

        logging.log_debug('Class {} has {} samples before relabeling'.format(
            self.dataset_train.id_to_name[c_id], len(self.relabeled_samples[c_id])), only_rank0=True)

        # shuffling seed depends on the epoch to make sure relabeling is new at each epoch
        shuffled_idx = torch.randperm(len(self.relabeled_samples[c_id]), generator=self.generator)
        num_relabeled_samples = int(self.relabel_rate * 0.01 * shuffled_idx.size(0) + 1)

        # current class with less classes
        current_class_reduced_samples = [self.relabeled_samples[c_id][i] for i in shuffled_idx[num_relabeled_samples:]]
        new_samples_parent_class = [self.relabeled_samples[c_id][i] for i in shuffled_idx[:num_relabeled_samples]]

        self.relabeled_samples[c_id] = current_class_reduced_samples
        if p_id in self.relabeled_samples.keys():
            self.relabeled_samples[p_id] += new_samples_parent_class
        else:
            self.relabeled_samples[p_id] = new_samples_parent_class

        logging.log_debug('Relabeled {} samples from {} to {}'.format(
            num_relabeled_samples, self.dataset_train.id_to_name[c_id], self.dataset_train.id_to_name[p_id]),
            only_rank0=True)
        if p_id not in unvisited:
            unvisited.append(p_id)

        return unvisited

    def _relabel_feat(self):
        """Updates *relabeled_samples*, a dict with class ids as keys and a list of dataset indices as values. After
        relabeling, each known class has assigned samples."""
        unvisited = list(self.leaf_ids)
        self.relabeled_samples = self.train_samples.copy()

        # relabel leaves first, then super classes
        while unvisited:
            c_id = unvisited[0]
            unvisited = unvisited[1:]
            if c_id != self.dataset_train.root:
                unvisited = self._relabel_to_parent(c_id, unvisited)
