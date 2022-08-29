import torch
import torch.nn as nn

from models.cosfaceML import CosFaceLossML
from utils import custom_logging as logging

_DEF_WEIGHTS = {'cosface': 1.,
                'hcenters': 10.,
                'ctriplets': 100.,
                'htriplets': 10.}
_EPS = 1e-8


class HierarchicalCosineLoss(CosFaceLossML):
    def __init__(self, dataset, num_classes, feat_dim, device, weights=_DEF_WEIGHTS, s=1., margin=0.,
                 hcenters_margin=0.05, ctriplets_margin=0., htriplets_margin=0.,  reduce=True):
        super(HierarchicalCosineLoss, self).__init__(num_classes, feat_dim, s=s, margin=margin, reduce=reduce)
        self.weights = weights
        self.hcenters_margin = hcenters_margin
        self.htriplets_margin = htriplets_margin
        self.ctriplets_margin = ctriplets_margin

        hierarchical_dist = dataset.dist_mat
        assert num_classes == hierarchical_dist.shape[0]
        self._make_triplets(hierarchical_dist, dataset.root, device)

    def _make_triplets(self, hierarchical_dist, root, device):
        """Makes triplets of classes such that d_h(i,j) < d_h(i,k), where d_h is the hierarchical distance in the
        taxonomy tree"""
        with torch.no_grad():
            self.classes = torch.arange(hierarchical_dist.shape[0], device=device)

            if self.weights['hcenters'] > _EPS or self.weights['htriplets'] > _EPS:
                upper_diag = self.classes.unsqueeze(0) > self.classes.unsqueeze(1)
                upper_diag[root, :] = False  # remove root from triplets
                upper_diag[:, root] = False
                diff_classes = upper_diag.unsqueeze(0) * upper_diag.unsqueeze(1)  # triplets such that i<k, j<k
                i, j, k = torch.where(diff_classes)
                triplets_idx = torch.bitwise_and(hierarchical_dist[i, j] < hierarchical_dist[i, k],
                                                 hierarchical_dist[i, j] < hierarchical_dist[i, root])  # triplets such that d_h(i,j) < d_h(i,k)
                self.i = i[triplets_idx]
                self.j = j[triplets_idx]
                self.k = k[triplets_idx]

    def _compute_hcenters_loss(self):
        """ Computes Hierarchical Centers Loss. It encodes the constraint that cos(W_i,W_j) > cos(W_i,W_k) for triplets
        (i,j,k) such that d_h(i,j) < d_h(i,k), where W_i is the class prototype of class i, d_h is the hierarchical
        distance in the taxonomy tree and cos() is a measure of similarity.
        """
        n_terms = torch.as_tensor(0.0).to(self.W.data.device)
        hcenters_loss = torch.as_tensor(0.0).to(self.W.data.device)
        if self.weights['hcenters'] < _EPS:
            return {'loss': hcenters_loss, 'n_terms': n_terms}

        d = self.distance(self.W.t())  # cos(W_i,W_j) cosine similarity among class prototypes [n_classes, n_classes]
        dif = d[self.i, self.j] - d[self.i, self.k]

        y = -torch.ones(dif.size(0), device=dif.device)
        hcenters_loss = nn.functional.hinge_embedding_loss(dif, y, margin=self.hcenters_margin, reduction='mean')

        if not self.reduce:
            n_terms = torch.as_tensor(dif.size(0), device=self.W.data.device)
        return {'loss': hcenters_loss, 'n_terms': n_terms}

    def _compute_htriplets_loss(self, dist_f_w, class_samples):
        """ Computes Hierarchical Triplets Loss. It encodes the constraint that cos(x_i,W_j) > cos(x_i,W_k), being
        d_h(i,j) < d_h(i,k), where W_i is the class prototype of class i, d_h is the hierarchical distance in the
        taxonomy tree and cos() is a measure of similarity.
        """
        n_terms = torch.as_tensor(0.0).to(class_samples.device)
        htriplets_loss = torch.as_tensor(0.0).to(class_samples.device)
        if self.weights['htriplets'] < _EPS:
            return {'loss': htriplets_loss, 'n_terms': n_terms}

        # Triplets with samples and classes: we repeat every j,k class pair for every sample of class i
        i_samples, j, k = self._get_idx_htriplets(class_samples)
        dif = dist_f_w[i_samples, j] - dist_f_w[i_samples, k]

        y = -torch.ones(dif.size(0), device=dif.device)
        htriplets_loss = nn.functional.hinge_embedding_loss(dif, y, margin=self.htriplets_margin,
                                                            reduction='mean')
        if not self.reduce:
            n_terms = torch.as_tensor(dif.size(0), device=self.W.data.device)
        return {'loss': htriplets_loss, 'n_terms': n_terms}

    def _get_idx_htriplets(self, class_samples):
        """This implementation runs OOM on larger datasets (MTSD, CUB) when training from precomputed features.

        Returns:
            tuple: (i_samples, j, k) triplets of indices where i_samples are the indices of the samples of class i and
                j,k are indices of classes j,k such that d_h(i,j) < d_h(i,k)

        """
        with torch.no_grad():
            ijk_entries, samples_idx = torch.where(class_samples[self.i])  # where are the samples of class i

        return samples_idx, self.j[ijk_entries], self.k[ijk_entries]

    def _compute_ctriplets_loss(self, dist_f_w, class_samples):
        """ Computes C-triplets loss. It encodes that cos(x_i,W_i) should be > cos(x_i,W_j) for all j!=i """
        n_terms = torch.as_tensor(0.0).to(class_samples.device)
        ctriplets_loss = torch.as_tensor(0.0).to(class_samples.device)
        if self.weights['ctriplets'] < _EPS:
            return {'loss': ctriplets_loss, 'n_terms': n_terms}

        # We make triplets with samples and classes: we repeat every i,j class pair for every sample of class i
        i_samples, i, j = self._get_idx_ctriplets(class_samples)
        dif = dist_f_w[i_samples, i] - dist_f_w[i_samples, j]

        y = -torch.ones(dif.size(0), device=dif.device)
        ctriplets_loss = nn.functional.hinge_embedding_loss(dif, y, margin=self.ctriplets_margin, reduction='mean')
        if not self.reduce:
            n_terms = torch.as_tensor(dif.size(0), device=self.W.data.device)
        return {'loss': ctriplets_loss, 'n_terms': n_terms}

    def _get_idx_ctriplets(self, class_samples):
        """This implementation runs OOM on larger datasets (MTSD, CUB) when training from precomputed features.

        Returns:
            tuple: (i_samples, i, j) triplets of indices where i_samples are the indices of the samples of class
                i and i,j are indices of classes i!=j
        """
        with torch.no_grad():
            diff_class = self.classes.unsqueeze(0) != self.classes.unsqueeze(1)
            i, j = torch.where(diff_class)
            ij_entries, samples_idx = torch.where(class_samples[i])  # where are the samples of class i

        return samples_idx, i[ij_entries], j[ij_entries]

    def forward(self, feat, labels):
        losses = {}
        losses['cosface'] = super().forward(feat, labels)
        losses['hcenters'] = self._compute_hcenters_loss()

        dist_f_w, class_samples = None, None
        if self.weights['ctriplets'] > _EPS or self.weights['htriplets'] > _EPS:
            # cosine similarity among features and class prototypes [n_samples, n_classes]:
            dist_f_w = self.distance(feat, self.W.t())
            # samples of each class [n_classes, n_samples]:
            class_samples = labels.unsqueeze(0) == self.classes.unsqueeze(1)

        losses['ctriplets'] = self._compute_ctriplets_loss(dist_f_w, class_samples)
        losses['htriplets'] = self._compute_htriplets_loss(dist_f_w, class_samples)

        device_losses = {n: d['loss'] for (n,d) in losses.items()}
        n_terms = {n: d['n_terms'].detach() for (n,d) in losses.items()}
        unweighted_losses = {n: v.detach().clone() for (n,v) in device_losses.items()}

        device_losses = weight_loss_terms(device_losses, self.weights)
        return device_losses, unweighted_losses, n_terms


class HierarchicalCosineLossLowerMemory(HierarchicalCosineLoss):
    """Implementation that does not run OOM on 11GB GPUs when training from precomputed features on larger datasets."""
    def _get_idx_htriplets(self, class_samples):
        """Returns:
            tuple: (i_samples, j, k) triplets of indices where i_samples are the indices of the samples of class i and
                j,k are indices of classes j,k such that d_h(i,j) < d_h(i,k) """
        with torch.no_grad():
            samples_per_class = class_samples.sum(dim=1)
            jk = torch.stack([self.j, self.k], dim=1)
            jk = torch.repeat_interleave(jk, samples_per_class[self.i], dim=0)
            samples_entries_idx = torch.cat([torch.zeros(1, dtype=torch.int64, device=class_samples.device),
                                             class_samples.sum(dim=1).cumsum(dim=0)])
            _, samples_entries = torch.where(class_samples)

            #assert (self.i.sort()[0] == self.i).all()  # i is sorted
            i_samples_idx = -torch.ones(jk.size(0), dtype=torch.int64, device=jk.device)
            i_samples = self.i.unsqueeze(0) == self.classes.unsqueeze(1)
            i_classes = i_samples.sum(dim=1)
            i_entries_idx = torch.cat([torch.zeros(1, dtype=torch.int64, device=class_samples.device),
                                       (i_classes * samples_per_class).cumsum(dim=0)])
            for c_id in range(class_samples.size(0)):  # for each class
                c_samples_c_id = samples_entries[
                                 samples_entries_idx[c_id]: samples_entries_idx[c_id + 1]]  # samples idx list for c_id
                i_samples_idx[i_entries_idx[c_id]: i_entries_idx[c_id + 1]] = c_samples_c_id.repeat(i_classes[c_id])

        return i_samples_idx, jk[:, 0], jk[:, 1]

    def _get_idx_ctriplets(self, class_samples):
        """Returns:
                tuple: (i_samples, i, j) triplets of indices where i_samples are the indices of the samples of class
                    i and i,j are indices of classes i!=j """
        with torch.no_grad():
            diff_class = self.classes.unsqueeze(0) != self.classes.unsqueeze(1)
            i, j = torch.where(diff_class)

            samples_per_class = class_samples.sum(dim=1)
            ij = torch.stack([i, j], dim=1)
            ij = torch.repeat_interleave(ij, samples_per_class[i], dim=0)
            samples_entries_idx = torch.cat([torch.zeros(1, dtype=torch.int64, device=class_samples.device),
                                             class_samples.sum(dim=1).cumsum(dim=0)])
            _, samples_entries = torch.where(class_samples)

            #assert (i.sort()[0] == i).all()  # i is sorted
            i_samples_idx = -torch.ones(ij.size(0), dtype=torch.int64, device=ij.device)
            i_samples = i.unsqueeze(0) == self.classes.unsqueeze(1)
            i_classes = i_samples.sum(dim=1)
            i_entries_idx = torch.cat([torch.zeros(1, dtype=torch.int64, device=class_samples.device),
                                       (i_classes * samples_per_class).cumsum(dim=0)])
            for c_id in range(class_samples.size(0)):  # for each class
                c_samples_c_id = samples_entries[
                                 samples_entries_idx[c_id]: samples_entries_idx[c_id + 1]]  # samples idx list for c_id
                i_samples_idx[i_entries_idx[c_id]: i_entries_idx[c_id + 1]] = c_samples_c_id.repeat(i_classes[c_id])

        return i_samples_idx, ij[:,0], ij[:,1]


def weight_loss_terms(losses_dict, weights):
    for (k, v) in losses_dict.items():  # Regularization
        losses_dict[k] = weights[k] * v
    losses_dict['loss'] = sum(losses_dict.values())  # Total loss
    return losses_dict