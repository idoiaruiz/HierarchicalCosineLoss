"""This code implementation is from the PyTorch Metric Learning library:
https://kevinmusgrave.github.io/pytorch-metric-learning/ """

import torch
import torch.nn as nn


class CosineSimilarity:
    def __init__(self, normalize_embeddings=True, p=2, power=1, is_inverted=True):
        self.normalize_embeddings = normalize_embeddings
        self.p = p
        self.power = power
        self.is_inverted = is_inverted
        assert self.is_inverted
        assert self.normalize_embeddings

    def __call__(self, query_emb, ref_emb=None):
        query_emb_normalized = self.maybe_normalize(query_emb)
        if ref_emb is None:
            ref_emb = query_emb
            ref_emb_normalized = query_emb_normalized
        else:
            ref_emb_normalized = self.maybe_normalize(ref_emb)
        mat = self.compute_mat(query_emb_normalized, ref_emb_normalized)
        if self.power != 1:
            mat = mat ** self.power
        assert mat.size() == torch.Size((query_emb.size(0), ref_emb.size(0)))
        return mat

    def compute_mat(self, query_emb, ref_emb):
        return torch.matmul(query_emb, ref_emb.t())

    def pairwise_distance(self, query_emb, ref_emb):
        return torch.sum(query_emb * ref_emb, dim=1)

    def smallest_dist(self, *args, **kwargs):
        if self.is_inverted:
            return torch.max(*args, **kwargs)
        return torch.min(*args, **kwargs)

    def largest_dist(self, *args, **kwargs):
        if self.is_inverted:
            return torch.min(*args, **kwargs)
        return torch.max(*args, **kwargs)

    # This measures the margin between x and y
    def margin(self, x, y):
        if self.is_inverted:
            return y - x
        return x - y

    def normalize(self, embeddings, dim=1, **kwargs):
        return torch.nn.functional.normalize(embeddings, p=self.p, dim=dim, **kwargs)

    def maybe_normalize(self, embeddings, dim=1, **kwargs):
        if self.normalize_embeddings:
            return self.normalize(embeddings, dim=dim, **kwargs)
        return embeddings

    def get_norm(self, embeddings, dim=1, **kwargs):
        return torch.norm(embeddings, p=self.p, dim=dim, **kwargs)


class CosFaceLossML(nn.Module):
    def __init__(self, num_classes, feat_dim, s=1., margin=4., reduce=True):
        super(CosFaceLossML, self).__init__()
        self.margin = margin
        self.feat_dim = feat_dim
        self.num_classes = num_classes
        self.scale = s
        self.reduce = reduce  # True when there is the same number of samples per device (training from features).
        # Training from samples has different number of samples per device

        self.W = torch.nn.Parameter(torch.randn(feat_dim, num_classes))
        self.nllloss = nn.CrossEntropyLoss(reduction='mean')

        self.distance = CosineSimilarity()

    def get_cosine(self, embeddings):
        return self.distance(embeddings, self.W.t())

    def get_target_mask(self, embeddings, labels):
        batch_size = labels.size(0)
        mask = torch.zeros(batch_size, self.num_classes, dtype=embeddings.dtype).to(embeddings.device)
        mask[torch.arange(batch_size), labels] = 1
        return mask

    def modify_cosine_of_target_classes(self, cosine_of_target_classes):
        return cosine_of_target_classes - self.margin

    def scale_logits(self, logits):
        return logits * self.scale

    def cast_types(self, dtype, device):
        self.W.data = self.W.data.to(device).type(dtype)

    def forward(self, feat, labels):
        self.cast_types(feat.dtype, feat.device)

        mask = self.get_target_mask(feat, labels)
        cosine = self.get_cosine(feat)
        cosine_of_target_classes = cosine[mask == 1]
        modified_cosine_of_target_classes = self.modify_cosine_of_target_classes(cosine_of_target_classes)
        diff = (modified_cosine_of_target_classes - cosine_of_target_classes).unsqueeze(1)
        logits = cosine + (mask * diff)
        logits = self.scale_logits(logits)
        loss = self.nllloss(logits, labels)

        if not self.reduce:
            batch_samples = torch.as_tensor(labels.size(0), device=self.W.data.device)
        else:
            batch_samples = torch.as_tensor(0.0).to(self.W.data.device)
        return {'loss': loss, 'n_terms': batch_samples}

