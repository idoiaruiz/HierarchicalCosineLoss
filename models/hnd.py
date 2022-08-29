import torch
from torch.utils.data import DataLoader

from datasets.hierarchical_sampler import DistributedClassSampler
from metrics.hnd import DistributedHNDMetrics, compute_auc, get_metrics_at_fixed_points
from utils import custom_logging as logging


class HND:
    """Hierarchical novelty detection base model."""
    def __init__(self, threshold_range, hcl_module, device):
        self.threshold_range = threshold_range
        self.hcl_module = hcl_module
        self.device = device

    def test(self, dataset_test, dataset_novel, **kwargs):
        """Test HND on known test and novel splits."""
        self.hcl_module.eval()
        with torch.no_grad():
            known_ids = sum([list(dataset_test.labels_to_indices[l].keys())
                             for l in range(dataset_test.num_lvls+1)],[])  # all known classes
            logging.log_debug('known labels are: {} \n total {} labels'.format(
                [(cid, dataset_test.id_to_name[cid]) for cid in known_ids], len(known_ids)))

            parent_ids = sum([list(dataset_test.labels_to_indices[l].keys())
                             for l in range(1, dataset_test.num_lvls+1)],[])  # parent classes

            logging.log_info('Testing novelty detection on known split.', only_rank0=False)
            known_metrics_dict, known_cm = self._test_dataset(
                dataset_test, known_ids, parent_ids, data_type='known', **kwargs)
            logging.log_info('Testing novelty detection on novel split.', only_rank0=False)
            novel_metrics_dict, novel_cm = self._test_dataset(
                dataset_novel, known_ids, parent_ids, data_type='novel', **kwargs)

            auc = compute_auc(known_metrics_dict['acc'], novel_metrics_dict['acc'])
            metrics_dict = get_metrics_at_fixed_points(known_metrics_dict, novel_metrics_dict, auc, self.device)

        return metrics_dict, known_metrics_dict, novel_metrics_dict, known_cm, novel_cm

    def _test_dataset(self, dataset_test, known_ids, parent_ids, **kwargs):
        gt_ids, input_data = self._get_test_data(dataset_test=dataset_test, **kwargs)

        logging.log_debug('ground truth labels are: {} \n total {} labels'.format(
            [(cid, dataset_test.id_to_name[cid]) for cid in gt_ids], len(gt_ids)))

        split_metrics_dict = {'acc': [], 'avg_dist': [], 'avg_dist_lca': []}
        best_acc = 0
        best_cm = None
        for threshold_offset in torch.arange(*self.threshold_range):
            metrics_dict, cm = self._test_hnd_for_threshold(parent_ids, threshold_offset, gt_ids, known_ids,
                                                            kwargs['dist_mat'], kwargs['dist_to_lca_mat'], **input_data)
            for (n, v) in metrics_dict.items():
                split_metrics_dict[n].append(v)

            # We keep the confusion matrix of the best accuracy point.
            if metrics_dict['acc'] > best_acc:
                best_acc = metrics_dict['acc']
                best_cm = cm
            elif best_cm is None:
                best_cm = cm

        return split_metrics_dict, best_cm

    def _get_test_data(self, **kwargs):
        raise NotImplementedError

    def _test_hnd_for_threshold(self, parent_ids, bias, gt_ids, known_ids, dist_mat, dist_to_lca_mat, **kwargs):
        raise NotImplementedError


class HNDFromFeat(HND):
    """Hierarchical novelty detection from precomputed features."""
    def _get_test_data(self, **kwargs):
        split_data = kwargs['{}_data'.format(kwargs['data_type'])]
        feat = split_data['feat'].to(self.device)
        self.hcl_module.module.cast_types(feat.dtype, feat.device)
        scores = self.hcl_module.module.get_cosine(feat)  # cosine similarity to class prototypes
        labels = split_data['labels'].to(self.device)
        gt_ids = labels.unique().tolist()  # ground truth labels
        input_data = {'scores': scores,
                      'labels': labels}
        return gt_ids, input_data

    def _test_hnd_for_threshold(self, parent_ids, bias, gt_ids, known_ids, dist_mat, dist_to_lca_mat, **kwargs):
        """It performs novelty detection + classification for 'bias'. """
        logging.log_info('Testing novelty detection at bias {}'.format(bias), only_rank0=False)
        metrics = DistributedHNDMetrics(gt_ids, known_ids)

        biased_scores = kwargs['scores'].clone()
        biased_scores[:, parent_ids] += bias
        final_pred = biased_scores.argmax(1)

        metrics.batch_metrics(final_pred, kwargs['labels'], dist_mat, dist_to_lca_mat)
        acc, avg_dist, avg_dist_lca = metrics.compute_metrics()  # metrics for this threshold value
        logging.log_info('Accuracy, avg dist error, avg dist lca error at bias {}: {}, {}, {}'.format(
            bias, acc, avg_dist, avg_dist_lca), only_rank0=False)
        confusion_matrix = metrics.confusion_matrix
        cm_labels_pred = list(metrics.cm_labels_pred.keys())
        cm_labels_true = list(metrics.cm_labels_true.keys())
        metrics_dict = {'acc': acc * 0.01,
                        'avg_dist': avg_dist,
                        'avg_dist_lca': avg_dist_lca}
        del metrics

        return metrics_dict, {'data': confusion_matrix, 'labels_true': cm_labels_true, 'labels_pred': cm_labels_pred}


class HNDFromSamples(HND):
    """Hierarchical novelty detection from dataset samples."""
    def __init__(self, model, batch_size, threshold_range, hcl_module, device):
        super(HNDFromSamples, self).__init__(threshold_range, hcl_module, device)
        self.model = model
        self.batch_size = batch_size

    def test(self, dataset_test, dataset_novel, **kwargs):
        self.model.eval()
        return super(HNDFromSamples, self).test(dataset_test, dataset_novel, **kwargs)

    def _get_test_data(self, **kwargs):
        dataset_test = kwargs['dataset_test']
        test_sampler_inference = DistributedClassSampler(data_source=dataset_test, batch_size=self.batch_size)
        testloader_inference = DataLoader(dataset_test, batch_sampler=test_sampler_inference, num_workers=4)

        gt_ids = list(dataset_test.labels_to_indices[0].keys())  # Only leaf classes can be the true class for test
        # split, for novel split this returns the parent label
        input_data = {'test_sampler_inference': test_sampler_inference,
                      'testloader_inference': testloader_inference}
        return gt_ids, input_data

    def _test_hnd_for_threshold(self, parent_ids, bias, gt_ids, known_ids, dist_mat, dist_to_lca_mat, **kwargs):
        """It performs novelty detection + classification for 'bias'. """
        test_sampler_inference = kwargs['test_sampler_inference']
        testloader_inference = kwargs['testloader_inference']

        logging.log_info('Testing novelty detection at bias {}'.format(bias), only_rank0=False)
        metrics = DistributedHNDMetrics(gt_ids, known_ids)
        for c_id in test_sampler_inference.get_dataset_classes_for_a_level(0):
            # Load all the samples of class c_id
            test_sampler_inference.new_sampling(c_id, 0)
            if test_sampler_inference.local_batches:  # not empty
                for i_batch, (images, im_data) in enumerate(testloader_inference):
                    images = images.to(self.device)
                    labels = im_data[0]['class_id'].to(self.device)

                    feat = self.model(images)
                    scores = self.hcl_module.module.get_cosine(feat)  # cosine similarity to class prototypes
                    scores[:, parent_ids] += bias
                    final_pred = scores.argmax(1)

                    if testloader_inference.dataset.split == 'novel':
                        gt = torch.ones_like(final_pred, device=final_pred.device)*labels[0]
                        metrics.batch_metrics(final_pred, gt, dist_mat, dist_to_lca_mat)
                    else:
                        metrics.batch_metrics(final_pred, labels, dist_mat, dist_to_lca_mat)
            else:
                # Do extra iterations for DDP synchro.
                for i in range(test_sampler_inference.num_batches):
                    _ = self.model(torch.zeros_like(testloader_inference.dataset[0][0].unsqueeze(0).to(self.device)))

        # Synchro GPUs
        metrics.all_reduce()
        acc, avg_dist, avg_dist_lca = metrics.compute_metrics()  # metrics for this threshold value
        logging.log_info('Accuracy, avg dist error, avg dist lca error at bias {}: {}, {}, {}'.format(
            bias, acc, avg_dist, avg_dist_lca), only_rank0=False)
        confusion_matrix = metrics.confusion_matrix
        cm_labels_pred = list(metrics.cm_labels_pred.keys())
        cm_labels_true = list(metrics.cm_labels_true.keys())
        metrics_dict = {'acc': acc*0.01,
                        'avg_dist': avg_dist,
                        'avg_dist_lca': avg_dist_lca}
        del metrics

        return metrics_dict, {'data': confusion_matrix, 'labels_true': cm_labels_true, 'labels_pred': cm_labels_pred}
