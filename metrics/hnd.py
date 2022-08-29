import os
from collections import OrderedDict

import torch
import torch.distributed as dist
from sklearn.metrics import auc

from metrics.plots import plot_confusion_matrix, plot_to_image
from utils import custom_logging as logging
from utils.tensorboard import write_params


class DistributedHNDMetrics:
    def __init__(self, true_ids, predicted_ids):
        if not dist.is_available():
            raise RuntimeError("Requires distributed package to be available")
        self.num_replicas = dist.get_world_size()
        self.rank = dist.get_rank()

        self.correct = torch.as_tensor(0).to(self.rank)
        self.total = torch.as_tensor(0).to(self.rank)

        # Confusion matrix. Dim 0 predicted, dim 1 ground truth class
        self.confusion_matrix = torch.zeros([len(predicted_ids), len(true_ids)], dtype=torch.int, device=self.rank)
        self.cm_labels_pred = OrderedDict([(c_id, idx) for (idx, c_id) in enumerate(predicted_ids)])
        self.cm_labels_true = OrderedDict([(c_id, idx) for (idx, c_id) in enumerate(true_ids)])

        # Error distances
        self.error_lca_dist_sum = torch.as_tensor(0).to(self.rank)
        self.error_dist_sum = torch.as_tensor(0).to(self.rank)

    def batch_metrics(self, predicted, classes, dist_mat=None, dist_to_lca_mat=None):
        # Compute total acc and acc per class and hierarchical distance of errors
        c = (predicted == classes)
        for (label, pred, is_correct) in zip(classes, predicted, c):
            self.confusion_matrix[self.cm_labels_pred[pred.item()], self.cm_labels_true[label.item()]] += 1
            if (dist_mat is not None) and (not is_correct):
                self.error_lca_dist_sum += dist_to_lca_mat[label.item(), pred.item()]
                self.error_dist_sum += dist_mat[label.item(), pred.item()]

        self.total += predicted.size(0)
        self.correct += c.sum().item()

    def all_reduce(self):
        dist.all_reduce(self.total, op=dist.ReduceOp.SUM)
        dist.all_reduce(self.correct, op=dist.ReduceOp.SUM)
        dist.all_reduce(self.confusion_matrix, op=dist.ReduceOp.SUM)
        dist.all_reduce(self.error_lca_dist_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(self.error_dist_sum, op=dist.ReduceOp.SUM)

    def _compute_acc(self):
        acc = 100 * torch.true_divide(self.correct, self.total)
        return acc

    def _compute_avg_error_dist(self):
        error_lca_dist = torch.true_divide(self.error_lca_dist_sum, self.total)
        error_dist = torch.true_divide(self.error_dist_sum, self.total)
        return error_dist, error_lca_dist

    def compute_metrics(self):
        acc = self._compute_acc()
        avg_dist_error, avg_lca_dist_error = self._compute_avg_error_dist()
        return acc, avg_dist_error, avg_lca_dist_error


def compute_auc(known_acc, novel_acc):
    return auc(known_acc, novel_acc)*100


def log_metrics_params(metrics_dict, writer, config, metrics_file=None, split='test'):
    if metrics_file:
        for (n,v) in metrics_dict.items():
            metrics_file.write('{}/{}: {}\n'.format(split, n, v))

    epoch = config['training'].getint('n_epochs')
    write_params(writer, config, metrics_dict, epoch=epoch, split=split)

    for (n, v) in metrics_dict.items():
        logging.log_info('{} on {} images: {}'.format(n, split, v))


def log_curve(x, y, name, writer, split='test'):
    if writer:
        for (x_i, y_i) in (zip(x, y)):
            writer.add_scalar('{}/{}'.format(split, name), y_i*100, x_i*100)


def log_confusion_matrix(cm, ids_true, ids_pred, dataset, writer, exp_dir, name='confusion_matrix', global_step=None,
                         split='test'):
    """Log the confusion matrix as an image summary."""
    class_names_true = [dataset.id_to_name[id] for id in ids_true]
    class_names_pred = [dataset.id_to_name[id] for id in ids_pred]
    write_confusion_matrix(cm, class_names_true, class_names_pred, exp_dir, name)
    figure = plot_confusion_matrix(cm.cpu().numpy(), class_names_true=class_names_true, class_names_pred=class_names_pred)
    cm_image = plot_to_image(figure, name=name, exp_dir=exp_dir)
    if writer:
        writer.add_image('{}/{}'.format(split, name), cm_image, global_step)


def write_confusion_matrix(cm, class_names_true, class_names_pred, exp_dir, name):
    metrics_file = open(os.path.join(exp_dir, '{}.csv'.format(name)), "a")
    assert (cm.size(0) == len(class_names_pred)) & (cm.size(1) == len(class_names_true))

    # Header
    metrics_file.write(',True labels \n')
    metrics_file.write('Predicted labels,')
    for c in class_names_true:
        metrics_file.write('{},'.format(c))
    metrics_file.write('\n')

    # Normalize CM over columns
    cm = cm.float()
    for j in range(cm.size(1)):
        total_pred = cm[:,j].sum()
        cm[:, j] = torch.true_divide(cm[:,j], total_pred)

    # Write CM
    for i, c in enumerate(class_names_pred):
        metrics_file.write('{},'.format(c))
        for j in range(len(class_names_true)):
            metrics_file.write('{},'.format(cm[i,j]))
        metrics_file.write('\n')

    metrics_file.close()


def get_metrics_at_fixed_points(known_metrics_dict, novel_metrics_dict, auc, device):
    met_at_50ka = get_metrics_at_known_acc(
        known_metrics_dict['acc'], novel_metrics_dict['acc'], novel_metrics_dict['avg_dist'],
        novel_metrics_dict['avg_dist_lca'], acc_point=0.5, device=device)
    met_at_70ka = get_metrics_at_known_acc(
        known_metrics_dict['acc'], novel_metrics_dict['acc'], novel_metrics_dict['avg_dist'],
        novel_metrics_dict['avg_dist_lca'], acc_point=0.7, device=device)
    met_at_80ka = get_metrics_at_known_acc(
        known_metrics_dict['acc'], novel_metrics_dict['acc'], novel_metrics_dict['avg_dist'],
        novel_metrics_dict['avg_dist_lca'], acc_point=0.8, device=device)

    metrics_dict = {'AUC': auc,
                    'novel_acc_at_50ka': met_at_50ka['novel_acc'],
                    'novel_dist_at_50ka': met_at_50ka['novel_dist'],
                    'novel_lcad_at_50ka': met_at_50ka['novel_lca_dist'],
                    'novel_acc_at_70ka': met_at_70ka['novel_acc'],
                    'novel_dist_at_70ka': met_at_70ka['novel_dist'],
                    'novel_lcad_at_70ka': met_at_70ka['novel_lca_dist'],
                    'novel_acc_at_80ka': met_at_80ka['novel_acc'],
                    'novel_dist_at_80ka': met_at_80ka['novel_dist'],
                    'novel_lcad_at_80ka': met_at_80ka['novel_lca_dist']
                    }
    return metrics_dict


def get_metrics_at_known_acc(known_acc_l, novel_acc_l, novel_dist_l, novel_lcad_l, acc_point=0.5, device=0):
    """Gets metrics at fixed known acc points via linear interpolation"""
    metrics_at_acc_point = {}

    known_acc = torch.as_tensor(known_acc_l).to(device)
    novel_acc = torch.as_tensor(novel_acc_l).to(device)
    novel_dist = torch.as_tensor(novel_dist_l).to(device)
    novel_lcad = torch.as_tensor(novel_lcad_l).to(device)

    loc = (known_acc - acc_point).abs().argmin()
    closest = known_acc[loc]

    if closest < acc_point:
        locs = torch.as_tensor([max(loc - 1, 0), loc]).to(device)
    elif closest == acc_point:
        locs = torch.as_tensor([loc, loc]).to(device)
    else:
        locs =torch.as_tensor([loc, min(loc + 1, len(known_acc) - 1)]).to(device)

    # linear interpolation
    x = known_acc[locs]
    if x[0] == x[1]:
        metrics_at_acc_point['novel_acc'] = novel_acc[loc]
        metrics_at_acc_point['novel_dist'] = novel_dist[loc]
        metrics_at_acc_point['novel_lca_dist'] = novel_lcad[loc]
    else:
        y = novel_acc[locs]
        metrics_at_acc_point['novel_acc'] = (acc_point - x[0]) * (y[1] - y[0]) / (x[1] - x[0]) + y[0]

        y = novel_dist[locs]
        metrics_at_acc_point['novel_dist'] = (acc_point - x[0]) * (y[1] - y[0]) / (x[1] - x[0]) + y[0]

        y = novel_lcad[locs]
        metrics_at_acc_point['novel_lca_dist'] = (acc_point - x[0]) * (y[1] - y[0]) / (x[1] - x[0]) + y[0]

    return metrics_at_acc_point
