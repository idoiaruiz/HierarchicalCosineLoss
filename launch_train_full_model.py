import argparse
import os

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import Adam, lr_scheduler
from torch.utils.data import DataLoader

from datasets.relabel_sampler import DistributedRelabelPKSampler
from datasets.textfile_hierarchical_dataset import TextfileHierarchicalDataset, NovelTextfileHierarchicalDataset
from metrics.hnd import log_confusion_matrix, log_metrics_params, log_curve
from metrics.plots import plot_auc, plot_xy, plot_dists_over_acc
from models import architectures
from models.hierarchical_cosine_loss import HierarchicalCosineLoss, HierarchicalCosineLossLowerMemory, weight_loss_terms
from models.hnd import HNDFromSamples as HND
from utils import custom_logging as logging
from utils.tensorboard import CustomWriter as SummaryWriter
from utils.utils import read_config_file, mk_dir

parser = argparse.ArgumentParser(description='Arguments')
parser.add_argument('--exp_dir', default='.', type=str, help='Experiment path')
parser.add_argument('--config', default='./config.ini', type=str, help='Path to config file')
parser.add_argument('--local_rank', type=int)
parser.add_argument('--checkpoint_dir', default=None, type=str, help='Path to checkpoint_dir')
parser.add_argument('--debug', action='store_true')
parser.add_argument('--only_eval', action='store_true')


def load_checkpoint(checkpoint, model):
    if checkpoint:
        state_dict = torch.load(checkpoint, map_location="cpu")
        model.load_state_dict(state_dict)
        logging.log_info('Loaded checkpoint from {}'.format(checkpoint))


def get_dataset(d_name, split, image_size):
    if split == 'novel':
        dataset = NovelTextfileHierarchicalDataset(d_name, split=split, image_size=image_size)
    else:
        dataset = TextfileHierarchicalDataset(d_name, split=split, image_size=image_size)
    return dataset


def create_datasets(dataset_name, image_size):
    dataset_train = get_dataset(dataset_name, 'train', image_size)
    dataset_val = get_dataset(dataset_name, 'val', image_size)
    dataset_test = get_dataset(dataset_name, 'test', image_size)
    dataset_novel = get_dataset(dataset_name, 'novel', image_size)
    return dataset_train, dataset_val, dataset_test, dataset_novel


def create_dataloaders(params):
    dataset_train, dataset_val, dataset_test, dataset_novel = create_datasets(params['data'].get('dataset'),
                                                                              params['data'].getint('image_size'))
    train_sampler = DistributedRelabelPKSampler(relabel_rate=params['model'].getfloat('relabel_rate'),
                                                P=params['training'].getint('batch_samples_per_class'),
                                                K=int(params['training'].getint('batch_size') /
                                                      params['training'].getint('batch_samples_per_class')),
                                                data_source=dataset_train,
                                                relabel_root=params['data'].getboolean('allow_children_of_root'))
    trainloader = DataLoader(dataset_train, batch_sampler=train_sampler, num_workers=4)
    return trainloader, dataset_val, dataset_test, dataset_novel


def training(device, exp_dir, writer, checkpoint_dir, params, debug, only_eval):
    """We train both the backbone model and the loss module. """
    trainloader, dataset_val, dataset_test, dataset_novel = create_dataloaders(params)
    num_classes = len(trainloader.dataset.id_to_name)

    model = architectures.ResNet101(num_classes, freeze_backbone=params['training'].getboolean('freeze_backbone'))
    feat_dim = model.feat_dim

    model = model.cuda(device)
    model = DDP(model, device_ids=[device], output_device=device)
    loss_fn = get_loss(feat_dim, num_classes, params, device, trainloader.dataset, False)
    if checkpoint_dir is not None:
        load_checkpoint(os.path.join(checkpoint_dir, "checkpoint_backbone.pth.tar"), model)
        load_checkpoint(os.path.join(checkpoint_dir, "checkpoint_loss.pth.tar"), loss_fn)

    if not only_eval:  # Training
        train(loss_fn, model, trainloader, dataset_val, dataset_novel, device, exp_dir, writer, params, debug)

    # Inference HND
    logging.log_info('Starting hnd inference')
    with torch.no_grad():
        hnd(loss_fn, model, dataset_test, dataset_novel, device, exp_dir, writer, params)


def get_loss(feat_dim, num_classes, params, device, dataset, reduce, lower_version=False):
    weights_list = params['training'].getlist('weights_hcl')
    margins_list = params['training'].getlist('margins_hcl')
    weights = {'cosface': weights_list[0], 'hcenters': weights_list[1], 'ctriplets': weights_list[2],
               'htriplets': weights_list[3]}
    if lower_version:
        loss_impl = HierarchicalCosineLossLowerMemory
    else:
        loss_impl = HierarchicalCosineLoss
    loss_fn = loss_impl(dataset, num_classes, feat_dim, device, weights, s=params['training'].getfloat('s'),
                        margin=margins_list[0], hcenters_margin=margins_list[1], ctriplets_margin=margins_list[2],
                        htriplets_margin=margins_list[3], reduce=reduce)

    loss_fn = loss_fn.cuda(device)
    loss_fn = DDP(loss_fn, device_ids=[device], output_device=device)

    return loss_fn


def train(loss_fn, model, dataset_loader, dataset_val, dataset_novel, device, exp_dir, writer, params, debug):
    loss_fn.train()
    model.train()

    lrs = params['training'].getlist('lr')
    optimizer = Adam([{'params': model.parameters(), 'lr': lrs[0]},
                      {'params': loss_fn.parameters(), 'lr': lrs[1]}])
    scheduler = lr_scheduler.StepLR(
        optimizer, step_size=params['training'].getint('lr_steps'), gamma=params['training'].getfloat('lr_gamma'))

    n_epochs = params['training'].getint('n_epochs')
    for epoch in range(n_epochs):
        dataset_loader.batch_sampler.new_sampling(epoch)
        sum_N, sum_losses = None, None
        for i_batch, ((images, im_data), current_local_labels) in enumerate(zip(
                dataset_loader, dataset_loader.batch_sampler.local_labels)):
            images = images.to(device)
            classes = torch.as_tensor(current_local_labels, dtype=torch.int64).to(device)  # Class ids loaded at this batch
            assert classes.size(0) == len(im_data[0]['class_id'])

            if debug:
                logging.log_debug('We are sampling classes {}, from all level classes {}'.format(classes, im_data))
                class_names = [dataset_loader.dataset.id_to_name[ci.item()] for ci in classes]
                for j, c_id in enumerate(classes):
                    logging.log_debug('Loaded image from class {}: {}'.format(c_id, class_names[j]))
                    for l in im_data:
                        logging.log_debug(
                            'Class at lvl {}: {} (id {})'.format(l, im_data[l]['class'][j], im_data[l]['class_id'][j]))

            out = model(images)

            optimizer.zero_grad()
            device_losses, reduced_losses, n_terms = loss_fn(out, classes)
            device_losses['loss'].backward()
            optimizer.step()
            scheduler.step()

            # log loss
            sum_N, sum_losses = log_batch_loss(reduced_losses, n_terms, sum_N, sum_losses, i_batch, epoch, writer, scheduler,
                                               device, debug, loss_fn.module.weights, params['training'].getint('log_rate'))

        if device == 0:
            # log epoch loss
            sum_losses = compute_total_loss(sum_losses, sum_N, loss_fn.module.weights)  # epoch loss
            logging.log_info('epoch %d,total loss: %.3f' % (epoch + 1, sum_losses['loss'].item()))
            if writer:
                writer.add_scalar("train/epoch_loss", sum_losses['loss'].item(), epoch + 1)
                writer.add_scalar("train/lr", scheduler.get_last_lr()[0], scheduler._step_count)

            # Save model
            checkpoint_path_backbone = os.path.join(exp_dir, "checkpoint_backbone.pth.tar")
            torch.save(model.state_dict(), checkpoint_path_backbone)
            checkpoint_path_loss = os.path.join(exp_dir, "checkpoint_loss.pth.tar")
            torch.save(loss_fn.state_dict(), checkpoint_path_loss)

        # Validation every val_rate epochs
        if (epoch+1) % params['training'].getint('val_rate') == 0:
            logging.log_info('Starting validation')
            with torch.no_grad():
                hnd(loss_fn, model, dataset_val, dataset_novel, device, exp_dir, writer, params, is_val=True,
                    global_step=epoch+1)

        if writer:
            writer.flush()


def log_batch_loss(reduced_losses, n_terms, sum_N, sum_losses, i_batch, epoch, writer, scheduler, device, debug,
                   loss_weights, log_rate):
    # Reduce across devices, not all devices have the same number of samples when training from samples
    for l in reduced_losses.keys():
        reduced_losses[l] = reduced_losses[l] * n_terms[l]  # undo mean reduction to sum
        dist.all_reduce(reduced_losses[l], op=dist.ReduceOp.SUM)
        dist.all_reduce(n_terms[l], op=dist.ReduceOp.SUM)

    if device == 0:
        # cumulate epoch losses
        if sum_N is None:
            sum_N = n_terms
            sum_losses = {n: v.clone() for (n, v) in reduced_losses.items()}
        else:
            for k in reduced_losses.keys():
                sum_N[k] += n_terms[k]
                sum_losses[k] += reduced_losses[k]

    reduced_losses = compute_total_loss(
        reduced_losses, n_terms, loss_weights)  # batch loss gathered from all devices
    logging.log_debug('loss {}'.format(reduced_losses['loss']))

    if (i_batch + 1) % log_rate == 0:
        logging.log_info('epoch %d, global_step %5d, loss: %.3f' % (
            epoch + 1, scheduler._step_count, reduced_losses['loss']))
        if debug:
            logging.log_debug(['{}: {}'.format(ln, lv) for (ln, lv) in reduced_losses.items()])
        if writer:
            for l_name, l_value in reduced_losses.items():
                writer.add_scalar("train/{}".format(l_name), l_value, scheduler._step_count)
    return sum_N, sum_losses


def compute_total_loss(losses_dict, n_terms_dict, weights):
    for l in losses_dict.keys():
        losses_dict[l] = torch.true_divide(losses_dict[l], n_terms_dict[l])
    losses_dict = weight_loss_terms(losses_dict, weights)
    return losses_dict


def hnd(hcl_module, model, dataset_known, dataset_novel, device, exp_dir, writer, params, is_val=False,
        global_step=None):
    hnd_model = HND(model, params['test'].getint('batch_size'), params['test'].getlist('threshold_range'), hcl_module,
                    device)
    if is_val:
        split = 'val'
        metrics_dir = os.path.join(exp_dir, 'val')
        if device == 0:
            mk_dir(metrics_dir)
    else:
        split = 'test'
        metrics_dir = exp_dir

    if device == 0:
        metrics_file = open(os.path.join(metrics_dir, "metrics.txt"), "a")
    else:
        metrics_file = None

    metrics_dict, known_metrics_dict, novel_metrics_dict, known_cm, novel_cm = hnd_model.test(
        dataset_known, dataset_novel, dist_mat=dataset_known.dist_mat, dist_to_lca_mat=dataset_known.dist_to_lca_mat)
    if device == 0:
        log_plot_file(metrics_dict, known_metrics_dict, novel_metrics_dict, metrics_dir)
        log_metrics(known_cm, novel_cm, metrics_dir, dataset_known, writer, known_metrics_dict, metrics_dict,
                    novel_metrics_dict, params, metrics_file, split, global_step=global_step)
        metrics_file.close()


def log_plot_file(metrics_dict, known_metrics_dict, novel_metrics_dict, exp_dir):
    plot_file = open(os.path.join(exp_dir, "plot_acc.csv"), "a")
    dist_plot_file = open(os.path.join(exp_dir, "plot_dist.csv"), "a")
    lcadist_plot_file = open(os.path.join(exp_dir, "plot_lcadist.csv"), "a")
    for r in known_metrics_dict['acc']:
        plot_file.write(str(r.item()) + ',')
        dist_plot_file.write(str(r.item()) + ',')
        lcadist_plot_file.write(str(r.item()) + ',')
    plot_file.write(';')
    dist_plot_file.write(';')
    lcadist_plot_file.write(';')
    for a,d,lcad in zip(novel_metrics_dict['acc'], novel_metrics_dict['avg_dist'], novel_metrics_dict['avg_dist_lca']):
        plot_file.write(str(a.item()) + ',')
        dist_plot_file.write(str(d.item()) + ',')
        lcadist_plot_file.write(str(lcad.item()) + ',')
    plot_file.write(';' + str(50.) + ';' + str(metrics_dict['novel_acc_at_50ka'].item() * 100.) + ';' +
                    str(70.) + ';' + str(metrics_dict['novel_acc_at_70ka'].item() * 100.) + ';' +
                    str(80.) + ';' + str(metrics_dict['novel_acc_at_80ka'].item() * 100.) + ';' +
                    str(metrics_dict['AUC'].item()) + '\n')
    dist_plot_file.write(';' + str(50.) + ';' + str(metrics_dict['novel_dist_at_50ka'].item()) +
                         ';' + str(70.) + ';' + str(metrics_dict['novel_dist_at_70ka'].item()) +
                         ';' + str(80.) + ';' + str(metrics_dict['novel_dist_at_80ka'].item()) + ';\n')
    lcadist_plot_file.write(';' + str(50.) + ';' + str(metrics_dict['novel_lcad_at_50ka'].item()) +
                            ';' + str(70.) + ';' + str(metrics_dict['novel_lcad_at_70ka'].item()) +
                            ';' + str(80.) + ';' + str(metrics_dict['novel_lcad_at_80ka'].item()) +';\n')
    plot_file.close()
    dist_plot_file.close()
    lcadist_plot_file.close()


def log_metrics(known_cm, novel_cm, exp_dir, dataset, writer, known_metrics_dict, metrics_dict,
                novel_metrics_dict, params, metrics_file, split='test', global_step=None):
    log_confusion_matrix(
        known_cm['data'], known_cm['labels_true'], known_cm['labels_pred'], dataset, writer, exp_dir,
        name='confusion_matrix_known', global_step=global_step, split=split)
    log_confusion_matrix(
        novel_cm['data'], novel_cm['labels_true'], novel_cm['labels_pred'], dataset, writer, exp_dir,
        name='confusion_matrix_novel', global_step=global_step, split=split)
    plot_auc(known_metrics_dict['acc'], novel_metrics_dict['acc'], metrics_dict['AUC'], 'Known accuracy',
             'Novel accuracy', exp_dir, 'auc')
    plot_xy(known_metrics_dict['avg_dist'], novel_metrics_dict['avg_dist'], 'Known error average distance',
            'Novel error average distance', exp_dir, title='avg_dist')
    plot_xy(known_metrics_dict['avg_dist_lca'], novel_metrics_dict['avg_dist_lca'], 'Known error average LCA distance',
            'Novel error average LCA distance', exp_dir, title='avg_dist_lca')
    plot_dists_over_acc(known_metrics_dict['acc'], known_metrics_dict['avg_dist'], novel_metrics_dict['avg_dist'],
                        'Known accuracy', 'Error average distance', exp_dir, y1_lab='Known', y2_lab='Novel',
                        title='avg_dist_over_known_acc')
    plot_dists_over_acc(known_metrics_dict['acc'], known_metrics_dict['avg_dist_lca'], novel_metrics_dict['avg_dist_lca'],
                        'Known accuracy', 'Error average LCA distance', exp_dir, y1_lab='Known', y2_lab='Novel',
                        title='avg_dist_lca_over_known_acc')
    log_curve(known_metrics_dict['acc'], novel_metrics_dict['acc'], 'acc_curve', writer, split=split)
    log_curve(known_metrics_dict['avg_dist'], novel_metrics_dict['avg_dist'], 'avg_dist_curve', writer,
              split=split)
    log_curve(known_metrics_dict['avg_dist_lca'], novel_metrics_dict['avg_dist_lca'], 'avg_dist_lca_curve', writer,
              split=split)
    log_curve(known_metrics_dict['acc'], known_metrics_dict['avg_dist'], 'known_avg_dist_over_known_acc', writer,
              split=split)
    log_curve(known_metrics_dict['acc'], known_metrics_dict['avg_dist_lca'], 'known_avg_dist_lca_over_known_acc',
              writer, split=split)
    log_curve(known_metrics_dict['acc'], novel_metrics_dict['avg_dist'], 'novel_avg_dist_over_known_acc', writer,
              split=split)
    log_curve(known_metrics_dict['acc'], novel_metrics_dict['avg_dist_lca'], 'novel_avg_dist_lca_over_known_acc',
              writer, split=split)
    log_metrics_params(metrics_dict, writer, params, metrics_file=metrics_file, split=split)


def main(args):
    # Read config file
    params = read_config_file(args.config)

    # DDP setup
    dist.init_process_group("nccl", init_method="env://")
    torch.cuda.set_device(args.local_rank)

    if args.local_rank == 0:
        mk_dir(args.exp_dir)
        logging.setup(args.exp_dir, 'train_full_model.log', debug=args.debug)
        writer = SummaryWriter(args.exp_dir)
    else:
        writer = None

    training(args.local_rank, args.exp_dir, writer, args.checkpoint_dir, params, args.debug, args.only_eval)

    if args.local_rank == 0:
        writer.close()
    dist.destroy_process_group()


if __name__ == '__main__':
    args = parser.parse_args()

    main(args)
