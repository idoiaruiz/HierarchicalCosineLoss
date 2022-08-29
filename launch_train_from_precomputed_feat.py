import argparse
import os

import h5py
import torch
import torch.distributed as dist
from torch.optim import Adam, lr_scheduler

from launch_train_full_model import get_loss, create_datasets, load_checkpoint, log_metrics, log_plot_file
from models.hnd import HNDFromFeat as HND
from utils import custom_logging as logging
from utils.relabel import RelabelFeatures
from utils.tensorboard import CustomWriter as SummaryWriter
from utils.utils import read_config_file, mk_dir

parser = argparse.ArgumentParser(description='Arguments')
parser.add_argument('--exp_dir', default='.', type=str, help='Experiment path')
parser.add_argument('--config', default='./config.ini', type=str, help='Path to config file')
parser.add_argument('--local_rank', type=int)
parser.add_argument('--checkpoint', default=None, type=str, help='Path to checkpoint')
parser.add_argument('--debug', action='store_true')
parser.add_argument('--only_eval', action='store_true')


def load_precomputed_feat(dataset_name):
    data = []
    for dtype in ['train', 'val', 'known', 'novel']:
        data_path = 'features/{dataset}/resnet101_{dtype}.h5'.format(dataset=dataset_name, dtype=dtype)
        with h5py.File(data_path, 'r') as f:
            inputs = torch.from_numpy(f['data'][:])
            targets = torch.from_numpy(f['labels'][:])
            data.append({'feat': inputs, 'labels': targets})

    return data[0], data[1], data[2], data[3]


def training(device, exp_dir, writer, params, debug, only_eval, checkpoint):
    """We load the precomputed features and only train the loss module. """
    dataset_train, _, _, _ = create_datasets(params['data'].get('dataset'), params['data'].getint('image_size'))
    # Dataset instances on this script are used only to get taxonomy information, we don't use the samples of their splits
    num_classes = len(dataset_train.id_to_name)

    train_data, val_data, test_data, novel_data = load_precomputed_feat(params['data'].get('dataset'))
    feat_dim = train_data['feat'].size(1)
    logging.log_debug('train_data feat size is {} and feat dim is {}'.format(train_data['feat'].size(), feat_dim))

    loss_fn = get_loss(feat_dim, num_classes, params, device, dataset_train, True,
                       lower_version=params['data'].get('dataset') in ['CUB', 'mtsd'])
    load_checkpoint(checkpoint, loss_fn)

    if not only_eval:  # Training
        # Relabel train data
        n_devices = dist.get_world_size()
        relabel = RelabelFeatures(train_data, dataset_train, params['model'].getfloat('relabel_rate'),
                                  params['data'].getboolean('allow_children_of_root'), device, n_devices)

        # a full-batch epoch is trained at each gpu, batches are not distributed across devices
        n_epochs = params['training'].getint('n_epochs') // n_devices
        logging.log_info('Training {} epochs at each gpu'.format(n_epochs))
        train(loss_fn, relabel, device, exp_dir, writer, params, n_epochs, debug, val_data=val_data,
              novel_data=novel_data)

    # Inference HND
    logging.log_info('Starting hnd inference')
    with torch.no_grad():
        hnd(loss_fn, test_data, novel_data, dataset_train, device, exp_dir, writer, params)


def train(loss_fn, relabel, device, exp_dir, writer, params, n_epochs, debug, ckpt_name='checkpoint', **kwargs):
    loss_fn.train()

    optimizer = Adam(loss_fn.parameters(), lr=params['training'].getfloat('lr'))
    scheduler = lr_scheduler.StepLR(
        optimizer, step_size=params['training'].getint('lr_steps'), gamma=params['training'].getfloat('lr_gamma'))
    for epoch in range(n_epochs):
        # Relabel train data
        with torch.no_grad():
            relabeled_data = relabel.new_epoch_relabel(epoch)

        # Full batch feat and labels:
        feat = relabeled_data['feat'].to(device)
        classes = relabeled_data['labels'].to(device)

        optimizer.zero_grad()
        losses, _, _ = loss_fn(feat, classes)
        losses['loss'].backward()
        optimizer.step()
        scheduler.step()

        # log loss
        for l in losses.keys():
            dist.all_reduce(losses[l], op=dist.ReduceOp.SUM)
            losses[l] = torch.true_divide(losses[l], dist.get_world_size())  # Same number of samples on each device
        logging.log_info('epoch %d, global_step %5d, loss: %.3f' % (epoch + 1, scheduler._step_count, losses['loss']))
        if writer:
            for l_name, l_value in losses.items():
                writer.add_scalar("train/{}".format(l_name), l_value, scheduler._step_count)
            writer.add_scalar("train/epoch_loss", losses['loss'].item(), epoch + 1)
            writer.add_scalar("train/lr", scheduler.get_last_lr()[0], scheduler._step_count)

        # Save model
        if device == 0:
            checkpoint_path = os.path.join(exp_dir, "{}_loss.pth.tar".format(ckpt_name))
            torch.save(loss_fn.state_dict(), checkpoint_path)

        # Validation every val_rate epochs
        if (epoch+1) % params['training'].getint('val_rate') == 0:
            logging.log_info('Starting hnd validation')
            with torch.no_grad():
                hnd(loss_fn, kwargs['val_data'], kwargs['novel_data'], relabel.dataset_train, device, exp_dir, writer,
                    params, is_val=True, global_step=epoch+1)

        if writer:
            writer.flush()


def hnd(hcl_module, known_data, novel_data, dataset, device, exp_dir, writer, params, is_val=False, global_step=None):
    if device == 0:  # Not distributed
        hnd_model = HND(params['test'].getlist('threshold_range'), hcl_module, device)
        if is_val:
            split = 'val'
            metrics_dir = os.path.join(exp_dir, 'val')
            if device == 0:
                mk_dir(metrics_dir)
        else:
            split = 'test'
            metrics_dir = exp_dir
        metrics_file = open(os.path.join(metrics_dir, "metrics.txt"), "a")

        metrics_dict, known_metrics_dict, novel_metrics_dict, known_cm, novel_cm = hnd_model.test(
            dataset, dataset, known_data=known_data, novel_data=novel_data,
            dist_mat=dataset.dist_mat, dist_to_lca_mat=dataset.dist_to_lca_mat)
        log_plot_file(metrics_dict, known_metrics_dict, novel_metrics_dict, metrics_dir)
        log_metrics(known_cm, novel_cm, metrics_dir, dataset, writer, known_metrics_dict, metrics_dict,
                    novel_metrics_dict, params, metrics_file, split, global_step=global_step)
        metrics_file.close()


def main(args):
    # Read config file
    params = read_config_file(args.config)

    # DDP setup
    dist.init_process_group("nccl", init_method="env://")
    torch.cuda.set_device(args.local_rank)

    if args.local_rank == 0:
        mk_dir(args.exp_dir)
        writer = SummaryWriter(args.exp_dir)
    else:
        writer = None
    logging.setup(args.exp_dir, 'train_from_precomputed_feat.log', debug=args.debug)

    if args.only_eval:
        assert args.checkpoint is not None, logging.log_error('If only eval, a checkpoint must be provided')

    # Do not train backbone, use precomputed saved features
    training(args.local_rank, args.exp_dir, writer, params, args.debug, args.only_eval, args.checkpoint)

    if args.local_rank == 0:
        writer.close()
    dist.destroy_process_group()


if __name__ == '__main__':
    args = parser.parse_args()

    main(args)
