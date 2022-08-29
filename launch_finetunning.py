import argparse
import os

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import Adam, lr_scheduler
from torch.utils.data import DataLoader

from datasets.hierarchical_sampler import DistributedHierarchicalPKSampler
from launch_train_full_model import create_datasets, load_checkpoint
from models import architectures
from models.architectures import FeatureExtractor
from utils.extract_features import create_dataloaders as create_dataloaders_extract_feat
from utils.extract_features import run_model, pack_features
from utils import custom_logging as logging
from utils.tensorboard import CustomWriter as SummaryWriter
from utils.utils import read_config_file, mk_dir

parser = argparse.ArgumentParser(description='Arguments')
parser.add_argument('--exp_dir', default='.', type=str, help='Experiment path')
parser.add_argument('--config', default='./config.ini', type=str, help='Path to config file')
parser.add_argument('--local_rank', type=int)
parser.add_argument('--debug', action='store_true')
parser.add_argument('--checkpoint', default=None, type=str, help='Path to checkpoint')


def create_dataloader(params):
    dataset_train, _, _, _ = create_datasets(params['data'].get('dataset'), params['data'].getint('image_size'))
    train_sampler = DistributedHierarchicalPKSampler(P=params['training'].getint('batch_samples_per_class'),
                                                     K=int(params['training'].getint('batch_size') /
                                                           params['training'].getint('batch_samples_per_class')),
                                                     data_source=dataset_train)
    trainloader = DataLoader(dataset_train, batch_sampler=train_sampler, num_workers=4)
    return trainloader


def finetunning(device, exp_dir, writer, params, checkpoint):
    trainloader = create_dataloader(params)
    num_classes = len(trainloader.dataset.id_to_name)

    model = architectures.ResNet101(num_classes, freeze_backbone=False, return_feat=False, finetune=True)
    model = model.cuda(device)
    model = DDP(model, device_ids=[device], output_device=device)

    if checkpoint is not None:
        load_checkpoint(checkpoint, model)

    if checkpoint is None:
        train(model, trainloader, device, exp_dir, writer, params)
    del trainloader

    extract_features(model, device, exp_dir, params)


def train(model, dataset_loader, device, exp_dir, writer, params):
    model.train()

    loss_fn = torch.nn.CrossEntropyLoss(reduction='mean')
    optimizer = Adam(model.parameters(), lr=params['training'].getfloat('lr'))
    scheduler = lr_scheduler.StepLR(
        optimizer, step_size=params['training'].getint('lr_steps'), gamma=params['training'].getfloat('lr_gamma'))

    n_epochs = params['training'].getint('n_epochs')
    for epoch in range(n_epochs):
        dataset_loader.batch_sampler.new_sampling(epoch, 0)
        running_loss = torch.as_tensor(0.0).to(device)
        for i_batch, (images, im_data) in enumerate(dataset_loader):
            images = images.to(device)
            classes = im_data[0]['class_id'].to(device)

            out = model(images)

            optimizer.zero_grad()
            loss = loss_fn(out, classes)
            loss.backward()
            optimizer.step()
            scheduler.step()

            # log loss
            device_loss = loss * images.size(0)
            running_loss += device_loss
            batch_sizes = torch.as_tensor(images.size(0), device=device)
            dist.all_reduce(device_loss, op=dist.ReduceOp.SUM)
            dist.all_reduce(batch_sizes, op=dist.ReduceOp.SUM)
            device_loss = torch.true_divide(device_loss, batch_sizes)  # batch size might be different on diff devices
            logging.log_debug('loss {}'.format(device_loss))

            if (i_batch+1) % params['training'].getint('log_rate') == 0:
                logging.log_info('epoch %d, global_step %5d, loss: %.3f' % (epoch + 1, scheduler._step_count, device_loss))
                if writer:
                    writer.add_scalar("train/loss", device_loss, scheduler._step_count)

        dist.all_reduce(running_loss, op=dist.ReduceOp.SUM)
        epoch_loss = torch.true_divide(running_loss, len(dataset_loader.dataset))
        logging.log_info('epoch %d,total loss: %.3f' % (epoch + 1, epoch_loss.item()))
        if writer:
            writer.add_scalar("train/epoch_loss", epoch_loss.item(), epoch + 1)
            writer.add_scalar("train/lr", scheduler.get_last_lr()[0], scheduler._step_count)

        # Save model
        if device == 0:
            torch.save(model.state_dict(), os.path.join(exp_dir, "checkpoint.pth.tar"))

        # Validation every val_rate epochs
        if (epoch+1) % params['training'].getint('val_rate') == 0:
            pass

        if writer:
            writer.flush()


def extract_features(model, device, exp_dir, params):
    feat_path = os.path.join(exp_dir, "features")
    mk_dir(feat_path)

    dataset_name = params['data'].get('dataset')
    da = 1
    dataset_train, dataset_val, dataset_test, dataset_novel = create_dataloaders_extract_feat(
        dataset_name, params['data'].getint('image_size'), da)
    split_datasets = {'train': dataset_train, 'val': dataset_val, 'known': dataset_test, 'novel': dataset_novel}

    feature_extractor = FeatureExtractor(model.module.backbone.avgpool)

    # Extract features in batches and save them in files
    model.eval()
    with torch.no_grad():
        run_model(model, feature_extractor, split_datasets, params['test'].getint('batch_size'), device, feat_path, da)

    # Load files and create a full-batch file
    pack_features(dataset_name, list(split_datasets.keys()), feat_path, exp_dir)


def main(args):
    # Read config file
    params = read_config_file(args.config)

    # DDP setup
    dist.init_process_group("nccl", init_method="env://")
    torch.cuda.set_device(args.local_rank)

    if args.local_rank == 0:
        mk_dir(args.exp_dir)
        logging.setup(args.exp_dir, 'finetunning.log', debug=args.debug)
        writer = SummaryWriter(args.exp_dir)
    else:
        writer = None

    finetunning(args.local_rank, args.exp_dir, writer, params, args.checkpoint)

    if args.local_rank == 0:
        writer.close()
    dist.destroy_process_group()


if __name__ == '__main__':
    args = parser.parse_args()

    main(args)
