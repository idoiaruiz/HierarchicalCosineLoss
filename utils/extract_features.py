import os
from glob import glob

import h5py
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from datasets.hierarchical_sampler import DistributedClassSampler
from datasets.textfile_hierarchical_dataset import TextfileHierarchicalDataset, NovelTextfileHierarchicalDataset
from utils import custom_logging as logging
from utils.utils import mk_dir


def create_dataloaders(dataset_name, image_size, da):
    transformations_no_da = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    if da > 1:  # Data augmentation
        transformations_da = transforms.Compose([
            transforms.RandomResizedCrop(image_size, scale=(0.8,1.0)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
    else:
        transformations_da = transformations_no_da

    dataset_train = TextfileHierarchicalDataset(
        dataset_name, split='train', transform=transformations_da, image_size=image_size)
    dataset_val = TextfileHierarchicalDataset(
        dataset_name, split='val', transform=transformations_no_da, image_size=image_size)
    dataset_test = TextfileHierarchicalDataset(
        dataset_name, split='test', transform=transformations_no_da, image_size=image_size)
    dataset_novel = NovelTextfileHierarchicalDataset(
        dataset_name, split='novel', transform=transformations_no_da, image_size=image_size)

    return dataset_train, dataset_val, dataset_test, dataset_novel


def run_model(model, feature_extractor, split_datasets, batch_size, device, feat_path, da_times):
    for split in split_datasets.keys():
        split_feat_path = os.path.join(feat_path, split)
        mk_dir(split_feat_path)

        logging.log_info('Extracting features for split {}'.format(split))
        data_split = split_datasets[split]
        test_sampler_inference = DistributedClassSampler(data_source=data_split, batch_size=batch_size)
        testloader_inference = DataLoader(data_split, batch_sampler=test_sampler_inference, num_workers=4)

        if split == 'train':
            repeat_data = da_times
        else:
            repeat_data = 1
        for t in range(repeat_data):
            for c_id in test_sampler_inference.get_dataset_classes_for_a_level(0):
                logging.log_info('Extracting features for class {} {}'.format(c_id, data_split.id_to_name[c_id]))
                # Load all the samples of class c_id
                test_sampler_inference.new_sampling(c_id, 0)
                if test_sampler_inference.local_batches:  # not empty
                    for i_batch, (images, im_data) in enumerate(testloader_inference):
                        images = images.to(device)
                        classes = im_data[0]['class_id'].to(device)
                        assert (classes == c_id).any(-1).all()
                        out = model(images)
                        feat = {'features': feature_extractor.features.squeeze(-1).squeeze(-1),
                                'labels': torch.ones(images.size(0))*c_id}

                        fn = os.path.join(split_feat_path, '{}_{}_{}.pt'.format(c_id, i_batch, t))
                        logging.log_debug('Saving features to {}'.format(fn))
                        torch.save(feat, fn)
                else:
                    # Do extra iterations for DDP synchro.
                    for i in range(test_sampler_inference.num_batches):
                        _ = model(torch.zeros_like(data_split[0][0].unsqueeze(0).to(device)))


def pack_features(dataset_name, dtypes, feat_path, exp_dir):
    """ Load files and create a full-batch file."""
    for dtype in dtypes:
        split_feat_path = os.path.join(feat_path, dtype)

        # load features
        all_features, all_labels = None, None
        feat_files = glob(os.path.join(split_feat_path, '*.pt'))
        for f in feat_files:
            data_dict = torch.load(f, map_location="cpu")
            logging.log_info('Loaded filename {} with classes {}'.format(f, data_dict['labels'].unique()))

            data = data_dict['features'].double()
            labels = data_dict['labels'].long()

            if all_features is not None:
                all_features = torch.cat([all_features, data], dim=0)
                all_labels = torch.cat([all_labels, labels], dim=0)
            else:
                all_features = data
                all_labels = labels

        all_features = all_features.numpy()
        all_labels = all_labels.numpy()

        save_path = '{exp_dir}/{cnn}_{dtype}.h5'.format(
            exp_dir=exp_dir, dataset=dataset_name, cnn='resnet101', dtype=dtype)
        with h5py.File(save_path, 'w') as f:
            f.create_dataset('data', data=all_features, compression='gzip', compression_opts=9)
            f.create_dataset('labels', data=all_labels, compression='gzip', compression_opts=9)

