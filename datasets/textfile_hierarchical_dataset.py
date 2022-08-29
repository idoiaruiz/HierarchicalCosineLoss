from collections import OrderedDict

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from utils import custom_logging as logging
from utils.data_utils import load_split_info

_DATASETS = ['CUB', 'AWA2', 'tsinghua', 'mtsd']
_TRAFFIC_SIGNS = ['tsinghua', 'mtsd']


class TextfileHierarchicalDataset(Dataset):
    """The taxonomy of the dataset is loaded from a text file (taxonomy/{dataset}/taxonomy.txt), with format:

    root
        class 1
            child 1 of class 1
    Lists of known and novel leaf classes are expected to be at taxonomy/{dataset}/known.txt and
    taxonomy/{dataset}/novel.txt, respectively.
    Splits samples information is loaded from taxonomy/{dataset_name}/splits_data/filenames_{split}.npy. These files
    include the paths to the images.
    ...
    """
    def __init__(self, dataset_name, split='train', extension='png', transform=None, image_size=224):
        if dataset_name not in _DATASETS:
            raise ValueError('Unknown dataset')
        if split not in ['train', 'val', 'test', 'novel']:
            raise ValueError('Unknown split')

        self.split = split
        self.extension = extension
        self.image_size = image_size
        self._set_preprocessing(transform, dataset_name)

        self.images, self.labels_to_indices = self._load_info_from_split_data(dataset_name)

        logging.log_info('{} hierarchical dataset created.'.format(split))

    def _set_preprocessing(self, transform, dataset_name):
        if transform:
            transformations = transform
        else:
            if self.split == 'train':
                if dataset_name in _TRAFFIC_SIGNS:
                    transformations = transforms.Compose([
                        transforms.RandomResizedCrop(self.image_size),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                             std=[0.229, 0.224, 0.225])
                    ])
                else:
                    transformations = transforms.Compose([
                        transforms.RandomResizedCrop(self.image_size),
                        transforms.RandomHorizontalFlip(),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                             std=[0.229, 0.224, 0.225])
                    ])
            elif self.split == 'test' or self.split == 'val' or self.split == 'novel':
                transformations = transforms.Compose([
                    transforms.Resize(256),
                    transforms.CenterCrop(self.image_size),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ])
            else:
                raise RuntimeError('Unknown split')
        self.transform = transformations

    def _load_info_from_split_data(self, dataset_name):
        """It loads the image files of the split from the .npy file at
         taxonomy/{dataset_name}/splits_data/filenames_{split}.npy.

         Returns:
             tuple: (self.images, self.labels_to_indices), where:
                 self.images (dict) = {index = {'path': path_to_image,
                                               'label': {lvl: {'class': c, 'class_id': c_id}} } }.
                 self.labels_to_indices (dict) = {lvl: {c_id: [samples]}
        """
        logging.log_info('Loading taxonomy info...')
        self.children, self.parents, self.id_to_name, self.name_to_id, self.depths = \
            self._load_taxonomy_from_textfile(dataset_name)
        self._empty_class_idx = 9999

        split_filenames = load_split_info(self.split, dataset_name)
        images, labels = self._get_hierarchical_gt(split_filenames)

        return images, labels

    def _get_hierarchical_gt(self, split_filenames):
        """Assigns samples to the classes of the split and provides hierarchical ground truth labels. It also builds the
        hierarchical distance matrices *self.dist_mat* and *self.dist_to_lca_mat*.

         Returns:
             tuple: (self.images, self.labels_to_indices), where:
                 self.images (dict) = {index = {'path': path_to_image,
                                               'label': {lvl: {'class': c, 'class_id': c_id}} } }
                 self.labels_to_indices (dict) = {lvl: {c_id: [sample ids]}
         """
        images = dict()
        class_files = {}
        labels = OrderedDict([(lvl, OrderedDict()) for lvl in range(self.num_lvls + 1)])
        # labels = {lvl: {c_id: [samples]}
        ancestors = [{i: 0} for i in range(len(self.id_to_name))]
        descendants = [{i: 0} for i in range(len(self.id_to_name))]
        id_offset = 0
        for c in split_filenames.keys():  # For each leaf class
            c_id = self.name_to_id[c]

            # Get labels of all ancestors
            # level_labels ={lvl: {'class': c, 'class_id': c_id}}
            level_labels = {lvl: {'class': 'Empty', 'class_id': self._empty_class_idx}
                            for lvl in range(self.num_lvls + 1)}  # Initialize all levels empty

            parent = [c_id]  # Leaf class
            while len(parent) == 1:  # Until we have no ancestors
                parent = parent[0]
                if parent in class_files:
                    class_files[parent] += list(range(id_offset, id_offset + len(split_filenames[c])))
                else:
                    class_files[parent] = list(range(id_offset, id_offset + len(split_filenames[c])))

                if parent == c_id:  # Leaf class
                    lvl = 0
                else:
                    lvl = self.num_lvls - self.depths[parent]
                level_labels[lvl] = {'class': self.id_to_name[parent], 'class_id': parent}

                parent = self.parents[parent]
            logging.log_debug('level_labels for ({}) {}: {}'.format(c_id, c, level_labels))

            for i, im in enumerate(split_filenames[c]):
                images[i + id_offset] = {'path': im, 'label': level_labels}

            dist = 0
            for lvl in level_labels.keys():  # Append samples to dict of levels, classes and samples
                if level_labels[lvl]['class_id'] == self._empty_class_idx:
                    continue

                # Update ancestors, descendants info
                ancestors[c_id].update({level_labels[lvl]['class_id']: dist})
                descendants[level_labels[lvl]['class_id']].update({c_id: dist})
                # Previous ancestors
                for (a, d) in ancestors[c_id].items():
                    ancestors[a].update({level_labels[lvl]['class_id']: dist - d})
                    descendants[level_labels[lvl]['class_id']].update({a: dist - d})
                dist += 1

                # Add samples to class
                if level_labels[lvl]['class_id'] in labels[lvl]:
                    labels[lvl][level_labels[lvl]['class_id']] += \
                        list(range(id_offset, id_offset + len(split_filenames[c])))
                else:
                    labels[lvl].update({
                        level_labels[lvl]['class_id']: list(range(id_offset, id_offset + len(split_filenames[c])))})
            id_offset += len(split_filenames[c])

        self.ancestors = ancestors
        self.descendants = descendants
        self.dist_mat, self.dist_to_lca_mat = self._make_dist_mats()   # hierarchical distance matrices
        return images, labels

    def _make_dist_mats(self):
        """Builds hierarchical distance matrices. Code from
        https://github.com/kibok90/cvpr2018-hnd/blob/master/build_taxonomy.py"""
        num_known_classes = len(self.id_to_name)
        MAX_DIST = 127
        dist_mat = MAX_DIST * torch.ones([num_known_classes, num_known_classes], dtype=torch.int8)
        dist_to_lca_mat, dist_to_lca_mat_2 = dist_mat.clone(), dist_mat.clone()
        for i in range(num_known_classes):
            for j in range(num_known_classes):
                dist = dist_to_lca_i = dist_to_lca_j = MAX_DIST
                for common_cid in list(set(self.ancestors[i]).intersection(self.ancestors[j])):
                    new_dist_to_lca_i = self.ancestors[i][common_cid]
                    new_dist_to_lca_j = self.ancestors[j][common_cid]
                    new_dist = new_dist_to_lca_i + new_dist_to_lca_j
                    if dist > new_dist:
                        dist = new_dist
                    if dist_to_lca_i > new_dist_to_lca_i:
                        dist_to_lca_i = new_dist_to_lca_i
                    if dist_to_lca_j > new_dist_to_lca_j:
                        dist_to_lca_j = new_dist_to_lca_j
                dist_mat[i, j] = dist  # sum of dist of i to LCA + dist of j to LCA
                dist_to_lca_mat[i, j] = dist_to_lca_i
                dist_to_lca_mat_2[j, i] = dist_to_lca_j
        assert (dist_mat == dist_mat.T).all()
        assert (dist_to_lca_mat == dist_to_lca_mat_2).all()

        return dist_mat, dist_to_lca_mat

    def _load_taxonomy_from_textfile(self, dataset_name):
        taxonomy = 'taxonomy/{dataset}/taxonomy.txt'.format(dataset=dataset_name)
        classes = open(taxonomy, 'r').read().splitlines()

        novel_classes = open('taxonomy/{}/novel.txt'.format(dataset_name), 'r').read().strip().splitlines()
        known_classes = [c for c in classes if c.strip() not in novel_classes]  # Known leaf classes + parent nodes

        self.root = 0  # Root class id.
        children, parents, depths, self.num_lvls, id_to_name, name_to_id = \
            self._get_parent_child_from_taxonomy(known_classes)

        return children, parents, id_to_name, name_to_id, depths

    def _get_parent_child_from_taxonomy(self, nodes):
        """Assigns indices to classes and gets parent-children relationships from text file information.

        Args:
            nodes (list): List of ordered classes with indentation that indicates their depth in the hierarchy.
        """
        id_to_name = dict()
        name_to_id = dict()
        last_class = self.root
        last_indent_lvl = 0
        children = dict()  # {class_idx : children_list}
        parents = []  # idx is class_id and value is a list of parents (of a single item for the datasets we use)
        depths = [0]  # Depth 0 is for root
        num_lvls = 0
        for c_id, c in enumerate(nodes):
            # Append to children dict
            if c_id not in children.keys():
                children[c_id] = []

            class_name = c.strip()
            id_to_name[c_id] = class_name
            name_to_id[class_name] = c_id

            if c_id == 0:  # root class
                parents.append([])
                continue

            # indentation of 4 spaces denotes a deeper level in the tree
            indent_lvl = int((len(c) - len(class_name)) / 4)
            if indent_lvl > num_lvls:
                num_lvls = indent_lvl
            depths.append(indent_lvl)

            # Get parent-children relationship
            diff = indent_lvl - last_indent_lvl
            if diff == 1:  # loaded class is child of previous one
                parent = last_class
            elif diff == 0:  # loaded class is a sibling
                parent = parents[last_class][0]
            else:  # loaded class is at a higher level wrt previous one
                parent = self._get_parent(parents, last_class, diff)
            children[parent].append(c_id)
            parents.append([parent])

            last_class = c_id
            last_indent_lvl = indent_lvl
        return children, parents, depths, num_lvls, id_to_name, name_to_id

    def _get_parent(self, parents, last_class, diff):
        if diff == -1:
            return parents[parents[last_class][0]][0]
        else:
            return self._get_parent(parents, parents[last_class][0], diff+1)

    def get_image(self, index):
        image = Image.open(self.images[index]['path']).convert(mode="RGB")
        image = self.transform(image)
        return image

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image = self.get_image(index)

        im_data = dict()
        for lvl in self.images[index]['label']:
            im_data[lvl] = {'class': self.images[index]['label'][lvl]['class'],
                            'class_id': torch.as_tensor(self.images[index]['label'][lvl]['class_id'], dtype=torch.int64)}

        return image, im_data


class NovelTextfileHierarchicalDataset(TextfileHierarchicalDataset):
    def _load_taxonomy_from_textfile(self, dataset_name):
        children, parents, id_to_name, name_to_id, depths = super(
            NovelTextfileHierarchicalDataset, self)._load_taxonomy_from_textfile(dataset_name)

        # Assign new ids to novel classes.
        taxonomy = 'taxonomy/{dataset}/taxonomy.txt'.format(dataset=dataset_name)
        classes = open(taxonomy, 'r').read().splitlines()
        known_leaf_classes = open('taxonomy/{}/known.txt'.format(dataset_name), 'r').read().strip().splitlines()
        novel_classes = [c for c in classes if c.strip() not in known_leaf_classes]  # Novel leaf classes + parent nodes
        _, parents_novel, _, _, id_to_name_novel, _ = self._get_parent_child_from_taxonomy(novel_classes)

        # Ids from parents_novel, id_to_name_novel do not match the known class ids. We convert them
        new_id = len(id_to_name)  # First available new class id
        novel_ids = []
        real_parents_novel = {}
        for id, p in enumerate(parents_novel):
            if id == 0:  # Root
                real_parents_novel[0] = []
                continue
            if id_to_name_novel[id] in name_to_id.keys():
                real_id = name_to_id[id_to_name_novel[id]]
            else:  # Not a known class
                real_id = new_id
                logging.log_debug('{} is a novel class. Assigned id {}'.format(id_to_name_novel[id], new_id))
                id_to_name[new_id] = id_to_name_novel[id]
                name_to_id[id_to_name_novel[id]] = new_id
                novel_ids.append(new_id)

                new_id += 1
            real_p = name_to_id[id_to_name_novel[p[0]]]
            real_parents_novel[real_id] = [real_p]

        self.novel_ids = novel_ids
        self.parents_novel = real_parents_novel

        return children, parents, id_to_name, name_to_id, depths

    def _get_hierarchical_gt(self, split_filenames):
        """Assigns samples to the classes of the split and provides hierarchical ground truth labels. """
        images = dict()
        labels = {0: OrderedDict()}
        id_offset = 0
        for c in split_filenames.keys():
            c_id = self.name_to_id[c]
            label = {'class': c,
                     'parent_ids': self.parents_novel[c_id],
                     'novel_id': c_id}

            for i, im in enumerate(split_filenames[c]):
                images[i + id_offset] = {'path': im, 'label': label}

            if label['parent_ids'][0] in labels[0]:
                labels[0][label['parent_ids'][0]] += list(range(id_offset, id_offset + len(split_filenames[c])))
            else:
                labels[0].update({label['parent_ids'][0]: list(range(id_offset, id_offset + len(split_filenames[c])))})
            id_offset += len(split_filenames[c])

        return images, labels

    def __getitem__(self, index):
        image = self.get_image(index)

        im_data = dict()
        im_data[0] = {'class': self.images[index]['label']['class'],
                      'novel_id': self.images[index]['label']['novel_id'],
                      'class_id': torch.as_tensor(self.images[index]['label']['parent_ids'], dtype=torch.int64)}

        return image, im_data
