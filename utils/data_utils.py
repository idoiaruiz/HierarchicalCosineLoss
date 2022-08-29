import numpy as np


def load_split_info(split_name, dataset):
    """

    Args:
        split_name (string): Split from which to get its filenames: train, test, val or novel
        dataset (string): Dataset name: AWA2, CUB, tsinghua, mtsd

    Returns:
        dict: Split info in dict with classes as keys and filenames as values.

    """
    split_filenames = np.load('taxonomy/{}/splits_data/filenames_{}.npy'.format(dataset, split_name),
                              allow_pickle=True).item()
    return split_filenames

