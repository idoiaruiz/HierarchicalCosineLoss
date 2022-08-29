import itertools
import os
from PIL import Image
from torchvision.transforms import ToTensor

import numpy as np
import matplotlib
matplotlib.use('agg')

import matplotlib.pyplot as plt


def plot_confusion_matrix(cm, class_names_true, class_names_pred):
    """
    Args:
        cm (array, shape = [n, m]): a confusion matrix of integer classes
        class_names_true (array, shape = [m]): String names of the integer classes
        class_names_pred (array, shape = [n]): String names of the integer classes


    Returns:
        A matplotlib figure containing the plotted confusion matrix.
    """
    # Compute the labels from the normalized confusion matrix.
    labels = np.around(cm.astype('float') / (cm.sum(axis=0)[np.newaxis,:] + 1e-5), decimals=2)

    figure = plt.figure(figsize=(50, 50))
    plt.imshow(labels, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("Confusion matrix")
    plt.colorbar()
    plt.xticks(np.arange(len(class_names_true)), class_names_true, rotation=75)
    plt.yticks(np.arange(len(class_names_pred)), class_names_pred)

    # Use white text if squares are dark; otherwise black.
    threshold = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        color = "white" if cm[i, j] > threshold else "black"
        plt.text(j, i, labels[i, j], horizontalalignment="center", color=color)

    plt.tight_layout()
    plt.ylabel('Predicted label')
    plt.xlabel('True label')
    return figure


def plot_to_image(figure, name='cm', exp_dir='.'):
    im_path = os.path.join(exp_dir, name+'.png')
    plt.savefig(im_path, format='png')
    plt.close(figure)
    pil_im = Image.open(im_path).convert(mode="RGB")
    image = (ToTensor()(pil_im))

    return image


def plot_auc(x, y, auc_value, xlab, ylab, path_save, title=''):
    plot_xy(x, y, xlab, ylab, path_save, title=title, label='area = %0.3f' % auc_value)


def plot_xy(x, y, xlab, ylab, path_save, title='', label=None):
    plt.figure()
    plt.plot(x, y, color='darkorange', lw=2, label=label)
    plt.xlabel(xlab)
    plt.ylabel(ylab)
    plt.title(title)
    plt.legend(loc="lower right")
    plt.grid(True)

    plt.savefig(os.path.join(path_save, title+'.png'))
    plt.savefig(os.path.join(path_save, title+'.eps'), format='eps', dpi=1000)


def plot_dists_over_acc(x, y1, y2, xlab, ylab, path_save, y1_lab='', y2_lab='', title=''):
    plt.figure()
    plt.plot(x, y1, lw=2, label=y1_lab)
    plt.plot(x, y2, lw=2, label=y2_lab)
    plt.xlabel(xlab)
    plt.ylabel(ylab)
    plt.title(title)
    plt.legend(loc="lower right")
    plt.grid(True)

    plt.savefig(os.path.join(path_save, title+'.png'))
    plt.savefig(os.path.join(path_save, title+'.eps'), format='eps', dpi=1000)

