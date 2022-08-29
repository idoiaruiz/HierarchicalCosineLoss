import torch

from torch.utils.tensorboard.summary import hparams
from torch.utils.tensorboard import SummaryWriter


class CustomWriter(SummaryWriter):
    def add_hparams(self, hparam_dict, metric_dict, epoch=None):
        """Modified method that does not create a subdirectory. Not needed in newer versions of PyTorch which include
        an attribute to specify the path"""
        torch._C._log_api_usage_once("tensorboard.logging.add_hparams")
        if type(hparam_dict) is not dict or type(metric_dict) is not dict:
            raise TypeError('hparam_dict and metric_dict should be dictionary.')
        exp, ssi, sei = hparams(hparam_dict, metric_dict)

        logdir = self._get_file_writer().get_logdir()
        with SummaryWriter(log_dir=logdir) as w_hp:
            w_hp.file_writer.add_summary(exp)
            w_hp.file_writer.add_summary(ssi)
            w_hp.file_writer.add_summary(sei)
            for k, v in metric_dict.items():
                w_hp.add_scalar(k, v, global_step=epoch)


def write_params(writer, config, metrics, epoch=None, split='test'):
    if writer:
        params = {}
        for s in config.sections():
            params.update({'{}-{}'.format(s, k): config[s][k] for k in config[s].keys()})
        writer.add_hparams(params, {'{}/{}'.format(split, m): metrics[m] for m in metrics}, epoch=epoch)


def write_metrics(writer, metric_dict, epoch=None, split='test'):
    if writer:
        for k, v in metric_dict.items():
            writer.add_scalar('{}/{}'.format(split, k), v, global_step=epoch)
