import os
import glob
import shutil
import torch

from collections import OrderedDict


class Saver(object):

    def __init__(self, args):
        self.args = args
        self.dir = os.path.join('mypath', args.dataset, args.checkname)
        self.runs = sorted(glob.glob(os.path.join(self.dir, 'experiment_*')))
        run_id = int(self.runs[-1].split('_')[-1]) + 1 if self.runs else 0

        self.experiment_dir = os.path.join(self.dir, f'experiment_{str(run_id)}')

        if not os.path.exists(self.dir):
            os.mkdir(self.dir)

        if not os.path.exists(self.experiment_dir):
            os.mkdir(self.experiment_dir)

    def save_checkpoint(self, state, is_best, filename='checkpoint.pth'):
        """Saves checkpoint to disk"""

        if is_best:
            epoch = state['epoch']

            if "val_best_pred" in list(state.keys()):
                filename = os.path.join(self.experiment_dir, filename)
                torch.save(state, filename)

                val_best_pred = state["val_best_pred"]

                with open(os.path.join(self.experiment_dir, "val_best_pred.txt"), "w") as f:
                    f.write(str(str(epoch) + "_" + str(val_best_pred)))

                if self.runs:
                    previous_miou = [0.0]

                    for run in self.runs:
                        run_id = run.split("_")[-1]
                        path = os.path.join(self.dir, f"experiment_{str(run_id)}", "val_best_pred.txt")

                        if os.path.exists(path):
                            with open(path, "r") as f:
                                miou = str(f.readline())
                                previous_miou.append(float(miou.split('_')[-1]))

                        else:
                            continue

                    max_miou = max(previous_miou)

                    if val_best_pred > max_miou:
                        shutil.copyfile(filename, os.path.join(self.experiment_dir, 'val_model_best.pth'))

    def save_experiment_config(self):
        logfile = os.path.join(self.experiment_dir, 'arg_parameters.txt')
        log_file = open(logfile, 'w')

        p = OrderedDict()

        if self.args.models == 'DeepLab':
            p['dataset'] = self.args.dataset
            p['input_channel'] = self.args.input_channel
            p['backbone'] = self.args.backbone
            p['out_stride'] = self.args.out_stride
            p['lr'] = self.args.lr
            p['weight_decay'] = self.args.weight_decay
            p['momentum'] = self.args.momentum
            p['nesterov'] = self.args.nesterov
            p['lr_scheduler'] = self.args.lr_scheduler
            p['loss_type'] = self.args.loss_type
            p['epoch'] = self.args.epochs
            p['base_size'] = self.args.base_size
            p['crop_size'] = self.args.crop_size

            for key, val in p.items():
                log_file.write(key + ':' + str(val) + '\n')

            log_file.close()

        elif self.args.models == 'SegNet':
            p['dataset'] = self.args.dataset
            p['input_channel'] = self.args.input_channel
            p['lr'] = self.args.lr
            p['loss_type'] = self.args.loss_type
            p['epoch'] = self.args.epochs
            p['base_size'] = self.args.base_size
            p['momentum'] = self.args.momentum

            for key, val in p.items():
                log_file.write(key + ':' + str(val) + '\n')

            log_file.close()

        elif self.args.models == 'UNet':
            p['dataset'] = self.args.dataset
            p['input_channel'] = self.args.input_channel
            p['lr'] = self.args.lr
            p['loss_type'] = self.args.loss_type
            p['epoch'] = self.args.epochs
            p['base_size'] = self.args.base_size
            p['momentum'] = self.args.momentum

            for key, val in p.items():
                log_file.write(key + ':' + str(val) + '\n')

            log_file.close()




