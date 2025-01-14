from utils import util
from datetime import datetime
import socket
import numpy as np
import torch
import os
import torch.nn.functional as F
import torch.optim as optim
import torchnet
from .loss import SegLoss, EigLoss, PredConLoss
from .tools import Losses, get_scheduler, Logger, Timer
from .util import check_values, extract_prototype
from model.basic_model import MixTrModel
from tqdm import tqdm


class MixTrTrainer(object):
    def __init__(self, args):
        self.args = args
        self.best_pred = 0
        self.date = datetime.now().strftime('%b%d_%H-%M-%S') + '_' + socket.gethostname()
        self.train_loader = util.get_loader(is_train=True, args=args)
        self.val_loader = util.get_loader(is_train=False, args=args)

        self.model = MixTrModel(args=args)
        param_groups = self.model.get_param_groups()
        self.optimizer = optim.AdamW(
            params=[
                {
                    "params": param_groups[0],
                    "lr": args.optimizer.learning_rate,
                    "weight_decay": args.optimizer.weight_decay,
                },
                {
                    "params": param_groups[1],
                    "lr": 0,
                    "weight_decay": 0,
                },
                {
                    "params": param_groups[2],
                    "lr": 10 * args.optimizer.learning_rate,
                    "weight_decay": args.optimizer.weight_decay,
                },
                {
                    "params": param_groups[3],
                    "lr": 10 * args.optimizer.learning_rate,
                    "weight_decay": args.optimizer.weight_decay,
                },
            ],
            lr=args.optimizer.learning_rate,
            weight_decay=args.optimizer.weight_decay,
            betas=args.optimizer.betas,
        )
        self.scheduler = get_scheduler(self.optimizer, args)
        self.criterion_pce = SegLoss(ignore_label=args.dataset.ignore_label)
        self.criterion_con = EigLoss()
        self.criterion_con_ce = PredConLoss()

        self.device = torch.device("cuda:%s" % args.train.gpu_ids[0]
                                   if torch.cuda.is_available() and len(args.train.gpu_ids) > 0
                                   else "cpu")
        print(self.device)
        self.model = self.model.to(self.device)
        self.class_num = args.dataset.num_classes
        self.global_num = args.prototype.global_num
        self.global_prototype = torch.zeros(self.class_num, self.global_num, args.prototype.dim)
        self.global_prototype_list = torch.zeros(self.class_num, dtype=torch.int)
        self.prototype_update_rate = args.prototype.update_rate
        self.use_prototype = False
        self.class_threshold = args.class_threshold
        self.wp = args.loss.without
        self.lp = args.loss.local.pce_local
        self.lc = args.loss.local.con_local
        self.gp = args.loss.global_.pce_global
        self.gcl = args.loss.global_.con_local
        self.gcg = args.loss.global_.con_global

        # define logger file
        logger_path = os.path.join(args.work_dir.ckpt_dir, 'log.txt')

        self.logger = Logger(logger_path)
        self.logger.write_dict_str(args.__dict__)

        # define timer
        self.timer = Timer()
        self.batch_size = args.dataset.batch_size

        #  training log
        self.mean_iou_ind = 0
        self.mean_iou_all = 0
        self.mean_acc_pix = 0

        #  training log local
        self.mean_iou_ind_local = 0
        self.mean_iou_all_local = 0
        self.mean_acc_pix_local = 0

        #  training log global
        self.mean_iou_ind_global = 0
        self.mean_iou_all_global = 0
        self.mean_acc_pix_global = 0

        self.best_val_iou = 0.0
        self.best_epoch_id = 0

        self.best_val_iou_local = 0.0
        self.best_epoch_id_local = 0

        self.best_val_iou_global = 0.0
        self.best_epoch_id_global = 0

        self.epoch_to_start = 0
        self.max_num_epochs = args.train.epoch

        self.global_step = 0
        self.steps_per_epoch = len(self.train_loader)
        self.total_steps = (self.max_num_epochs - self.epoch_to_start) * self.steps_per_epoch

        self.is_training = False
        self.batch_id = 0
        self.epoch_id = 0
        self.checkpoint_dir = args.work_dir.ckpt_dir

        self.losses = Losses()

        # check and create model dir
        if os.path.exists(self.checkpoint_dir) is False:
            os.mkdir(self.checkpoint_dir)

    def load_checkpoint(self, ckpt_name='last_ckpt.pt'):

        if os.path.exists(os.path.join(self.checkpoint_dir, ckpt_name)):
            self.logger.write('loading last checkpoint...\n')
            # load the entire checkpoint
            checkpoint = torch.load(os.path.join(self.checkpoint_dir, ckpt_name),
                                    map_location=self.device)
            # update model states
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

            self.model.to(self.device)

            # update some other states
            self.epoch_to_start = checkpoint['epoch_id'] + 1
            self.best_val_iou = checkpoint['best_val_iou']
            self.best_epoch_id = checkpoint['best_epoch_id']
            self.best_val_iou_local = checkpoint['best_val_iou_local']
            self.best_epoch_id_local = checkpoint['best_epoch_id_local']
            self.best_val_iou_global = checkpoint['best_val_iou_global']
            self.best_epoch_id_global = checkpoint['best_epoch_id_global']

            self.total_steps = (self.max_num_epochs - self.epoch_to_start) * self.steps_per_epoch

            self.logger.write('Epoch_to_start = %d, Historical_best_acc = %.4f (at epoch %d)\n' %
                              (self.epoch_to_start, self.best_val_iou, self.best_epoch_id))

            self.logger.write('Epoch_to_start = %d, (local) Historical_best_acc = %.4f (at epoch %d)\n' %
                              (self.epoch_to_start, self.best_val_iou_local, self.best_epoch_id_local))

            self.logger.write('Epoch_to_start = %d, (global) Historical_best_acc = %.4f (at epoch %d)\n' %
                              (self.epoch_to_start, self.best_val_iou_global, self.best_epoch_id_global))
            self.logger.write('\n')

        else:
            print('training from scratch...')

    def timer_update(self):
        self.global_step = (self.epoch_id - self.epoch_to_start) * self.steps_per_epoch + self.batch_id

        self.timer.update_progress((self.global_step + 1) / self.total_steps)
        est = self.timer.estimated_remaining()
        imps = (self.global_step + 1) * self.batch_size / self.timer.get_stage_elapsed()
        return imps, est

    def save_checkpoint(self, ckpt_name):
        torch.save({
            'epoch_id': self.epoch_id,
            'best_val_iou': self.best_val_iou,
            'best_epoch_id': self.best_epoch_id,
            'best_val_iou_local': self.best_val_iou_local,
            'best_epoch_id_local': self.best_epoch_id_local,
            'best_val_iou_global': self.best_val_iou_global,
            'best_epoch_id_global': self.best_epoch_id_global,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
        }, os.path.join(self.checkpoint_dir, ckpt_name))

    def save_global_prototype(self, prototype_name):
        torch.save(self.global_prototype,
                   os.path.join(self.checkpoint_dir, prototype_name))

    def update_lr_schedulers(self):
        self.scheduler.step()

    def collect_epoch_states(self):
        state = 'Valid'
        self.logger.write('%s: Epoch: [%d,%d]\n' % (state, self.epoch_id + 1, self.max_num_epochs))
        self.logger.write(' * IOU_All {iou}\n'.format(iou=self.mean_iou_all))
        self.logger.write(' * IOU_Ind {iou}\n'.format(iou=self.mean_iou_ind))
        self.logger.write(' * ACC_Pix {acc}\n'.format(acc=self.mean_acc_pix))
        if self.use_prototype:
            self.logger.write(' *  (local) IOU_All {iou}\n'.format(iou=self.mean_iou_all_local))
            self.logger.write(' *  (local) IOU_Ind {iou}\n'.format(iou=self.mean_iou_ind_local))
            self.logger.write(' *  (local) ACC_Pix {acc}\n'.format(acc=self.mean_acc_pix_local))

            self.logger.write(' *  (global) IOU_All {iou}\n'.format(iou=self.mean_iou_all_global))
            self.logger.write(' *  (global) IOU_Ind {iou}\n'.format(iou=self.mean_iou_ind_global))
            self.logger.write(' *  (global) ACC_Pix {acc}\n'.format(acc=self.mean_acc_pix_global))

    def update_checkpoints(self):

        # save current model and prototype
        self.save_checkpoint(ckpt_name='last_ckpt.pt')
        if self.use_prototype:
            self.save_global_prototype(prototype_name='last_global_prototype.pt')

        # update the best model (based on eval acc)
        if self.mean_iou_all > self.best_val_iou:
            self.best_val_iou = self.mean_iou_all
            self.best_epoch_id = self.epoch_id
            self.save_checkpoint(ckpt_name='best_ckpt.pt')
            if self.use_prototype:
                self.save_global_prototype(prototype_name='best_global_prototype.pt')
                self.logger.write('*' * 10 + 'Best prototype updated!' + '*' * 10 + '\n')
            self.logger.write('*' * 10 + 'Best model updated!' + '*' * 10 + '\n')
            self.logger.write('\n')
        if self.use_prototype:
            # update the best model (based on eval acc)
            if self.mean_iou_all_local > self.best_val_iou_local:
                self.best_val_iou_local = self.mean_iou_all_local
                self.best_epoch_id_local = self.epoch_id
                self.save_checkpoint(ckpt_name='best_ckpt_local.pt')
                if self.use_prototype:
                    self.save_global_prototype(prototype_name='best_global_prototype_local.pt')
                    self.logger.write('*' * 10 + 'Best prototype updated(local)!' + '*' * 10 + '\n')
                self.logger.write('*' * 10 + 'Best model updated(local)!' + '*' * 10 + '\n')
                self.logger.write('\n')

            # update the best model (based on eval acc)
            if self.mean_iou_all_global > self.best_val_iou_global:
                self.best_val_iou_global = self.mean_iou_all_global
                self.best_epoch_id_global = self.epoch_id
                self.save_checkpoint(ckpt_name='best_ckpt_global.pt')
                if self.use_prototype:
                    self.save_global_prototype(prototype_name='best_global_prototype_global.pt')
                    self.logger.write('*' * 10 + 'Best prototype updated(global)!' + '*' * 10 + '\n')
                self.logger.write('*' * 10 + 'Best model updated(global)!' + '*' * 10 + '\n')
                self.logger.write('\n')
        self.logger.write('(without) Latest model updated. Epoch_IoU=%.4f, Historical_best_IOU=%.4f (at epoch %d)\n'
                          % (self.mean_iou_all, self.best_val_iou, self.best_epoch_id + 1))
        if self.use_prototype:
            self.logger.write('(local) Latest model updated. Epoch_IoU=%.4f, Historical_best_IOU=%.4f (at epoch %d)\n'
                              % (self.mean_iou_all_local, self.best_val_iou_local, self.best_epoch_id_local + 1))
            self.logger.write('(global) Latest model updated. Epoch_IoU=%.4f, Historical_best_IOU=%.4f (at epoch %d)\n'
                              % (self.mean_iou_all_global, self.best_val_iou_global, self.best_epoch_id_global + 1))
        self.logger.write('\n')

    def update_prototypes(self, prototype, labels, infer=False):
        n = prototype.size()[0]

        if infer:
            indices = labels
        else:
            indices = [torch.nonzero(row).flatten().tolist() for row in labels]

        for i in range(n):
            for j in indices[i]:
                if self.global_prototype_list[j] < self.global_num:
                    self.global_prototype[j][self.global_prototype_list[j]] = prototype[i][j]
                    self.global_prototype_list[j] += 1
                else:
                    similarity = -1
                    k_ = 0
                    for k in range(self.global_num):

                        cur_similarity = F.cosine_similarity(prototype[i][j], self.global_prototype[j][k], dim=0)
                        if cur_similarity > similarity:
                            similarity = cur_similarity
                            k_ = k
                    self.global_prototype[j][k_] = self.global_prototype[j][k_] * self.prototype_update_rate + (
                            prototype[i][j] * (1 - self.prototype_update_rate))

    def train(self):
        # Iterate over data.
        self.logger.write('lr: %0.7f\n' % self.optimizer.param_groups[0]['lr'])
        if self.epoch_id >= self.args.prototype.use_epoch:
            self.use_prototype = True
        else:
            self.use_prototype = False
        tbar = tqdm(self.train_loader)
        for self.batch_id, batch in enumerate(tbar):

            img, label, class_label, image_path = batch

            input_size = img.size()[2:4]

            img_ = img.to(self.device)
            label_ = label.to(self.device)
            class_label_ = class_label.to(self.device)

            class_labels = torch.cat((torch.ones((class_label_.size(0), 1)).to(self.device), class_label_), dim=1)

            feat, pred = self.model(img_)

            pred_ = F.interpolate(pred, size=input_size, mode='bilinear', align_corners=True)
            loss_pce = self.criterion_pce(pred_, label_)
            if self.use_prototype:

                with torch.no_grad():
                    cur_prototype = extract_prototype(feat, pred, class_labels, False)
                    self.update_prototypes(cur_prototype, class_labels, False)
                use_global_prototype = check_values(self.global_prototype_list, self.global_num)
                feat_aug_local, pred_aug_local = self.model(feat, cur_prototype.to(self.device),
                                                            class_labels, False, False)

                loss_con_local = self.criterion_con_ce(pred, pred_aug_local)
                pred_aug_local_ = F.interpolate(pred_aug_local, size=input_size, mode='bilinear', align_corners=True)
                loss_pce_local = self.criterion_pce(pred_aug_local_, label_)
                if use_global_prototype:
                    feat_aug_global, pred_aug_global = self.model(feat, self.global_prototype.to(self.device),
                                                                  class_labels, True, False)

                    loss_con_global = self.criterion_con_ce(pred, pred_aug_global)
                    pred_aug_global_ = F.interpolate(pred_aug_global, size=input_size, mode='bilinear',
                                                     align_corners=True)
                    loss_pce_global = self.criterion_pce(pred_aug_global_, label_)

                    loss = loss_pce_global * self.gp + loss_con_local * self.gcl + loss_con_global * self.gcg
                else:
                    loss = loss_pce_local * self.lp + loss_con_local * self.lc
            else:
                loss = loss_pce * self.wp
            self.losses.update(loss.item(), img.size(0))
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            tbar.set_description(
                'Train[{0}] Loss{loss.val:.3f} {loss.avg:.3f} '.format(
                    self.epoch_id + 1, loss=self.losses))
            self.timer_update()

    def val(self):
        confusion_meter = torchnet.meter.ConfusionMeter(self.class_num, normalized=False)
        confusion_meter_local = torchnet.meter.ConfusionMeter(self.class_num, normalized=False)
        confusion_meter_global = torchnet.meter.ConfusionMeter(self.class_num, normalized=False)
        self.model.eval()
        with torch.no_grad():
            tbar = tqdm(self.val_loader)
            for self.batch_id, batch in enumerate(tbar):
                img, label, class_label, image_path = batch
                input_size = img.size()[2:4]

                img_ = img.to(self.device)
                label_ = label.to(self.device)
                class_label_ = class_label.to(self.device)

                class_labels = torch.cat((torch.ones((class_label_.size(0), 1)).to(self.device), class_label_), dim=1)

                feat, pred = self.model(img_)

                pred_ = F.interpolate(pred, size=input_size, mode='bilinear', align_corners=True)
                loss_pce = self.criterion_pce(pred_, label_)

                valid_pixel = label.ne(255)
                pred_label = torch.max(pred_, 1, keepdim=True)[1]
                pred_label = torch.squeeze(pred_label, 1)

                confusion_meter.add(pred_label[valid_pixel], label[valid_pixel])

                if self.use_prototype:
                    cur_prototype = extract_prototype(feat, pred, class_labels, False)

                    feat_aug_local, pred_aug_local = self.model(feat, cur_prototype.to(self.device), class_labels,
                                                                False, False)
                    loss_con_local = self.criterion_con_ce(pred, pred_aug_local)

                    pred_aug_local = F.interpolate(pred_aug_local, size=input_size, mode='bilinear', align_corners=True)
                    loss_pce_local = self.criterion_pce(pred_aug_local, label_)

                    pred_label_local = torch.max(pred_aug_local, 1, keepdim=True)[1]
                    pred_label_local = torch.squeeze(pred_label_local, 1)
                    confusion_meter_local.add(pred_label_local[valid_pixel], label[valid_pixel])

                    feat_aug_global, pred_aug_global = self.model(feat, self.global_prototype.to(self.device),
                                                                  class_labels, True, False)
                    loss_con_global = self.criterion_con_ce(pred, pred_aug_global)
                    pred_aug_global = F.interpolate(pred_aug_global, size=input_size, mode='bilinear',
                                                    align_corners=True)
                    loss_pce_global = self.criterion_pce(pred_aug_global, label_)
                    # pred with global prototype
                    pred_label_global = torch.max(pred_aug_global, 1, keepdim=True)[1]
                    pred_label_global = torch.squeeze(pred_label_global, 1)
                    confusion_meter_global.add(pred_label_global[valid_pixel], label[valid_pixel])

                    loss = loss_pce_global * self.gp + loss_con_local * self.gcl + loss_con_global * self.gcg

                else:
                    loss = loss_pce * self.wp

                self.losses.update(loss.item(), img.size(0))
                tbar.set_description(
                    'Valid[{0}] Loss{loss.val:.3f} {loss.avg:.3f} '.format(
                        self.epoch_id + 1, loss=self.losses))

            # pred without
            confusion_matrix = confusion_meter.value()
            inter = np.diag(confusion_matrix)
            union = confusion_matrix.sum(1).clip(min=1e-12) + confusion_matrix.sum(0).clip(min=1e-12) - inter

            self.mean_iou_ind = inter / union
            self.mean_iou_all = self.mean_iou_ind.mean()
            self.mean_acc_pix = float(inter.sum()) / float(confusion_matrix.sum())

            if self.use_prototype:
                #  pred with local prototype
                confusion_matrix_local = confusion_meter_local.value()
                inter_local = np.diag(confusion_matrix_local)
                union_local = confusion_matrix_local.sum(1).clip(min=1e-12) + \
                              confusion_matrix_local.sum(0).clip(min=1e-12) - inter_local

                self.mean_iou_ind_local = inter_local / union_local
                self.mean_iou_all_local = self.mean_iou_ind_local.mean()
                self.mean_acc_pix_local = float(inter_local.sum()) / float(confusion_matrix_local.sum())

                # pred with global prototype
                confusion_matrix_global = confusion_meter_global.value()
                inter_global = np.diag(confusion_matrix_global)
                union_global = confusion_matrix_global.sum(1).clip(min=1e-12) + \
                               confusion_matrix_global.sum(0).clip(min=1e-12) - inter_global

                self.mean_iou_ind_global = inter_global / union_global
                self.mean_iou_all_global = self.mean_iou_ind_global.mean()
                self.mean_acc_pix_global = float(inter_global.sum()) / float(confusion_matrix_global.sum())

    def train_model(self):
        self.load_checkpoint()

        for self.epoch_id in range(self.epoch_to_start, self.max_num_epochs):
            self.is_training = False
            self.model.train()
            self.train()
            self.update_lr_schedulers()
            self.val()
            self.collect_epoch_states()
            self.update_checkpoints()
