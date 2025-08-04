import time
import torch
import numpy as np
from argparse import ArgumentParser

from .incremental_learning import Inc_Learning_Appr
from datasets.exemplars_dataset import ExemplarsDataset
from networks.sow_libs_V3 import SOW_V3

class Appr(Inc_Learning_Appr):
    """Class implementing the finetuning baseline"""

    def __init__(self, model, device, nepochs=100, lr=0.05, lr_min=1e-4, lr_factor=3, lr_patience=5,
                 clipgrad=10000, momentum=0, wd=0, sow_lr=0.05, sow_mo=0.9,
                 multi_softmax=False, wu_nepochs=0, wu_lr_factor=1,
                 fix_bn=False, eval_on_train=False, logger=None, exemplars_dataset=None,
                 all_outputs=False, loss_margin=0.005, acc_margin=0.005):
        super(Appr, self).__init__(model, device, nepochs, lr, lr_min, lr_factor, lr_patience,
                                   clipgrad, momentum, wd, multi_softmax, wu_nepochs, wu_lr_factor,
                                   fix_bn, eval_on_train, logger, exemplars_dataset)
        self.all_out = all_outputs
        self.sow_lr = sow_lr
        self.sow_mo = sow_mo
        self.loss_margin = loss_margin
        self.acc_margin = acc_margin
        self.epoch_count = 0

    @staticmethod
    def exemplars_dataset_class():
        return ExemplarsDataset

    @staticmethod
    def extra_parser(args):
        """Returns a parser containing the approach specific parameters"""
        parser = ArgumentParser()
        parser.add_argument('--all-outputs', action='store_true', required=False,
                            help='Allow all weights related to all outputs to be modified (default=%(default)s)')
        parser.add_argument('--sow-lr', type=float, default=0.05,
                            help='SOW用学習率 (default=%(default)s)')
        parser.add_argument('--sow-mo', type=float, default=0.9,
                            help='SOW用SGDのmomentum (default=%(default)s)')
        parser.add_argument('--loss-margin', type=float, default=0.005)
        parser.add_argument('--acc-margin', type=float, default=0.005)
        return parser.parse_known_args(args)

    def _get_optimizer(self):
        """Returns the optimizer"""
        Non_SOW_params = {'params': [], 'lr': self.lr, 'momentum': self.momentum, 'weight_decay': self.wd}
        SOW_params = {'params': [], 'lr': self.sow_lr, 'momentum': self.sow_mo, 'weight_decay': 0}
        #print(f'len(exemplars_dataset)={len(self.exemplars_dataset)}, len(heads)={len(self.model.heads)}, all_out={self.all_out}')
        if len(self.exemplars_dataset) == 0 and len(self.model.heads) > 1 and not self.all_out:
            # if there are no exemplars, previous heads are not modified
            #params = list(self.model.model.parameters()) + list(self.model.heads[-1].parameters())
            for name1, m in self.model.model.named_modules():
                if isinstance(m, SOW_V3) :
                    for name2, prm in m.named_parameters():
                        SOW_params['params'].append(prm)
                        #print('  * ', name1, '\t', name2, '\t', prm.shape)
                else:
                    if len(list(m.modules())) == 1 :
                        for name2, prm in m.named_parameters():
                            Non_SOW_params['params'].append(prm)
                            #print('    ', name1, '\t', name2, '\t', prm.shape)
            for name1, m in self.model.heads[-1].named_modules():
                if isinstance(m, SOW_V3) :
                    for name2, prm in m.named_parameters():
                        SOW_params['params'].append(prm)
                        #print('  * ', name1, '\t', name2, '\t', prm.shape)
                else:
                    if len(list(m.modules())) == 1 :
                        for name2, prm in m.named_parameters():
                            Non_SOW_params['params'].append(prm)
                            #print('    ', name1, '\t', name2, '\t', prm.shape)
        else:
            for name1, m in self.model.named_modules():
                if isinstance(m, SOW_V3) :
                    for name2, prm in m.named_parameters():
                        SOW_params['params'].append(prm)
                        #print('  * ', name1, '\t', name2, '\t', prm.shape)
                else:
                    if len(list(m.modules())) == 1 :
                        for name2, prm in m.named_parameters():
                            Non_SOW_params['params'].append(prm)
                            #print('    ', name1, '\t', name2, '\t', prm.shape)
        param_groups = [Non_SOW_params, SOW_params]
        #for i in range(len(param_groups)): print(' ', len(param_groups[i]['params']))
        return torch.optim.SGD(param_groups)

    def check_optimizer(self):
        print(self.optimizer)
        for n, pg in enumerate(self.optimizer.param_groups) :
            print(f' Group[{n}]  param_num=%d, free_param_num=%d'
                  % (len(pg['params']), sum([prm.requires_grad for prm in pg['params']])))
        return

    def check_rank_of_SOW(self, sow):
        rank = sow.rank
        freezed_rank = sow.get_freezed_rank()
        print(f'sow: rank={rank}, freezed_rank={freezed_rank}')
        if freezed_rank == 0 :
            print(f' a:       ', [f"{num:+.2e}" for num in sow.a[0:2].tolist()],
                  '....', [f"{num:+.2e}" for num in sow.a[-2:].tolist()])
        else :
            print(f' a:       ', [f"{num:+.2e}" for num in sow.a[0:2].tolist()],
                  '....', [f"{num:+.2e}" for num in sow.a[freezed_rank-1:freezed_rank+1].tolist()],
                  '....', [f"{num:+.2e}" for num in sow.a[-2:].tolist()])
        if sow.a_backup != None :
            print(f' a_backup:', [f"{num:+.2e}" for num in sow.a_backup[0:2].tolist()],
                  '....', [f"{num:+.2e}" for num in sow.a_backup[freezed_rank-1:freezed_rank+1].tolist()],
                  '....', [f"{num:+.2e}" for num in sow.a_backup[-2:].tolist()])
        return

    def train(self, t, trn_loader, val_loader):
        """Main train structure"""
        super().pre_train_process(t, trn_loader)
        self.train_loop(t, trn_loader, val_loader, known_best_loss=None)
        self.post_train_process(t, trn_loader, val_loader)
        return

    def train_loop(self, t, trn_loader, val_loader, known_best_loss=None):
        """Contains the epochs loop"""
        lr = self.lr
        sow_lr = self.sow_lr
        if known_best_loss == None : best_loss = np.inf
        else: best_loss = known_best_loss
        patience = self.lr_patience
        best_model = self.model.get_copy()

        self.optimizer = self._get_optimizer()

        '''# Debug-1 (Main)
        self.check_optimizer()
        for sow in self.model.model.ListCustomModules() :
            self.check_rank_of_SOW(sow)
        '''

        # Loop epochs
        for e in range(self.nepochs):
            # Train
            clock0 = time.time()
            success = self.train_epoch(t, trn_loader)
            clock1 = time.time()
            if self.eval_on_train:
                train_loss, train_acc, _ = self.eval(t, trn_loader)
                clock2 = time.time()
                print('| Epoch {:3d}, lr={:7.1e} {:7.1e} time={:5.1f}s/{:5.1f}s | Train: loss={:.3f}, TAw acc={:5.1f}% |'.format(
                    e + 1, lr, sow_lr, clock1 - clock0, clock2 - clock1, train_loss, 100 * train_acc), end='')
                self.logger.log_scalar(task=t, iter=e + 1, name="loss", value=train_loss, group="train")
                self.logger.log_scalar(task=t, iter=e + 1, name="acc", value=100 * train_acc, group="train")
            else:
                print('| Epoch {:3d}, lr={:7.1e} time={:5.1f}s | Train: skip eval |'.format(e + 1, lr, clock1 - clock0), end='')

            # Valid
            clock3 = time.time()
            valid_loss, valid_acc, _ = self.eval(t, val_loader)
            clock4 = time.time()
            print(' Valid: time={:5.1f}s loss={:.3f}, TAw acc={:5.1f}% |'.format(
                clock4 - clock3, valid_loss, 100 * valid_acc), end='')
            self.logger.log_scalar(task=t, iter=e + 1, name="loss", value=valid_loss, group="valid")
            self.logger.log_scalar(task=t, iter=e + 1, name="acc", value=100 * valid_acc, group="valid")

            # Adapt learning rate - patience scheme - early stopping regularization
            if valid_loss < best_loss:
                # if the loss goes down, keep it as the best model and end line with a star ( * )
                best_loss = valid_loss
                best_model = self.model.get_copy()
                patience = self.lr_patience
                print(' *', end='')
            else:
                # if the loss does not go down, decrease patience
                patience -= 1
                if not success : patience = 0	# if train_epoch() returns False, we may reduce leraning rate (lr)
                if patience <= 0:
                    # if it runs out of patience, reduce the learning rate
                    lr /= self.lr_factor
                    sow_lr /= self.lr_factor
                    print(' lr={:.1e}'.format(lr), end='')
                    if lr < self.lr_min:
                        # if the lr decreases below minimum, stop the training session
                        print()
                        break
                    # reset patience and recover best model so far to continue training
                    patience = self.lr_patience
                    self.optimizer.param_groups[0]['lr'] = lr
                    self.optimizer.param_groups[1]['lr'] = sow_lr
                    self.model.set_state_dict(best_model)
            self.logger.log_scalar(task=t, iter=e + 1, name="patience", value=patience, group="train")
            self.logger.log_scalar(task=t, iter=e + 1, name="lr", value=lr, group="train")
            print()
            #print(f' best_loss=%.3f, valid_loss=%.3f'%(best_loss, valid_loss))
        self.model.set_state_dict(best_model)
        return

    def post_train_process(self, t, trn_loader, val_loader):
        for sow in self.model.model.ListCustomModules() :
            sow.backup_full_rank_a()
            best_r = sow.max_rank
            sow.save_low_rank_a(t)	# 最善の低ランク a をタスク毎に保管
            sow.save_rank(t, best_r)		# タスク固有のランクを保存
            sow.update_freezed_rank(best_r)	# 必要なら固定ランクの更新
            print(f'post_train_process: rank={sow.rank}, freezed={sow.freezed_rank}')
            print('low_rank_a:', sow.low_rank_a)
            print('ranks:', sow.ranks)
            print('a:', sow.a)
            print('a_backup:', sow.a_backup)
        super().post_train_process(t, trn_loader)
        return

    def train_epoch(self, t, trn_loader):
        """Runs a single epoch"""
        self.model.train()	# Set Trainning mode to LLL_Net
        '''
        print(F'train_epoch: {type(self)}')
        print(F' features: training={self.model.model.features.training}, fix_features={self.model.model.fix_features}')
        print(F' classifier: training={self.model.model.classifier.training}, fix_classifier={self.model.model.fix_classifier}')
        print(F' heads: training={self.model.heads.training}')
        '''
        if self.fix_bn:
            self.model.freeze_bn()
        step = 0
        success = True
        for images, targets in trn_loader:
            # Forward current model
            outputs = self.model(images.to(self.device))
            loss = self.criterion(t, outputs, targets.to(self.device))
            if torch.isnan(loss) : success = False
            if loss > 1.0e+3 : success = False
            # Backward
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clipgrad)
            self.optimizer.step()
            if not success : break
            #
            if step < 16 :
                sow = self.model.model.classifier[1]
                sz1 = len(sow.tan_2phi)
                sz2 = len(sow.a)
                sz3 = len(sow.tan_2theta)
                print('csv,%d,%d,%.4e' %(self.epoch_count, step, loss), end='')
                for i in range(0,3) : print(',%.4e' % (sow.tan_2phi[i]), end='')
                for i in range(sz1-3, sz1) : print(',%.4e' % (sow.tan_2phi[i]), end='')
                for i in range(0,3) : print(',%.4e' % (sow.a[i]), end='')
                for i in range(sz2-3, sz2) : print(',%.4e' % (sow.a[i]), end='')
                for i in range(0,3) : print(',%.4e' % (sow.tan_2theta[i]), end='')
                for i in range(sz3-3, sz3) : print(',%.4e' % (sow.tan_2theta[i]), end='')
                print()
            #
            step += 1
        self.epoch_count += 1
        # epoch毎に adjist_SOW() を呼び出す
        '''
        ## 直交化処理(adjust)前の精度評価
        train_loss, train_acc, _ = self.eval(t, trn_loader)
        print(f'[Before] train_loss=%.3f, train_acc=%.3f'%(train_loss, train_acc))
        self.model.train()
        if self.fix_bn: self.model.freeze_bn()
        ##
        '''
        current_lr = self.optimizer.param_groups[0]['lr']
        '''
        for m in self.model.model.ListCustomModules():
            # 収束状況を観測するため epoch 毎に計測
            #m.check_quality()
        ## 直交化処理(adjust)後の精度評価
        trn_loss, trn_acc, _ = self.eval(t, trn_loader)
        print(f'[After] trn_loss=%.3f, trn_acc=%.3f'%(trn_loss, trn_acc))
        self.model.train()
        if self.fix_bn: self.model.freeze_bn()
        ##
        '''
        # non発生の場合，epoch は失敗
        return success

    def criterion(self, t, outputs, targets):
        """Returns the loss value"""
        if self.all_out or len(self.exemplars_dataset) > 0:
            return torch.nn.functional.cross_entropy(torch.cat(outputs, dim=1), targets)
        return torch.nn.functional.cross_entropy(outputs[t], targets - self.model.task_offset[t])
