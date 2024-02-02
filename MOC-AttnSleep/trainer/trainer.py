import numpy as np
import torch
from base import BaseTrainer
from utils import inf_loop, MetricTracker,load_folds_data
import torch.nn as nn


selected_d = {"outs": [], "trg": []}
class Trainer(BaseTrainer):
    """
    Trainer class
    """
    def __init__(self, model, criterion, metric_ftns, optimizer, config, data_loader, fold_id,
                 valid_data_loader=None,full_test_loader=None,full_train_loader=None, class_weights=None):
        super().__init__(model, criterion, metric_ftns, optimizer, config, fold_id)
        self.config = config
        self.data_loader = data_loader
        self.len_epoch = len(self.data_loader)

        self.valid_data_loader = valid_data_loader
        self.full_test_loader = full_test_loader
        self.full_train_loader=full_train_loader
        self.do_validation = self.valid_data_loader is not None
        self.lr_scheduler = optimizer
        self.log_step = int(data_loader.batch_size) * 1  # reduce this if you want more logs

        self.train_metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_ftns])
        self.valid_metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_ftns])

        self.fold_id = fold_id
        self.selected = 0
        self.class_weights = class_weights

    def _train_epoch(self, epoch, total_epochs):
        """
        Training logic for an epoch

        :param epoch: Integer, current training epoch.
               total_epochs: Integer, the total number of epoch
        :return: A log that contains average loss and metric in this epoch.
        """
        self.model.train()
        self.train_metrics.reset()
        overall_outs = []
        overall_trgs = []
        for batch_idx, (data, target) in enumerate(self.data_loader):
            data, target = data.to(self.device), target.to(self.device)

            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target, self.class_weights, self.device)

            loss.backward()
            self.optimizer.step()

            self.train_metrics.update('loss', loss.item())
            for met in self.metric_ftns:
                self.train_metrics.update(met.__name__, met(output, target))

            if batch_idx % self.log_step == 0:
                self.logger.debug('Train Epoch: {} {} Loss: {:.6f} '.format(
                    epoch,
                    self._progress(batch_idx),
                    loss.item(),
                ))

            if batch_idx == self.len_epoch:
                break
        log = self.train_metrics.result()

        if self.do_validation:
            val_log, outs, trgs = self._valid_epoch(epoch)
            log.update(**{'val_' + k: v for k, v in val_log.items()})
            if val_log["accuracy"] > self.selected:
                print(f"Updating result {epoch}")
                self.selected = val_log["accuracy"]
                selected_d["outs"] = outs
                selected_d["trg"] = trgs
                self.full_pred(epoch)
                self.train_saver(epoch)
            if epoch == total_epochs:
                overall_outs.extend(selected_d["outs"])
                overall_trgs.extend(selected_d["trg"])
                # self.train_saver()

            # THIS part is to reduce the learning rate after 10 epochs to 1e-4
            if epoch == 10:
                for g in self.lr_scheduler.param_groups:
                    g['lr'] = 0.0001

        return log, overall_outs, overall_trgs

    def _valid_epoch(self, epoch):
        """
        Validate after training an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains information about validation
        """
        self.model.eval()
        self.valid_metrics.reset()
        with torch.no_grad():
            outs = np.array([])
            trgs = np.array([])
            outs_prob = np.array([])
            for batch_idx, (data, target) in enumerate(self.valid_data_loader):
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                output_tensor = torch.nn.functional.softmax(output,dim=1)
                outs_prob = np.append(outs_prob,output_tensor.cpu().numpy())
                loss = self.criterion(output, target, self.class_weights, self.device)
                # print(output.shape)
                self.valid_metrics.update('loss', loss.item())
                for met in self.metric_ftns:
                    self.valid_metrics.update(met.__name__, met(output, target))

                preds_ = output.data.max(1, keepdim=True)[1].cpu()

                outs = np.append(outs, preds_.cpu().numpy())
                trgs = np.append(trgs, target.data.cpu().numpy())
        outs_prob=outs_prob.reshape(-1,2)
        # print(f"Output probability validation {epoch}")
        outs_prob_name="outs_prob_val_" + str(self.fold_id)
        np.save(self.config._save_dir / outs_prob_name, outs_prob)

        return self.valid_metrics.result(), outs, trgs
    
    def train_saver(self,epoch):
        self.model.eval()
        print("Entered train saver -")
        with torch.no_grad():
            outs = np.array([])
            trgs = np.array([])
            outs_prob = np.array([])
            for batch_idx, (data, target) in enumerate(self.full_train_loader):
                    data, target = data.to(self.device), target.to(self.device)
                    output = self.model(data)
                    # print("Extracted output")
                    # print(output)
                    # print(output.shape)
                    output_tensor = torch.nn.functional.softmax(output,dim=1)
                    outs_prob = np.append(outs_prob,output_tensor.cpu().numpy())
                    # Print the result
                    # print(output_tensor)
                    # loss = self.criterion(output, target, self.class_weights, self.device)
                    # try:
                    preds_ = output.data.max(1, keepdim=True)[1].cpu()
                    # except RuntimeError as e:
                    #     print(e)
                    #     import sys
                    #     sys.exit()
                    # for i in range(len(target)):
                    #     if int(target[i])==int(preds_[i]):
                    #             print(f"Correct detection of class {int(preds_[i])} : {output_tensor[i][int(preds_[i])]}")
                    #     else:
                    #             print(f"Predicted class probability {int(preds_[i])} : {output_tensor[i][int(preds_[i])]} and target class probability {int(target[i])} : {output_tensor[i][int(target[i])]}")

                    outs = np.append(outs, preds_.cpu().numpy())
                    trgs = np.append(trgs, target.data.cpu().numpy())
        outs_prob=outs_prob.reshape(-1,2)
        outs_name = "outs_train_" + str(self.fold_id) + "_" + str(epoch)
        trgs_name = "trgs_train_" + str(self.fold_id) + "_" + str(epoch)
        outs_prob_name="outs_prob_train_" + str(self.fold_id) + "_" + str(epoch)
        np.save(self.config._save_dir / outs_name, outs)
        np.save(self.config._save_dir / trgs_name, trgs)
        np.save(self.config._save_dir / outs_prob_name, outs_prob)
        print(outs.shape)
        print(trgs.shape)
        print(outs_prob.shape)

    def full_pred(self,epoch):
        self.model.eval()
        # print('Entered train saver')
        with torch.no_grad():
            outs = np.array([])
            trgs = np.array([])
            outs_prob = np.array([])
            for batch_idx, (data, target) in enumerate(self.full_test_loader):
                    # print("Extracted sample")
                    data, target = data.to(self.device), target.to(self.device)
                    output = self.model(data)
                    # print("The output is-")
                    # print(output)
                    # print("Entering softmax -")
                    output_tensor = torch.nn.functional.softmax(output,dim=1)
                    outs_prob = np.append(outs_prob,output_tensor.cpu().numpy())
                    # Print the result
                    # print(output_tensor)
                    preds_ = output.data.max(1, keepdim=True)[1].cpu()

                    # for i in range(len(target)):
                    #     if int(target[i])==int(preds_[i]):
                    #             print(f"Correct detection of class {int(preds_[i])} : {output_tensor[i][int(preds_[i])]}")
                    #     else:
                    #             print(f"Predicted class probability {int(preds_[i])} : {output_tensor[i][int(preds_[i])]} and target class probability {int(target[i])} : {output_tensor[i][int(target[i])]}")

                    outs = np.append(outs, preds_.cpu().numpy())
                    trgs = np.append(trgs, target.data.cpu().numpy())
        outs_prob=outs_prob.reshape(-1,2)
        outs_name = "model2_outsv" + str(self.fold_id) + "_" + str(epoch)
        trgs_name = "model2_trgsv" + str(self.fold_id) + "_" + str(epoch)
        outs_prob_name="model2_outs_probv" + str(self.fold_id) + "_" + str(epoch)
        np.save(self.config._save_dir / outs_name, outs)
        np.save(self.config._save_dir / trgs_name, trgs)
        np.save(self.config._save_dir / outs_prob_name, outs_prob)
        # print(outs.shape)
        # print(trgs.shape)
        # print(outs_prob.shape)


    def _progress(self, batch_idx):
        base = '[{}/{} ({:.0f}%)]'
        if hasattr(self.data_loader, 'n_samples'):
            current = batch_idx * self.data_loader.batch_size
            total = self.data_loader.n_samples
        else:
            current = batch_idx
            total = self.len_epoch
        return base.format(current, total, 100.0 * current / total)
