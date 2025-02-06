import heapq
import logging
import os
from time import time
from typing import Dict

import numpy as np
import torch
from torch import optim
from transformers import (get_constant_schedule_with_warmup,
                          get_linear_schedule_with_warmup)

from .utils import delete_file, ensure_dir


class Trainer(object):

    def __init__(self, args, model, data_num):
        self.args = args
        self.model = model
        self.logger = logging.getLogger()

        self.lr = args.lr
        self.learner = args.learner
        self.lr_scheduler_type = args.lr_scheduler_type

        self.weight_decay = args.weight_decay
        self.epochs = args.epochs
        self.warmup_steps = args.warmup_epochs * data_num
        self.max_steps = args.epochs * data_num

        self.save_limit = args.save_limit
        self.best_save_heap = []
        self.newest_save_queue = []
        self.eval_step = min(args.eval_step, self.epochs)
        self.device = args.device
        self.device = torch.device(self.device)
        self.ckpt_dir = args.ckpt_dir
        ensure_dir(self.ckpt_dir)

        self.best_loss = np.inf
        self.best_collision_rate = np.inf
        self.best_loss_ckpt = "best_loss_model.pth"
        self.best_collision_ckpt = "best_collision_model.pth"
        self.optimizer = self._build_optimizer()
        self.scheduler = self._get_scheduler()
        self.model = self.model.to(self.device)

    def _build_optimizer(self):

        params = self.model.parameters()
        learner = self.learner
        learning_rate = self.lr
        weight_decay = self.weight_decay

        if learner.lower() == "adam":
            optimizer = optim.Adam(params,
                                   lr=learning_rate,
                                   weight_decay=weight_decay)
        elif learner.lower() == "sgd":
            optimizer = optim.SGD(params,
                                  lr=learning_rate,
                                  weight_decay=weight_decay)
        elif learner.lower() == "adagrad":
            optimizer = optim.Adagrad(params,
                                      lr=learning_rate,
                                      weight_decay=weight_decay)
            for state in optimizer.state.values():
                for k, v in state.items():
                    if torch.is_tensor(v):
                        state[k] = v.to(self.device)
        elif learner.lower() == "rmsprop":
            optimizer = optim.RMSprop(params,
                                      lr=learning_rate,
                                      weight_decay=weight_decay)
        elif learner.lower() == 'adamw':
            optimizer = optim.AdamW(params,
                                    lr=learning_rate,
                                    weight_decay=weight_decay)
        else:
            self.logger.warning(
                "Received unrecognized optimizer, set default Adam optimizer")
            optimizer = optim.Adam(params, lr=learning_rate)
        return optimizer

    def _get_scheduler(self):
        if self.lr_scheduler_type.lower() == "linear":
            lr_scheduler = get_linear_schedule_with_warmup(
                optimizer=self.optimizer,
                num_warmup_steps=self.warmup_steps,
                num_training_steps=self.max_steps)
        else:
            lr_scheduler = get_constant_schedule_with_warmup(
                optimizer=self.optimizer, num_warmup_steps=self.warmup_steps)

        return lr_scheduler

    def _check_nan(self, loss):
        if torch.isnan(loss):
            raise ValueError("Training loss is nan")

    def _train_epoch(self, train_data, epoch_idx):

        self.model.train()

        total_loss = 0
        total_recon_loss = 0
        total_quant_loss = 0
        iter_data = train_data

        for batch_idx, data in enumerate(iter_data):
            if isinstance(data, Dict):
                for k, v in data.items():
                    if isinstance(v, torch.Tensor):
                        data[k] = v.to(self.device)
            else:
                data = data.to(self.device)

            self.optimizer.zero_grad()
            out, rq_loss, indices = self.model(data)
            loss, loss_recon, quant_loss = self.model.compute_loss(out,
                                                                   rq_loss,
                                                                   xs=data)
            self._check_nan(loss)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            self.scheduler.step()
            # print(self.scheduler.get_last_lr())
            total_loss += loss.item()
            total_recon_loss += loss_recon.item()
            total_quant_loss += quant_loss.item()

        return total_loss, total_recon_loss, total_quant_loss

    @torch.no_grad()
    def _valid_epoch(self, valid_data):

        self.model.eval()
        iter_data = valid_data

        semantic_indices_set = set()
        collaborate_indices_set = set()

        num_sample = 0
        for batch_idx, data in enumerate(iter_data):
            # num_sample += len(data)
            num_sample += data['batch_size']
            # data = data.to(self.device)
            if isinstance(data, Dict):
                for k, v in data.items():
                    if isinstance(v, torch.Tensor):
                        data[k] = v.to(self.device)
            else:
                data = data.to(self.device)
            indices = self.model.get_indices(data)

            semantic_indices, collaborate_indices = indices
            semantic_indices = semantic_indices.view(
                -1, semantic_indices.shape[-1]).cpu().numpy()
            for index in semantic_indices:
                code = "-".join([str(int(_)) for _ in index])
                semantic_indices_set.add(code)

            collaborate_indices = collaborate_indices.view(
                -1, collaborate_indices.shape[-1]).cpu().numpy()
            for index in collaborate_indices:
                code = "-".join([str(int(_)) for _ in index])
                collaborate_indices_set.add(code)

        semantic_collision_rate = (
            num_sample - len(list(semantic_indices_set))) / num_sample
        collaborate_collision_rate = (
            num_sample - len(list(collaborate_indices_set))) / num_sample
        collision_rate = (semantic_collision_rate +
                          collaborate_collision_rate) / 2.0

        return collision_rate

    def _save_checkpoint(self, epoch, collision_rate=1, ckpt_file=None):

        ckpt_path = os.path.join(self.ckpt_dir,ckpt_file) if ckpt_file \
            else os.path.join(self.ckpt_dir, 'epoch_%d_collision_%.4f_model.pth' % (epoch, collision_rate))
        state = {
            "args": self.args,
            "epoch": epoch,
            "best_loss": self.best_loss,
            "best_collision_rate": self.best_collision_rate,
            "state_dict": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }
        torch.save(state, ckpt_path, pickle_protocol=4)

        self.logger.info("Saving current" + f": {ckpt_path}")

        return ckpt_path

    def _generate_train_loss_output(self, epoch_idx, s_time, e_time, loss,
                                    recon_loss, quant_loss):

        train_loss_output = ("epoch %d training" + " [" + "time" +
                             ": %.2fs, ") % (epoch_idx, e_time - s_time)
        train_loss_output += "train loss" + ": %.4f" % loss + ", "
        train_loss_output += "reconstruction loss" + ": %.4f" % recon_loss + ", "
        train_loss_output += "quantize loss" + ": %.4f" % quant_loss

        return train_loss_output + "]"

    def fit(self, data):

        cur_eval_step = 0

        for epoch_idx in range(self.epochs):
            training_start_time = time()
            train_loss, train_recon_loss, train_quant_loss = self._train_epoch(
                data, epoch_idx)
            training_end_time = time()
            train_loss_output = self._generate_train_loss_output(
                epoch_idx, training_start_time, training_end_time, train_loss,
                train_recon_loss, train_quant_loss)

            if (epoch_idx + 1) % self.eval_step == 0:
                self.logger.info(train_loss_output)

                valid_start_time = time()
                collision_rate = self._valid_epoch(data)

                if train_loss < self.best_loss:
                    self.best_loss = train_loss
                    self._save_checkpoint(epoch=epoch_idx,
                                          ckpt_file=self.best_loss_ckpt)

                if collision_rate < self.best_collision_rate:
                    self.best_collision_rate = collision_rate
                    cur_eval_step = 0
                    self._save_checkpoint(epoch_idx,
                                          collision_rate=collision_rate,
                                          ckpt_file=self.best_collision_ckpt)
                else:
                    cur_eval_step += 1

                valid_end_time = time()
                valid_score_output = ("epoch %d evaluating" + " [" + "time" +
                                      ": %.2fs, " + "collision_rate" + ": %f]"
                                      ) % (epoch_idx, valid_end_time -
                                           valid_start_time, collision_rate)

                self.logger.info(valid_score_output)
                ckpt_path = self._save_checkpoint(
                    epoch_idx, collision_rate=collision_rate)
                now_save = (-collision_rate, ckpt_path)
                if len(self.newest_save_queue) < self.save_limit:
                    self.newest_save_queue.append(now_save)
                    heapq.heappush(self.best_save_heap, now_save)
                else:
                    old_save = self.newest_save_queue.pop(0)
                    self.newest_save_queue.append(now_save)
                    if collision_rate < -self.best_save_heap[0][0]:
                        bad_save = heapq.heappop(self.best_save_heap)
                        heapq.heappush(self.best_save_heap, now_save)

                        if bad_save not in self.newest_save_queue:
                            delete_file(bad_save[1])

                    if old_save not in self.best_save_heap:
                        delete_file(old_save[1])

        return self.best_loss, self.best_collision_rate
