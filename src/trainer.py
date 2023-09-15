import os
from typing import Dict
import numpy as np
import scipy as sp

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.metrics import compute_metrics
from src.data import AdDataset


class Trainer:
    def __init__(
            self,
            model: nn.Module,
            datasets: Dict[str, AdDataset],
            optimizer: torch.optim.Optimizer,
            loss_fn: nn.Module,
    ):
        self.model = model
        self.datasets = datasets
        self.optimizer = optimizer
        self.loss_fn = loss_fn

    def train(self, n_epochs: int, batch_size: int, report_frequency: int):
        train_dataloader = DataLoader(self.datasets['train'], batch_size=batch_size, drop_last=True)
        for epoch in range(1, n_epochs + 1):
            self.model.train()

            epoch_losses = []
            for iteration, batch in enumerate(train_dataloader):
                self.optimizer.zero_grad()

                x_batch, y_batch = batch
                
                predict = self.model(x_batch)
                predict = predict.squeeze(1)

                loss = self.loss_fn(predict, y_batch)
                loss.backward()

                epoch_losses.append(loss.detach().cpu().numpy())
                
                self.optimizer.step()
                
                # if iteration % report_frequency == 0:
                #     print(f'(epoch) {epoch:3d} (iteration) {iteration:5d} (loss) {loss.item():.4f}')

            train_metrics, train_loss = self._evaluate('train', batch_size)
            val_metrics, val_loss = self._evaluate('val', batch_size)

            print(
                f'E {epoch:03d} | '
                f'hell: t {train_metrics["hellinger"]:.4f}, v {val_metrics["hellinger"]:.4f} | '
                f'loss t {train_loss:.6f}, v {val_loss:.6f}'
            )
            
            # self._save_checkpoint(self.model, epoch, self.checkpoints_dir)
            # self._save_metrics(
            #     {'train': train_metrics, 'val': val_metrics},
            #     {'train': train_loss, 'val': val_loss},
            #     epoch
            # )
            # self._save_loss(epoch_losses)

    @torch.no_grad()
    def _evaluate(self, part_name: str, batch_size: int):
        self.model.eval()

        dataloader = DataLoader(self.datasets[part_name], batch_size=batch_size, drop_last=True)

        predict = []
        target = []
        for batch in dataloader:
            x_batch, y_batch = batch

            predict.append(self.model(x_batch))
            target.append(y_batch)

        predict = torch.cat(predict).squeeze(1).cpu().numpy()
        predict = np.round(sp.special.expit(predict))

        target = torch.cat(target).cpu().numpy()

        loss = float(self.loss_fn(torch.tensor(predict), torch.tensor(target)).cpu())
        metrics = compute_metrics(predict, target)

        return metrics, loss

        