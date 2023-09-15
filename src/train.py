import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from torch.optim import AdamW
import torch.nn as nn
from scipy.spatial.distance import jensenshannon

import time

from src.loss import JSD, JSD2
from src.data import AdDataset
from src.models import FCSM
from src.trainer import Trainer


N_FOLDS = 5




def prepare_df():
    # start = time.perf_counter()
    # df = pd.read_excel('data/v3.xlsx')
    # print(f'load time xlsx: {time.perf_counter() - start}')
    # df.drop(columns=['tgmessageid'], inplace=True)
    
    # df = df.iloc[1:, :]

    # df.to_csv('data/v3.csv')
    # start = time.perf_counter()
    df = pd.read_csv('data/v3.csv')
    # print(f'load time csv: {time.perf_counter() - start}')

    target_features = ['hour' + str(idx) for idx in range(48)]
    df = df.astype(np.float32)
    # df[[*target_features, 'sum_costs']] = df[[*target_features, 'sum_costs']].astype(np.float64)
    # df[['day_of_week', 'hour_of_day']] = df[['day_of_week', 'hour_of_day']].astype(np.int64)

    df[target_features] = df[target_features].divide(df['sum_costs'], axis=0)

    np.random.seed(42)

    inds = np.arange(df.shape[0])
    np.random.shuffle(inds)

    df_shuffled = df.iloc[inds, :]

    return df_shuffled


if __name__ == '__main__':
    df = prepare_df()
    folds = np.array_split(df, N_FOLDS)

    learning_rate = 0.5
    n_epochs = 100
    weight_decay = 0.0
    batch_size = 64

    for idx, val in enumerate(folds):
        print(f'CV fold {idx}')
        train = pd.concat([fold for fold in folds if fold is not val])
        
        datasets = {
            'train': AdDataset(train),
            'val': AdDataset(val)
        }

        model = FCSM(7 + 2)
        # loss_fn = JSD2()
        loss_fn = nn.KLDivLoss(reduction='batchmean', log_target=False)
        
        trainer = Trainer(
            model=model,
            datasets=datasets,
            optimizer=AdamW(
                model.parameters(),
                lr=learning_rate,
                weight_decay=weight_decay
            ),
            loss_fn=loss_fn
        )

        trainer.train(n_epochs=n_epochs, batch_size=batch_size, report_frequency=1)
