from __future__ import annotations
from typing import Any, Tuple

import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from sklego.preprocessing import RepeatingBasisFunction
from sklearn.preprocessing import OneHotEncoder

class SinCosEncoder:
    def fit_transform(self, data: np.ndarray):
        return np.stack((np.cos(data), np.sin(data)), axis=1)


class AdDataset(Dataset):
    def __init__(
            self,
            data: pd.DataFrame,
            time_encoder: SinCosEncoder | RepeatingBasisFunction = SinCosEncoder(),
            day_encoder: OneHotEncoder = OneHotEncoder(sparse_output=False),
            use_3_time_bins: bool = False
    ):
        super().__init__()
        self.data = data
        self.time_encoder = time_encoder
        self.day_encoder = day_encoder
        self.use_3_time_bins = use_3_time_bins

        self.TARGET_FEATURES = ['hour' + str(idx) for idx in range(48)]

        self._prepare_data()

    def _prepare_data(self):
        self.y = self.data[self.TARGET_FEATURES].to_numpy()

        x = self.data.drop(columns=self.TARGET_FEATURES)
        
        x_day_encoded = self.day_encoder.fit_transform(x['day_of_week'].to_numpy().reshape(-1, 1)).astype(np.float32)
        x_time_encoded = self.time_encoder.fit_transform(x['hour_of_day'].to_numpy()).astype(np.float32)

        self.x = np.concatenate([x_day_encoded, x_time_encoded], axis=1)

    def __getitem__(self, index) -> Tuple[np.ndarray, np.ndarray]:
        return self.x[index], self.y[index]
    
    def __len__(self):
        return self.x.shape[0]


