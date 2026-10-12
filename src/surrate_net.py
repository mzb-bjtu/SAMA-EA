"""XGBoost, random-forest and neural-network screening models."""
import torch
import torch.nn as nn
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
from copy import deepcopy

class BaseSurrogate:

    def __init__(self):
        self.is_trained = False

    def train(self, X, y):

        raise NotImplementedError

    def predict(self, X):

        raise NotImplementedError

    def predict_single(self, x):

        return self.predict(x.reshape(1, -1))[0]

class DNNSurrogate(BaseSurrogate):
    def __init__(self, input_dim, device='cuda'):
        super().__init__()
        self.device = device
        self.model = self._build_network(input_dim)
        self.optimizer = None
        self.criterion = nn.MSELoss()

    def _build_network(self, input_dim):
        return nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def set_optimizer(self, lr=1e-3):
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

    def train(self, X, y, epochs=10):

        if len(X) < 5:
            return

        X = torch.tensor(X, dtype=torch.float32).to(self.device)
        y = torch.tensor(y, dtype=torch.float32).view(-1, 1).to(self.device)

        self.model.train()
        for _ in range(epochs):
            self.optimizer.zero_grad()
            pred = self.model(X)
            loss = self.criterion(pred, y)
            loss.backward()
            self.optimizer.step()

        self.is_trained = True

    def predict(self, X):
        if len(X) == 0:
            return np.array([])

        X = torch.tensor(X, dtype=torch.float32).to(self.device)
        self.model.eval()
        with torch.no_grad():
            pred = self.model(X).cpu().numpy().flatten()
        return pred

class RandomForestSurrogate(BaseSurrogate):
    def __init__(self, input_dim, n_estimators=100, max_depth=10, n_jobs=1):
        super().__init__()
        self.input_dim = input_dim
        self.model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=n_jobs
        )

    def train(self, X, y, epochs=None):
        if len(X) < 3:
            return

        if len(X.shape) == 1:
            X = X.reshape(-1, 1)

        self.model.fit(X, y)
        self.is_trained = True

    def predict(self, X):
        if len(X) == 0:
            return np.array([])

        if len(X.shape) == 1:
            X = X.reshape(-1, 1)

        return self.model.predict(X)

    def get_uncertainty(self, X):

        if not self.is_trained or len(X) == 0:
            return np.ones(len(X))

        if len(X.shape) == 1:
            X = X.reshape(-1, 1)

        tree_predictions = np.array([
            tree.predict(X) for tree in self.model.estimators_
        ])
        return np.std(tree_predictions, axis=0)

class XGBoostSurrogate(BaseSurrogate):
    def __init__(self, input_dim, n_estimators=100, max_depth=6, learning_rate=0.1):
        super().__init__()
        self.input_dim = input_dim
        self.model = xgb.XGBRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            reg_alpha=0.1,
            random_state=42,
            n_jobs=1
        )

    def train(self, X, y, epochs=None):
        if len(X) < 3:
            return

        if len(X.shape) == 1:
            X = X.reshape(-1, 1)

        self.model.fit(X, y)
        self.is_trained = True

    def predict(self, X):
        if len(X) == 0:
            return np.array([])

        if len(X.shape) == 1:
            X = X.reshape(-1, 1)

        return self.model.predict(X)

def create_surrogate_model(model_type, input_dim, device='cuda', **kwargs):

    if model_type == 'dnn':
        model = DNNSurrogate(input_dim, device)
        model.set_optimizer(lr=kwargs.get('lr', 1e-3))
        return model
    elif model_type == 'rf':

        return RandomForestSurrogate(
            input_dim,
            n_estimators=kwargs.get('n_estimators', 100),
            max_depth=kwargs.get('max_depth', 10)
        )
    elif model_type == 'xgb':

        return XGBoostSurrogate(
            input_dim,
            n_estimators=kwargs.get('n_estimators', 100),
            max_depth=kwargs.get('max_depth', 6),
            learning_rate=kwargs.get('learning_rate', 0.1)
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")
