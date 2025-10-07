import numpy as np
import pandas as pd
from basemodel import LinearModel
from basetrainer import Trainer


_TRAIN_FEATURE_MEAN = None
_TRAIN_FEATURE_STD = None

def load_and_preprocess_data(data_file: str = "data/train.csv"):
    dataset = pd.read_csv(data_file)
    """
    Divide the dataset into features and target

    You can do all possible modifications to features, but DO NOT change the targets

    return:
        features (np.ndarray): Input features, shape [num_samples, in_features]
        targets (np.ndarray): Target values, shape [num_samples]
    """

    if "Run_time" not in dataset.columns:
        raise ValueError("Expected 'Run_time' column to be present in the dataset.")

    feature_columns = [col for col in dataset.columns if col != "Run_time"]
    features = dataset[feature_columns].to_numpy(dtype=np.float64)
    targets = dataset["Run_time"].to_numpy(dtype=np.float64)

    global _TRAIN_FEATURE_MEAN, _TRAIN_FEATURE_STD
    if _TRAIN_FEATURE_MEAN is None or _TRAIN_FEATURE_STD is None:
        _TRAIN_FEATURE_MEAN = features.mean(axis=0)
        _TRAIN_FEATURE_STD = features.std(axis=0, ddof=0)
        _TRAIN_FEATURE_STD[_TRAIN_FEATURE_STD == 0] = 1.0

    features = (features - _TRAIN_FEATURE_MEAN) / _TRAIN_FEATURE_STD

    print(f"Data size: {features.shape[0]}. Features num: {features.shape[1]}")
    return features, targets

class LinearRegressionModel(LinearModel):
    def __init__(self, in_features: int, out_features: int):
        """
        Linear regression model, inherits from LinearModel.

        Args:
            in_features (int): Number of input features.
            out_features (int): Number of output features (usually 1).
        """
        self.in_features = in_features
        self.out_features = out_features
        self.weight = np.random.randn(in_features, out_features) * 0.01
        self.bias = np.zeros((out_features,), dtype=np.float64)

    def forward(self, features: np.ndarray) -> np.ndarray:
        """
        Predict the output given input.

        Args:
            features (np.ndarray): Input features, shape [batch_size, in_features].
        """
        features = np.asarray(features, dtype=np.float64)
        single_input = False
        if features.ndim == 1:
            features = features.reshape(1, -1)
            single_input = True
        outputs = features @ self.weight + self.bias
        if single_input:
            return outputs.reshape(-1)
        return outputs

    def gradient(self, features: np.ndarray, targets: np.ndarray, predictions: np.ndarray) -> tuple:
        """
        Compute gradients for MSE loss.

        Args:
            features (np.ndarray): Input features, shape [batch_size, in_features].
            targets (np.ndarray): True values, shape [batch_size, out_features].
            predictions (np.ndarray): Predicted values, shape [batch_size, out_features].

        Returns:
            tuple: (dw, db), gradients for weights and bias.
        """
        features = np.asarray(features, dtype=np.float64)
        targets = np.asarray(targets, dtype=np.float64)
        predictions = np.asarray(predictions, dtype=np.float64)

        if features.ndim == 1:
            features = features.reshape(1, -1)
        if targets.ndim == 1:
            targets = targets.reshape(-1, 1)
        if predictions.ndim == 1:
            predictions = predictions.reshape(-1, 1)

        batch_size = features.shape[0]
        residual = predictions - targets
        dw = (2.0 / batch_size) * features.T @ residual
        db = (2.0 / batch_size) * residual.sum(axis=0)
        return dw, db

    def backpropagation(self, features: np.ndarray, targets: np.ndarray, predictions: np.ndarray, learning_rate: float = 0.01) -> float:
        """
        Perform backpropagation, compute MSE loss and update parameters.

        Args:
            features (np.ndarray): Input features, shape [batch_size, in_features].
            targets (np.ndarray): True values, shape [batch_size, out_features].
            predictions (np.ndarray): True values, shape [batch_size, out_features].
            learning_rate (float): Learning rate, default 0.01.
        """
        features = np.asarray(features, dtype=np.float64)
        targets = np.asarray(targets, dtype=np.float64)
        predictions = np.asarray(predictions, dtype=np.float64)

        if targets.ndim == 1:
            targets = targets.reshape(-1, 1)
        if predictions.ndim == 1:
            predictions = predictions.reshape(-1, 1)

        loss = np.mean((predictions - targets) ** 2)
        dw, db = self.gradient(features, targets, predictions)
        self.weight -= learning_rate * dw
        self.bias -= learning_rate * db
        return float(loss)

class LinearRegressionTrainer(Trainer):
    def __init__(self, model, train_dataloader, eval_dataloader=None, 
                 save_dir=None, learning_rate=0.01, eval_strategy="epoch", 
                 eval_steps=100, num_epochs=10, eval_metric="mae"):
        super().__init__(model, train_dataloader, eval_dataloader, save_dir, 
                         learning_rate, eval_strategy, eval_steps, num_epochs, eval_metric)

    def compute_loss(self, batch_pred, batch_grd):
        """
        Compute loss based on model type with detailed checks for linear regression.

        Args:
            batch_pred: Predicted values, shape [batch_size, out_features].
            batch_grd: True values/labels, shape [batch_size, out_features].

        Returns:
            float: Mean loss for the batch.
        """
        batch_pred = np.asarray(batch_pred, dtype=np.float64)
        batch_grd = np.asarray(batch_grd, dtype=np.float64)

        if batch_pred.ndim == 1:
            batch_pred = batch_pred.reshape(-1, 1)
        if batch_grd.ndim == 1:
            batch_grd = batch_grd.reshape(-1, 1)

        return float(np.mean((batch_pred - batch_grd) ** 2))

def linear_regression_analytic(X, y):
    """
    Calculate the analytical linear regression results.

    Args:
        X (np.ndarray): Input features, shape [num_samples, in_features]
        y (np.ndarray): True values, shape [num_samples, out_features]

    Return:
        weight (np.ndarray): Model weight
        bias (np.ndarray | float): Model bias
    """
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    if X.ndim != 2:
        raise ValueError("Input features X must be a 2D array.")

    if y.ndim == 1:
        y = y.reshape(-1, 1)

    ones = np.ones((X.shape[0], 1), dtype=np.float64)
    X_aug = np.hstack([X, ones])

    theta, *_ = np.linalg.lstsq(X_aug, y, rcond=None)
    weight = theta[:-1]
    bias = theta[-1].reshape(-1)
    if bias.size == 1:
        bias = bias.item()
    return weight, bias

class LogisticRegressionModel(LinearModel):
    def __init__(self, in_features: int, out_features: int):
        """
        Logistic regression model, inherits from LinearModel.

        Args:
            in_features (int): Number of input features.
            out_features (int): Number of output features (usually 1 for binary classification).
        """
        self.in_features = in_features
        self.out_features = out_features
        limit = np.sqrt(6.0 / (in_features + out_features))
        self.weight = np.random.uniform(-limit, limit, size=(in_features, out_features)).astype(np.float64)
        self.bias = np.zeros((out_features,), dtype=np.float64)

    def _sigmoid(self, x: np.ndarray) -> np.ndarray:
        """
        Compute sigmoid function.

        Args:
            x (np.ndarray): Input values.

        Returns:
            np.ndarray: Sigmoid output.
        """
        x = np.asarray(x, dtype=np.float64)
        # Clip inputs to avoid overflow
        x = np.clip(x, -500, 500)
        return 1.0 / (1.0 + np.exp(-x))

    def forward(self, features: np.ndarray) -> np.ndarray:
        """
        Predict the output given input.

        Args:
            features (np.ndarray): Input features, shape [batch_size, in_features].
        """
        features = np.asarray(features, dtype=np.float64)
        single_input = False
        if features.ndim == 1:
            features = features.reshape(1, -1)
            single_input = True
        logits = features @ self.weight + self.bias
        probs = self._sigmoid(logits)
        if single_input:
            return probs.reshape(-1)
        return probs

    def gradient(self, features: np.ndarray, targets: np.ndarray, predictions: np.ndarray) -> tuple:
        """
        Compute gradients for binary cross-entropy loss.

        Args:
            features (np.ndarray): Input features, shape [batch_size, in_features].
            targets (np.ndarray): True labels (0 or 1), shape [batch_size, out_features].
            predictions (np.ndarray): Predicted probabilities, shape [batch_size, out_features].

        Returns:
            tuple: (dw, db), gradients for weights and bias.
        """
        features = np.asarray(features, dtype=np.float64)
        targets = np.asarray(targets, dtype=np.float64)
        predictions = np.asarray(predictions, dtype=np.float64)

        if features.ndim == 1:
            features = features.reshape(1, -1)
        if targets.ndim == 1:
            targets = targets.reshape(-1, 1)
        if predictions.ndim == 1:
            predictions = predictions.reshape(-1, 1)

        batch_size = features.shape[0]
        diff = predictions - targets
        dw = (features.T @ diff) / batch_size
        db = diff.mean(axis=0)
        return dw, db
    
    def backpropagation(self, features: np.ndarray, targets: np.ndarray, predictions: np.ndarray, learning_rate: float = 0.01) -> float:
        """
        Perform backpropagation, compute binary cross-entropy loss and update parameters.

        Args:
            features (np.ndarray): Input features, shape [batch_size, in_features].
            targets (np.ndarray): True labels (0 or 1), shape [batch_size, out_features].
            learning_rate (float): Learning rate, default 0.01.

        Returns:
            float: Binary cross-entropy loss for the batch.
        """
        features = np.asarray(features, dtype=np.float64)
        targets = np.asarray(targets, dtype=np.float64)
        predictions = np.asarray(predictions, dtype=np.float64)

        if targets.ndim == 1:
            targets = targets.reshape(-1, 1)
        if predictions.ndim == 1:
            predictions = predictions.reshape(-1, 1)

        predictions = np.clip(predictions, 1e-12, 1 - 1e-12)
        loss = -np.mean(targets * np.log(predictions) + (1 - targets) * np.log(1 - predictions))

        dw, db = self.gradient(features, targets, predictions)
        self.weight -= learning_rate * dw
        self.bias -= learning_rate * db
        return float(loss)

class LogisticRegressionTrainer(Trainer):
    def __init__(self, model, train_dataloader, eval_dataloader=None, 
                 save_dir=None, learning_rate=0.01, eval_strategy="epoch", 
                 eval_steps=100, num_epochs=10, eval_metric="f1"):
        super().__init__(model, train_dataloader, eval_dataloader, save_dir, 
                         learning_rate, eval_strategy, eval_steps, num_epochs, eval_metric)
        
    def compute_loss(self, batch_pred, batch_grd):
        batch_pred = np.asarray(batch_pred, dtype=np.float64)
        batch_grd = np.asarray(batch_grd, dtype=np.float64)

        if batch_pred.ndim == 1:
            batch_pred = batch_pred.reshape(-1, 1)
        if batch_grd.ndim == 1:
            batch_grd = batch_grd.reshape(-1, 1)

        batch_pred = np.clip(batch_pred, 1e-12, 1 - 1e-12)
        loss = -np.mean(batch_grd * np.log(batch_pred) + (1 - batch_grd) * np.log(1 - batch_pred))
        return float(loss)
