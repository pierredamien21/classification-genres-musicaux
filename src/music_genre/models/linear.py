"""Numeric-only export of a standardized multinomial logistic classifier."""

import numpy as np
from scipy.special import softmax


class LinearProbabilityModel:
    def __init__(self, mean, scale, coefficients, intercept, classes):
        self.mean = np.asarray(mean, dtype=np.float64)
        self.scale = np.asarray(scale, dtype=np.float64)
        self.coefficients = np.asarray(coefficients, dtype=np.float64)
        self.intercept = np.asarray(intercept, dtype=np.float64)
        self.classes = np.asarray(classes, dtype=np.int64)
        if (self.mean.ndim != 1 or self.scale.shape != self.mean.shape
                or self.classes.ndim != 1 or len(self.classes) < 3
                or len(set(self.classes)) != len(self.classes)
                or self.coefficients.shape != (len(self.classes), len(self.mean))
                or self.intercept.shape != (len(self.classes),)
                or np.any(self.scale <= 0)
                or any(not np.isfinite(a).all()
                       for a in (self.mean, self.scale, self.coefficients, self.intercept))):
            raise ValueError("Invalid linear model arrays")

    def predict_proba(self, x):
        x = np.asarray(x, dtype=np.float64)
        if x.ndim != 2 or x.shape[1] != len(self.mean) or not np.isfinite(x).all():
            raise ValueError("Invalid feature matrix")
        scores = ((x - self.mean) / self.scale) @ self.coefficients.T + self.intercept
        return softmax(scores, axis=1)

    def save(self, path):
        np.savez(path, mean=self.mean, scale=self.scale, coefficients=self.coefficients,
                 intercept=self.intercept, classes=self.classes)

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as arrays:
            return cls(**{key: arrays[key] for key in
                          ("mean", "scale", "coefficients", "intercept", "classes")})
