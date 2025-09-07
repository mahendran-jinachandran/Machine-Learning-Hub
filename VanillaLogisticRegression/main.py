import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score



class LogisticRegressorGD:
    def __init__(self, alpha, num_iters, normalize, l2, tol, verbose=False, random_state= 42):
        self.alpha = alpha
        self.num_iters = num_iters
        self.normalize = normalize
        self.l2 = l2
        self.tol = tol
        self.verbose = verbose
        self.random_state = random_state

        self.w_ = None
        self.b_ = None
        self.mu_ = None
        self.sigma_ = None
        self.history_ = []
    
    @staticmethod
    def _to_xy(X, y=None):
        X = np.asarray(X, float)
        y = None if y is None else np.asarray(y, float).ravel()
        return X, y
    
    @staticmethod
    def _standardize_train(X):
        mu = X.mean(axis=0)
        sigma = X.std(axis=0)
        sigma[sigma == 0] = 1.0
        return (X - mu)/sigma, mu, sigma
    
    @staticmethod
    def _standardize_apply(X, mu, sigma):
        sigma = np.where(sigma == 0, 1.0, sigma)
        return (X - mu)/sigma
    

    @staticmethod
    def _sigmoid(z):
        # numerically stable sigmoid
        # for large negative z, exp(-z) overflows; use piecewise
        out = np.empty_like(z, dtype=float)
        pos = z >= 0
        neg = ~pos
        out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
        exp_z = np.exp(z[neg])
        out[neg] = exp_z / (1.0 + exp_z)
        return out
    

    def _loss(self, X, y, w, b):
        """
        Binary cross-entropy (log loss) with L2 on w (not on b):
        L = -1/m * sum[y*log(p)+(1-y)*log(1-p)] + (l2/(2m))*||w||^2
        """
        m = X.shape[0]
        p = self._sigmoid(X @ w + b)

        eps = 1e-12
        p = np.clip(p, eps, 1 - eps)
        ce = - (y * np.log(p) + (1 - y) * np.log(1 - p)).mean()
        if self.l2 > 0:
            ce += (self.l2 / (2*m)) * (w @ w)
        return float(ce)
    
    def _grad(self, X, y, w, b):
        m = X.shape[0]
        p = self._sigmoid(X @ w + b)
        err = p - y
        dj_dw = (X.T @ err) / m
        dj_db = float(err.mean())
        if self.l2 > 0:
            dj_dw += (self.l2 / m) * w
        return dj_dw, dj_db
    
    def fit(self, X, y, w_init=None, b_init=0.0):
        X, y = self._to_xy(X, y)
        if self.normalize:
            Xn, self.mu_, self.sigma_ = self._standardize_train(X)
        else:
            Xn, self.mu_, self.sigma_ = X, None, None

        n = Xn.shape[1]
        if w_init is None:
            w = np.zeros(n)
        else:
            w = np.asarray(w_init, float).ravel()
            assert w.shape[0] == n, "w_init shape mismatch."
        b = float(b_init)

        self.history_ = []
        last = None
        small_moves = 0

        for i in range(self.num_iters):
            dj_dw, dj_db = self._grad(Xn, y, w, b)
            w -= self.alpha * dj_dw
            b -= self.alpha * dj_db

            L = self._loss(Xn, y, w, b)
            self.history_.append(L)

            if self.verbose and i % max(1, self.n_iters//10) == 0:
                delta = 0.0 if last is None else (L - last)
                print(f"iter {i:4d} | loss {L:.6f} | Δ {delta:.2e}")
            if self.tol > 0 and last is not None:
                if abs(L - last) < self.tol:
                    small_moves += 1
                    if small_moves >= 5:
                        if self.verbose:
                            print(f"Early stop at iter {i} (Δloss < {self.tol})")
                        break
                else:
                    small_moves = 0
            last = L

        self.w_, self.b_ = w, b
        return self
    
    def _prepare_X(self, X):
        X = np.asarray(X, float)
        if self.normalize:
            if self.mu_ is None or self.sigma_ is None:
                raise ValueError("Trained with normalize=True but mu_/sigma_ missing.")
            X = self._standardize_apply(X, self.mu_, self.sigma_)
        return X

    def predict_proba(self, X):
        if self.w_ is None:
            raise ValueError("Call fit() first.")
        Xn = self._prepare_X(X)
        return self._sigmoid(Xn @ self.w_ + self.b_)

    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X) >= threshold).astype(int)
        

if __name__ == "__main__":
    data = load_breast_cancer()
    X, y = data.data, data.target

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    model = LogisticRegressorGD(alpha=1e-2, num_iters=2000, normalize=True, l2=0.0, tol=0.0)
    model.fit(X_train, y_train)

    # predict
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)

    # evaluate with sklearn
    print("Accuracy :", accuracy_score(y_test, y_pred))
    print("Precision:", precision_score(y_test, y_pred))
    print("Recall   :", recall_score(y_test, y_pred))
    print("F1 Score :", f1_score(y_test, y_pred))
    print("ROC AUC  :", roc_auc_score(y_test, y_proba))
