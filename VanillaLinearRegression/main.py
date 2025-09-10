import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, root_mean_squared_error, r2_score

class LinearRegressorGD:
    def __init__(self, alpha=1e-2, n_iters=1000, normalize=True, tol=0.0, random_state=None, verbose=False):
        self.alpha = alpha
        self.n_iters = n_iters
        self.normalize = normalize
        self.tol = tol
        self.random_state = random_state
        self.verbose = verbose

        self.w_ = None
        self.b_ = None
        self.mu_ = None
        self.sigma_ = None
        self.history_ = []

    @staticmethod
    def _to_xy(X, y=None):
        X = np.asarray(X, dtype=float)
        y = None if y is None else np.asarray(y, dtype=float).ravel()
        return X, y

    @staticmethod
    def _standardize_train(X):
        mu = X.mean(axis=0)
        sigma = X.std(axis=0)
        sigma[sigma == 0] = 1.0
        return (X - mu) / sigma, mu, sigma

    @staticmethod
    def _standardize_apply(X, mu, sigma):
        sigma = np.where(sigma == 0, 1.0, sigma)
        return (X - mu) / sigma

    @staticmethod
    def _compute_cost(X, y, w, b):
        preds = X @ w + b
        err = preds - y
        return float((err @ err) / (2 * X.shape[0]))

    @staticmethod
    def _compute_gradient(X, y, w, b):
        m = X.shape[0]
        preds = X @ w + b
        err = preds - y
        dj_dw = (X.T @ err) / m
        dj_db = float(err.mean())
        return dj_dw, dj_db
    
    def _standardize(self, X):
        if self.normalize:
            Xn, self.mu_, self.sigma_ = self._standardize_train(X)
        else:
            Xn = X
            self.mu_ = None
            self.sigma_ = None
        
        return Xn
    
    def set_random_seed(self):
        # init params
        if self.random_state is not None:
            rng = np.random.default_rng(self.random_state)
        else:
            rng = np.random.default_rng()
    
    def get_w(self, Xn, w_init):

        n_features = Xn.shape[1]
        if w_init is None:
            w = np.zeros(n_features)
        else:
            w = np.asarray(w_init, dtype=float).ravel()
            assert w.shape[0] == n_features, "w_init has wrong shape."
        
        return w

    def fit(self, X, y, w_init=None, b_init=0.0):
        self.set_random_seed()

        X, y = self._to_xy(X, y)
        
        Xn = self._standardize(X)
        w = self.get_w(Xn, w_init)
        b = float(b_init)

        self.history_ = []
        last_cost = None
        small_moves = 0

        for i in range(self.n_iters):
            dj_dw, dj_db = self._compute_gradient(Xn, y, w, b)
            w -= self.alpha * dj_dw
            b -= self.alpha * dj_db

            J = self._compute_cost(Xn, y, w, b)
            self.history_.append(J)

            if self.verbose and (i % max(1, self.n_iters // 10) == 0):
                delta = np.nan if last_cost is None else (J - last_cost)
                print(f"iter {i:4d} | cost {J:.6e} | Δ {delta if last_cost is not None else 0:.2e}")

            if self.tol > 0 and last_cost is not None:
                if abs(J - last_cost) < self.tol:
                    small_moves += 1
                    if small_moves >= 5:
                        if self.verbose:
                            print(f"Early stop at iter {i} (Δcost < {self.tol})")
                        break
                else:
                    small_moves = 0
            last_cost = J

        self.w_, self.b_ = w, b
        return self

    def _prepare_X(self, X):
        X = np.asarray(X, dtype=float)
        if self.normalize:
            if self.mu_ is None or self.sigma_ is None:
                raise ValueError("Model was trained with normalize=True but mu_/sigma_ are missing.")
            X = self._standardize_apply(X, self.mu_, self.sigma_)
        return X

    def predict(self, X):
        if self.w_ is None or self.b_ is None:
            raise ValueError("Call fit() before predict().")
        Xn = self._prepare_X(X)
        return Xn @ self.w_ + self.b_

    def get_params(self):
        return {
            "alpha": self.alpha,
            "n_iters": self.n_iters,
            "normalize": self.normalize,
            "tol": self.tol,
            "random_state": self.random_state,
            "verbose": self.verbose,
        }

    def set_params(self, **kwargs):
        for k, v in kwargs.items():
            if not hasattr(self, k):
                raise ValueError(f"Unknown parameter: {k}")
            setattr(self, k, v)
        return self


def load_data():
    house = pd.read_csv('https://github.com/YBIFoundation/Dataset/raw/main/Boston.csv')
    y = house['MEDV']
    X = house.drop(['MEDV'],axis=1)
    X_train, X_test, y_train, y_test = train_test_split(X,y,train_size=0.7, random_state=42)
    return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    X_train, X_test, y_train, y_test = load_data()

    model = LinearRegressorGD( 
        alpha=1e-2,
        n_iters=1000,
        normalize=True,
        tol=1e-8,
        random_state=42,
        verbose= True
    )

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print("MSE :", mean_squared_error(y_test, y_pred))
    print("RMSE:", root_mean_squared_error(y_test, y_pred))
    print("R²  :", r2_score(y_test, y_pred))

    # Inspect learned params & training curve
    print("w shape:", model.w_.shape, " b:", model.b_)