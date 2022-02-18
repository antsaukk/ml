import math
import numpy as numpy
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

"""
ML-II from http://web4.cs.ucl.ac.uk/staff/D.Barber/textbook/200620.pdf page 383-384
"""
class BayesianLinearRegression:
    def __init__(self, X, y, trn_size=0.8, tst_size=0.5, nbf=11, lambd=0.17, ll_optim=True):
        self.size = len(X)
        self.nbf = nbf
        self.lambda_val = lambd
        self.ll_optim = ll_optim
        self.best_log_likelihood = -np.inf
        self.best_alpha = None
        self.best_beta = None
        self.best_mean = None
        self.lb = math.floor(np.min(X))
        self.ub = math.ceil(np.max(X))
        self.alphas = np.logspace(-3, 3, 100)
        self.betas = np.logspace(-3, 3, 100)
        self.centers = np.linspace(self.lb, self.ub, self.nbf)
        self.x_points = np.linspace(self.lb, self.ub, 100)
        self.unpack(X, y, trn_size, tst_size)

    def unpack(self, X, y, trn_size, tst_size):
        (X_train, X_rem, y_train, y_rem) = train_test_split(X, y, train_size=trn_size)
        (X_valid, X_test, y_valid, y_test) = train_test_split(X_rem, y_rem, test_size=tst_size)
        self.X_train = X_train
        self.y_train = y_train
        self.X_valid = X_valid
        self.y_valid = y_valid
        self.X_test = X_test
        self.y_test = y_test
        self.phi_X_train = self.radial_basis_functions(X_train)
        self.phi_X_valid = self.radial_basis_functions(X_valid)
        self.phi_X_test = self.radial_basis_functions(X_test)

    def radial_basis_functions(self, X):
        X_centered = X[:,np.newaxis] - self.centers[np.newaxis,:]
        y = np.exp(-0.5 * (np.square(X_centered)) / self.lambda_val)
        return y

    def marginal_log_likelihood(self, y, N, M, S, d, alpha, beta):
        MLL = 0.5 * (-beta * np.dot(y, y) + d @ S @ d + np.log(np.linalg.det(2 * np.pi * S)) +
                            M * np.log(alpha) + N * np.log(beta) - N * np.log(2 * np.pi))
        return MLL

    def b_regression(self, Phi_X, y, alpha, beta):
        N, M = Phi_X.shape

        S_inv = beta * np.dot(Phi_X.T, Phi_X) + alpha * np.eye(M)
        S = np.linalg.inv(S_inv)
        m = beta * np.dot(S, np.dot(Phi_X.T, y))


        d = beta * np.dot(Phi_X.T, y)
        log_likelihood = self.marginal_log_likelihood(y, N, M, S, d, alpha, beta)
        return m, S, log_likelihood

    def log_likelihood_grid_search_optimization(self):
        # Grid search optimization
        for alpha in self.alphas:
            for beta in self.betas:
        
                _, _, lg_like_t = self.b_regression(self.phi_X_valid, self.y_valid, alpha, beta)
        
                if lg_like_t > self.best_log_likelihood:
                    self.best_alpha = alpha
                    self.best_beta = beta
                    self.best_log_likelihood = lg_like_t

                    self.best_mean, _, _ = self.b_regression(self.phi_X_train, self.y_train, self.best_alpha, self.best_beta)

    def mse_grid_search_optimization(self):
        mse_valid = np.zeros((len(self.alphas), len(self.betas)))
        for a, alpha in enumerate(self.alphas):
            for b, beta in enumerate(self.betas):

                mu_, _, _ = self.b_regression(self.phi_X_train, self.y_train, alpha, beta)
                y_pred_valid = np.dot(self.phi_X_valid, mu_)
                mse_valid[a][b] = self.mse(y_pred_valid, self.y_valid)

        ai, bi = np.unravel_index(mse_valid.argmin(), mse_valid.shape)
        assert(mse_valid[ai][bi] == np.min(mse_valid))
        self.best_alpha = self.alphas[ai]
        self.best_beta = self.betas[bi]

        x_train_valid = np.concatenate((self.X_train, self.X_valid))
        y_train_valid = np.concatenate((self.y_train, self.y_valid))
        phi_X_train_valid = self.radial_basis_functions(x_train_valid)
        self.best_mean, _, _ = self.b_regression(phi_X_train_valid, y_train_valid, self.best_alpha, self.best_beta)


    def predict(self):
        self.y_mean = np.dot(self.radial_basis_functions(self.x_points), self.best_mean)
        self.y_pred = np.dot(self.phi_X_test, self.best_mean)

    def mse(self, y_pred, y_test):
        return np.mean(np.square(y_pred - y_test))

    def display(self):
        plt.plot(self.x_points, self.y_mean, label="learned model")
        plt.plot(self.X_train, self.y_train, 'kx', label="training data")
        plt.plot(self.X_test, self.y_test, 'rx', label="testing data")
        plt.plot(self.X_test, self.y_pred, 'gx', label="testing predictions")
        plt.legend()
        plt.title("ML-II: $\\alpha$=%.3f, $\\beta$=%.3f, mse=%.4f" % (self.best_alpha, self.best_beta, self.mse_test))
        plt.show()

    def run(self):
        if self.ll_optim:
            self.log_likelihood_grid_search_optimization()
        else:
            self.mse_grid_search_optimization()
        self.predict()
        self.mse_test = self.mse(self.y_pred, self.y_test)
        self.display()