"""
The implementation of the paper https://dl.acm.org/doi/pdf/10.1145/3394486.3403374
Multi-arm multi-objective contextual bandits
The feature vectors x_{t,a} is generated by Gaussian distribution
"""
import numpy as np
import matplotlib.pyplot as plt
from misc import projection_simplex_bisection

class MOCB:
    def __init__(self, K, D, M, T, lam, eta, I):
        """
        :param K: number of bandits
        :param D: dimension of rewards
        :param M: dimension of features
        :param T: maximum rounds
        :param lam: weighting parameter for A
        :param eta: step-size for GD
        :param I: iteration number for GD
        """
        self.K = K
        self.D = D
        self.M = M
        self.T = T
        self.lam = lam
        self.eta = eta
        self.I = I
        self.w = np.power(2, np.linspace(0, -D+1, num=D))

    def GGI(self, alpha, Theta, X):
        """
        Calculate the generalized gini index

        :param alpha: input
        :return: GGI
        """
        x = np.zeros(self.D)
        for k in range(self.K):
            x += alpha[k] * Theta.T.dot(X[k])

        return np.sort(x).dot(self.w)

    def grad(self, alpha, Theta_hat, X):
        """
        Computes the gradient of the GGI

        :param Theta_hat:
        :param X:
        :return: grad of GGI
        """
        x = np.zeros(self.D)
        for k in range(self.K):
            x += alpha[k] * Theta_hat.T.dot(X[k])
        order = np.argsort(x)

        gradient = np.zeros(self.K)
        for k in range(self.K):
            temp = Theta_hat.T.dot(X[k])
            gradient[k] = self.w.dot(temp[order])

        return np.array(gradient)

    def grad_descent(self, alpha, Theta_hat, X):
        for i in range(self.I):
            alpha = projection_simplex_bisection(alpha + self.eta * self.grad(alpha, Theta_hat, X))

        return alpha

    def mirror_descent(self, alpha, Theta_hat, X):
        for i in range(self.I):
            temp = np.multiply(alpha, np.exp(self.eta * self.grad(alpha, Theta_hat, X)))
            alpha = temp / np.sum(temp)

        return alpha

    def gaussian_test(self):
        """
        Conduct the test based on Gaussian distribution

        :return: GGI of average rewards
        """
        print("Starting Gaussian test on MO-LinUCB")
        A = self.lam * np.identity(self.M)
        B = np.zeros((self.M, self.D))
        X = {}  # set of features
        Theta = np.random.uniform(0, 1, (self.M, self.D))  # the true theta
        GGI_list = [0] * self.T
        alpha = 1 / self.K * np.ones(self.K)  # initial alpha
        r_ave = np.zeros(self.D)
        for t in range(self.T):
            Theta_hat = np.linalg.inv(A).dot(B)

            for k in range(self.K):
                X[k] = np.random.normal(1/self.M, 1/self.M, self.M)

            alpha = self.mirror_descent(alpha, Theta_hat, X)  # gradient steps

            k = np.argmax(alpha)

            mu = X[k].T.dot(Theta)  # The mean reward
            r = mu + np.array([np.random.normal(0, 0.1 * abs(mu[d])) for d in range(self.D)])  # The reward
            r_ave += r

            A += X[k].dot(X[k].T)
            B += X[k].reshape(self.M, 1).dot(r.reshape(1, self.D))

            GGI_list[t] = np.sort(r_ave / t).dot(self.w)

            if t % 50 == 0:
                print("Round ", t, ", GGI: ", GGI_list[t], ", Theta_hat - Theta: ", np.linalg.norm(Theta_hat - Theta))

        # plots
        plt.plot(GGI_list)
        plt.show()



if __name__=="__main__":
    #           K, D, M, T, lam, eta, I
    mocb = MOCB(50, 5, 10, 10000, 0.1, 1, 10)
    mocb.gaussian_test()