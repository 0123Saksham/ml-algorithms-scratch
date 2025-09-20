import numpy as np

X = np.loadtxt("/kaggle/input/dataset-4/q4x.dat")   

y_str = np.loadtxt("/kaggle/input/dataset-4/q4y.dat", dtype=str)

y = np.array([0 if label == "Alaska" else 1 for label in y_str])

X = (X - X.mean(axis=0)) / X.std(axis=0)

print("Unique labels in y:", np.unique(y))
print("First 10 y:", y[:10])


def gda_parameters(X, y):
    m, n = X.shape
    phi = np.mean(y)   
   
    mu0 = X[y == 0].mean(axis=0)
    mu1 = X[y == 1].mean(axis=0)
    
    diff0 = X[y == 0] - mu0
    diff1 = X[y == 1] - mu1
    Sigma = (diff0.T @ diff0 + diff1.T @ diff1) / m
    
    return phi, mu0, mu1, Sigma

phi, mu0, mu1, Sigma = gda_parameters(X, y)

print("phi =", phi)
print("mu0 =", mu0)
print("mu1 =", mu1)
print("Sigma =\n", Sigma)


def plot_data(X, y):
    plt.scatter(X[y==0, 0], X[y==0, 1], label="Alaska (0)", marker="o")
    plt.scatter(X[y==1, 0], X[y==1, 1], label="Canada (1)", marker="x")
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.legend()
    plt.grid(True)
    plt.title("Training Data")
    plt.show()

plot_data(X, y)


import matplotlib.pyplot as plt
import numpy as np

def plot_data(X, y):
    """Scatter plot of training data with labels"""
    plt.scatter(X[y==0, 0], X[y==0, 1], c='blue', marker='o', label="Alaska (0)")
    plt.scatter(X[y==1, 0], X[y==1, 1], c='red', marker='x', label="Canada (1)")
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.legend()

def plot_linear_boundary(X, y, mu0, mu1, Sigma):
    invSigma = np.linalg.inv(Sigma)
    theta = invSigma @ (mu1 - mu0)
    c = 0.5 * (mu0 @ invSigma @ mu0 - mu1 @ invSigma @ mu1)

    # Decision boundary: θ^T x + c = 0
    x_vals = np.linspace(X[:,0].min()-1, X[:,0].max()+1, 100)
    y_vals = -(theta[0]*x_vals + c) / theta[1]

    plot_data(X, y)

    plt.plot(x_vals, y_vals, 'k-', linewidth=2, label="Linear Boundary")

    plt.title("Gaussian Discriminant Analysis - Linear Decision Boundary")
    plt.legend()
    plt.show()
plot_linear_boundary(X, y, mu0, mu1, Sigma)


def gda_quadratic(X, y):
    mu0 = X[y==0].mean(axis=0)
    mu1 = X[y==1].mean(axis=0)
    
    diff0 = X[y==0] - mu0
    diff1 = X[y==1] - mu1
    Sigma0 = (diff0.T @ diff0) / len(diff0)
    Sigma1 = (diff1.T @ diff1) / len(diff1)
    
    return mu0, mu1, Sigma0, Sigma1

mu0_q, mu1_q, Sigma0, Sigma1 = gda_quadratic(X, y)

print("mu0 =", mu0_q)
print("mu1 =", mu1_q)
print("Sigma0 =\n", Sigma0)
print("Sigma1 =\n", Sigma1)

def plot_quadratic_boundary(X, y, mu0, mu1, Sigma0, Sigma1):
    inv0 = np.linalg.inv(Sigma0)
    inv1 = np.linalg.inv(Sigma1)
    logdet0 = np.log(np.linalg.det(Sigma0))
    logdet1 = np.log(np.linalg.det(Sigma1))
    x1, x2 = np.meshgrid(
        np.linspace(X[:,0].min()-2, X[:,0].max()+2, 200),
        np.linspace(X[:,1].min()-2, X[:,1].max()+2, 200)
    )
    grid = np.c_[x1.ravel(), x2.ravel()]

    def quad_term(x, mu, invSigma):
        d = x - mu
        return np.sum(d @ invSigma * d, axis=1)

    f0 = quad_term(grid, mu0, inv0) + logdet0
    f1 = quad_term(grid, mu1, inv1) + logdet1
    boundary = (f0 - f1).reshape(x1.shape)

    plt.scatter(X[y==0, 0], X[y==0, 1], c='blue', marker='o', label="Alaska (0)")
    plt.scatter(X[y==1, 0], X[y==1, 1], c='red', marker='x', label="Canada (1)")

    Sigma_avg = (Sigma0 + Sigma1) / 2
    invSigma_avg = np.linalg.inv(Sigma_avg)
    theta = invSigma_avg @ (mu1 - mu0)
    c = 0.5 * (mu0 @ invSigma_avg @ mu0 - mu1 @ invSigma_avg @ mu1)
    x_vals = np.linspace(X[:,0].min()-2, X[:,0].max()+2, 100)
    y_vals = -(theta[0]*x_vals + c) / theta[1]
    plt.plot(x_vals, y_vals, 'k-', linewidth=2, label="Linear Boundary")

    plt.contour(x1, x2, boundary, levels=[0], colors="r", linewidths=2)

    plt.title("Quadratic Boundary (red) + Linear (black)")
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.legend()
    plt.show()
plot_quadratic_boundary(X, y, mu0_q, mu1_q, Sigma0, Sigma1)

