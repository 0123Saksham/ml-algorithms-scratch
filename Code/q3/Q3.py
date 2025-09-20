import numpy as np

X = np.loadtxt("/kaggle/input/data-set2/logisticX.csv", delimiter=",")   
y = np.loadtxt("/kaggle/input/data-set2/logisticY.csv", delimiter=",")   

m, d = X.shape
assert d == 2, "Expected 2 features"

X_mean = X.mean(axis=0)
X_std  = X.std(axis=0, ddof=0)
X_std[X_std == 0] = 1.0

X_norm = (X - X_mean) / X_std

X_b = np.c_[np.ones((m,1)), X_norm]   # shape (m, 3)

def sigmoid(z):
    z = np.asarray(z)
    out = np.empty_like(z, dtype=float)
    pos = z >= 0
    neg = ~pos
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    expz = np.exp(z[neg])
    out[neg] = expz / (1.0 + expz)
    return out

def newton_logistic(X_b, y, max_iter=100, tol=1e-6, verbose=True):
    m, n = X_b.shape
    theta = np.zeros(n)   
    eps = 1e-12

    for it in range(max_iter):
        z = X_b @ theta          
        h = sigmoid(z)           
        grad = X_b.T @ (y - h)  

        S = h * (1 - h)         

        XT_S_X = X_b.T @ (S[:, None] * X_b)   

        try:
            delta = np.linalg.solve(XT_S_X, grad)
        except np.linalg.LinAlgError:
            delta = np.linalg.pinv(XT_S_X) @ grad

        theta_new = theta + delta   # IRLS step

        if np.linalg.norm(theta_new - theta, ord=2) < tol:
            theta = theta_new
            if verbose:
                print(f"Converged after {it+1} iterations (||Δθ|| < {tol}).")
            break

        theta = theta_new

    final_z = X_b @ theta
    final_h = sigmoid(final_z)
    final_S = final_h * (1 - final_h)
    Hessian_loglik = - (X_b.T @ (final_S[:, None] * X_b))

    # final log-likelihood
    ll = np.sum(y * np.log(final_h + eps) + (1 - y) * np.log(1 - final_h + eps))

    return theta, Hessian_loglik, ll

theta_hat, H_loglik, loglik_val = newton_logistic(X_b, y, max_iter=100, tol=1e-8, verbose=True)

print("Estimated coefficients (theta) [intercept, theta1, theta2] (on normalized features):")
print(theta_hat)
print("\nLog-likelihood at solution:", loglik_val)
print("\nHessian of log-likelihood at solution (H):")
print(H_loglik)


import matplotlib.pyplot as plt

# Decision boundary: θ0 + θ1*x1 + θ2*x2 = 0  => x2 = - (θ0 + θ1*x1) / θ2

theta0, theta1, theta2 = theta_hat

x1_min, x1_max = X_norm[:,0].min() - 0.5, X_norm[:,0].max() + 0.5
x1_vals = np.linspace(x1_min, x1_max, 200)
x2_vals = - (theta0 + theta1 * x1_vals) / (theta2 + 1e-16)

plt.figure(figsize=(8,6))
plt.scatter(X_norm[y==0,0], X_norm[y==0,1], marker='o', facecolors='none', edgecolors='C0', label='y=0')
plt.scatter(X_norm[y==1,0], X_norm[y==1,1], marker='x', color='C1', label='y=1')

plt.plot(x1_vals, x2_vals, 'k-', linewidth=2, label='Decision boundary (h=0.5)')

plt.xlabel("x1 (normalized)")
plt.ylabel("x2 (normalized)")
plt.title("Logistic Regression: data and decision boundary (normalized features)")
plt.legend()
plt.grid(True)
plt.show()
