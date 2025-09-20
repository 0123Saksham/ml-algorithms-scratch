import numpy as np
from sklearn.model_selection import train_test_split

N = 1_000_000  
theta_true = np.array([3, 1, 2])  

x1 = np.random.normal(3, 4, N)
x2 = np.random.normal(-1, 4, N)

X = np.column_stack((np.ones(N), x1, x2))

noise = np.random.normal(0, np.sqrt(2), N)

y = X @ theta_true + noise

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=42)

print("Train shape:", X_train.shape, y_train.shape)
print("Test shape:", X_test.shape, y_test.shape)


def compute_cost(X, y, theta):
    m = len(y)
    residuals = X @ theta - y
    return (1 / (2*m)) * np.sum(residuals**2)


import numpy as np

def sgd4(x, y, theta_init, lr, batch_size, max_epochs, eps_c=1e-4, patience=2, verbose=True):
    """
    SGD with moving-average gradient convergence criterion.
    Stores cost per iteration (not per epoch).
    """
    
    n, d = x.shape
    theta = theta_init.copy()
    
    K = max(1, n // batch_size)
    
    grad_history = []
    prev_avg_grad = None
    consecutive = 0
    cost_history = []   
    theta_path = [theta.copy()]
    check = False
    iteration = 0
    
    for epoch in range(max_epochs):
        indices = np.random.permutation(n)
        x_shuff, y_shuff = x[indices], y[indices]
        
        for i in range(0, n, batch_size):
            xb = x_shuff[i:i+batch_size]
            yb = y_shuff[i:i+batch_size]
            
            g = (2 / xb.shape[0]) * xb.T @ (xb @ theta - yb)
            
            theta = theta - lr * g
            
            grad_history.append(g)
            if len(grad_history) > K:
                grad_history.pop(0)
            
            avg_grad = np.mean(grad_history, axis=0)
            

            if prev_avg_grad is not None:
                diff = np.linalg.norm(avg_grad - prev_avg_grad)
                if diff <= eps_c:
                    consecutive += 1
                else:
                    consecutive = 0
                if consecutive >= patience:
                    iter_loss = compute_cost(x,y,theta)
                    cost_history.append(iter_loss)
                    theta_path.append(theta.copy())
    
                    if verbose:
                        print(f"Converged at epoch {epoch+1}, iter {iteration}, diff={diff:.6f}")
                    
                    return theta, cost_history, np.array(theta_path), iteration, epoch+1
            
            prev_avg_grad = avg_grad
            
            iter_loss = compute_cost(x, y,theta)
            cost_history.append(iter_loss)
            theta_path.append(theta.copy())
            iteration += 1
            if iteration == 400000:
                check = True
                break
        if check == True:
            break
    
    return theta, cost_history, np.array(theta_path),iteration, max_epochs


from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
batch_sizes = [8000]
def plot_theta_movement(x, y, batch_sizes, lr, max_epochs):
    """
    Plot the movement of theta in 3D parameter space for varying batch sizes.
    Assumes sgd3 returns (theta, cost_history, theta_path).
    """
    fig = plt.figure(figsize=(15, 5))

    for i, batch_size in enumerate(batch_sizes):
        # Run SGD
        theta, cost_history, theta_path, iterations,epoch = sgd4(X_train, y_train,theta_init=np.zeros(x.shape[1]),lr=0.001,batch_size = batch_size, max_epochs=40000)

        print(f"Batch size {batch_size}: Final theta = {theta}, iterations: = {iterations}, epochs={epoch},Final cost = {cost_history[-1]}")

        theta_path = np.array(theta_path)  
        
        # Ensure only first 3 parameters are plotted
        if theta_path.shape[1] < 3:
            raise ValueError("Need at least 3 parameters (theta) for 3D plotting")

        ax = fig.add_subplot(1, len(batch_sizes), i+1, projection="3d")
        ax.plot(theta_path[:,0], theta_path[:,1], theta_path[:,2], marker='o', markersize=1, linewidth=0.5)
        ax.scatter(theta_path[0,0], theta_path[0,1], theta_path[0,2], color="red", label="Start")
        ax.scatter(theta_path[-1,0], theta_path[-1,1], theta_path[-1,2], color="green", label="Converged")
        ax.set_title(f"Batch size = {batch_size}")
        ax.set_xlabel(r"$\theta_0$")
        ax.set_ylabel(r"$\theta_1$")
        ax.set_zlabel(r"$\theta_2$")
        ax.legend()

    plt.tight_layout()
    plt.show()
plot_theta_movement(X_train, y_train, batch_sizes, lr, max_epochs=40000)


# theta_true, cost_history = sgd(X_train, y_train, theta_init, eta, 8000, max_epochs=50)
XT_X = X_train.T @ X_train
XT_y = X_train.T @ y_train
theta_closed = np.linalg.solve(XT_X, XT_y)

print("\n=== Parameters ===")
print("True θ        :", theta_true)
print("Closed-form θ :", theta_closed)
# print("Final Cost: ", cost_history[-1])

def mse(X, y, theta):
    return np.mean((y - X @ theta)**2)

train_mse_closed = mse(X_train, y_train, theta_closed)
test_mse_closed  = mse(X_test,  y_test,  theta_closed)

train_J_closed = compute_cost(X_train, y_train, theta_closed)
test_J_closed  = compute_cost(X_test,  y_test,  theta_closed)

print("Closed-form:")
print(" theta_closed:", theta_closed)
print(f"  Train MSE = {train_mse_closed:}, Train J = {train_J_closed:}")
print(f"   Test MSE = {test_mse_closed:},  Test J = {test_J_closed:}")
print()
