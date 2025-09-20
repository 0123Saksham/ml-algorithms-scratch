import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

X = pd.read_csv("linearX.csv",header = None).values
y = pd.read_csv("linearY.csv",header = None).values

m = X.shape[0]

X_b = np.c_[np.ones((m,1)),X]

theta = np.zeros((2,1))

alpha = 0.001 # learning rate (to be tuned)
tol = 1e-6    # tolerance for cost change
max_iter = 10000

def compute_cost(X, y ,theta):
    predictions = X.dot(theta)
    error = predictions - y
    return (1/(2*m)) * np.sum(error**2)

J_history = []
theta_history = []
for it in range(max_iter):
    gradients = (1/m) * X_b.T.dot(X_b.dot(theta) - y)
    theta = theta - alpha * gradients
    cost = compute_cost(X_b, y,theta)
    J_history.append(cost)
    theta_history.append(theta.copy())

    if it > 0 and abs(J_history[-2] - J_history[-1]) < tol:
        print(f"Converged after {it} iterations.")
        # new_max_itr = it
        break

print("Final theta:", theta.ravel())
print("Final cost:", J_history[-1])


plt.plot(J_history)
plt.xlabel("Iterations")
plt.ylabel("Cost J(θ)")
plt.title("Gradient Descent Convergence")
plt.savefig("grad_desc.jpg")
plt.show()

plt.scatter(X, y, color='blue',s=1, label="Training Data")
plt.plot(X, X_b.dot(theta), color='red', linewidth=1,label="Prediction Line")
plt.xlabel("Acidity (X)")
plt.ylabel("Density (y)")
plt.legend()
plt.savefig("Hypothesis.png")
plt.show()

theta_history = np.array(theta_history).reshape(-1, 2)

# ----- Part (a): Plot mesh (with fixed wider range) -----

theta0_vals = np.linspace(-10, 10, 100)
theta1_vals = np.linspace(-1, 4, 100)

T0, T1 = np.meshgrid(theta0_vals, theta1_vals)
J_vals = np.zeros_like(T0)

# Compute cost for each (theta0, theta1)
for i in range(len(theta0_vals)):
    for j in range(len(theta1_vals)):
        t = np.array([[T0[j, i]], [T1[j, i]]])  # shape (2,1)
        J_vals[j, i] = compute_cost(X_b, y, t)

# --- 3D surface plot ---
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(T0, T1, J_vals, cmap='viridis', alpha=0.8)

ax.set_xlabel(r"$\theta_0$")
ax.set_ylabel(r"$\theta_1$")
ax.set_zlabel(r"$J(\theta)$")
plt.title("Cost Function Surface (Convex Bowl)")
plt.show()
# # ----- Part (a): Plot mesh -----
# t0_min, t0_max = theta_history[:,0].min(), theta_history[:,0].max()
# t1_min, t1_max = theta_history[:,1].min(), theta_history[:,1].max()

# pad0 = 0.1 * (t0_max - t0_min) if t0_max > t0_min else 1.0
# pad1 = 0.1 * (t1_max - t1_min) if t1_max > t1_min else 1.0

# theta0_vals = np.linspace(t0_min - pad0, t0_max + pad0, 100)
# theta1_vals = np.linspace(t1_min - pad1, t1_max + pad1, 100)
# T0, T1 = np.meshgrid(theta0_vals, theta1_vals)
# J_vals = np.zeros_like(T0)

# for i in range(len(theta0_vals)):
#     for j in range(len(theta1_vals)):
#         t = np.array([[T0[j,i]], [T1[j,i]]])
#         J_vals[j,i] = compute_cost(X_b, y, t)

# fig = plt.figure(figsize=(10,7))

ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(T0, T1, J_vals, cmap='viridis', alpha=0.7)


# ----- Part (b): Animate gradient descent path -----
for i in range(len(theta_history)):
    th0, th1 = theta_history[i]
    J_val = J_history[i]
    ax.scatter(th0, th1, J_val, color='red', s=20)
    plt.pause(0.2)  

ax.set_xlabel('θ0')
ax.set_ylabel('θ1')
ax.set_zlabel('Cost J(θ)')
plt.title("Gradient Descent on Cost Function Surface")
plt.show()
plt.figure(figsize=(8,6))


CS = plt.contour(T0, T1, J_vals, levels=30, cmap="viridis")
plt.clabel(CS, inline=1, fontsize=8)

for i in range(len(theta_history)):
    plt.scatter(theta_history[i,0], theta_history[i,1], 
                c='r', marker='x')
    plt.title(f"Iteration {i+1}")
    plt.xlabel(r"$\theta_0$")
    plt.ylabel(r"$\theta_1$")
    plt.pause(0.2)   

plt.show()







# t0_min, t0_max = theta_history[:, 0].min(), theta_history[:, 0].max()
# t1_min, t1_max = theta_history[:, 1].min(), theta_history[:, 1].max()

# pad0 = 0.1 * (t0_max - t0_min) if t0_max > t0_min else 1.0
# pad1 = 0.1 * (t1_max - t1_min) if t1_max > t1_min else 1.0

# theta0_vals = np.linspace(t0_min - pad0, t0_max + pad0, 100)
# theta1_vals = np.linspace(t1_min - pad1, t1_max + pad1, 100)
# T0, T1 = np.meshgrid(theta0_vals, theta1_vals)
# J_vals = np.zeros_like(T0)

# for i in range(len(theta0_vals)):
#     for j in range(len(theta1_vals)):
#         t = np.array([[T0[j, i]], [T1[j, i]]])
#         J_vals[j, i] = compute_cost(X_b, y, t)

# # --- Contour Plot with Animation ---
# plt.figure(figsize=(8, 6))
# CS = plt.contour(T0, T1, J_vals, levels=30, cmap="viridis")
# plt.clabel(CS, inline=1, fontsize=8)

# for i in range(len(theta_history)):
#     plt.scatter(theta_history[i, 0], theta_history[i, 1],
#                 c='r', marker='x')
#     plt.title(f"Iteration {i+1}")
#     plt.xlabel(r"$\theta_0$")
#     plt.ylabel(r"$\theta_1$")
#     plt.pause(0.2)   # pause for animation

# plt.show()