import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# Exponential Growth Model
def exponential_growth(t, N, r):
    return r * N

# Logistic Growth Model
def logistic_growth(t, N, r, K):
    return r * N * (1 - N / K)

# Parameters
r = 0.3          # growth rate
K = 1000         # carrying capacity
N0 = [10]        # initial tumor size
t_span = (0, 50)
t_eval = np.linspace(0, 50, 500)

# Solve Exponential Model
exp_solution = solve_ivp(
    exponential_growth,
    t_span,
    N0,
    args=(r,),
    t_eval=t_eval
)

# Solve Logistic Model
log_solution = solve_ivp(
    logistic_growth,
    t_span,
    N0,
    args=(r, K),
    t_eval=t_eval
)

# Plot Results
plt.figure()
plt.plot(exp_solution.t, exp_solution.y[0], label="Exponential Growth")
plt.plot(log_solution.t, log_solution.y[0], label="Logistic Growth")
plt.xlabel("Time")
plt.ylabel("Tumor Cell Population")
plt.legend()
plt.title("Tumor Growth Modeling Using ODEs")
plt.show()
