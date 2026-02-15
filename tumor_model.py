import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import curve_fit
np.random.seed(42)
N0 = 20
r_true = 0.35
K_true = 1200
c_true = 0.25
t_data = np.linspace(0, 40, 15)
def logistic(t, N, r, K):
    return r * N * (1 - N / K)
def logistic_treatment(t, N, r, K, c):
    return r * N * (1 - N / K) - c * N
def simulate(model, params, t_points):
    sol = solve_ivp(model, (t_points[0], t_points[-1]), [N0], args=params, t_eval=t_points)
    return sol.y[0]
untreated_clean = simulate(logistic, (r_true, K_true), t_data)
treated_clean = simulate(logistic_treatment, (r_true, K_true, c_true), t_data)
noise = 50
untreated_data = untreated_clean + np.random.normal(0, noise, len(t_data))
treated_data = treated_clean + np.random.normal(0, noise, len(t_data))
def logistic_fit(t, r, K):
    return simulate(logistic, (r, K), t)
def treatment_fit(t, r, K, c):
    return simulate(logistic_treatment, (r, K, c), t)
popt1, _ = curve_fit(logistic_fit, t_data, untreated_data, p0=[0.3, 1000])
r_fit, K_fit = popt1
popt2, _ = curve_fit(treatment_fit, t_data, treated_data, p0=[0.3, 1000, 0.2])
r_treat, K_treat, c_fit = popt2
t_smooth = np.linspace(0, 40, 300)
plt.figure(figsize=(10, 6))
plt.scatter(t_data, untreated_data, label="Untreated data")
plt.scatter(t_data, treated_data, label="Treated data")
plt.plot(t_smooth, logistic_fit(t_smooth, r_fit, K_fit), label="Fitted logistic model")
plt.plot(t_smooth, treatment_fit(t_smooth, r_treat, K_treat, c_fit), label="Fitted treatment model")
plt.xlabel("Time")
plt.ylabel("Tumor size")
plt.legend()
plt.grid(True)
plt.show()
print("True parameters:")
print("r =", r_true, "K =", K_true, "c =", c_true)
print("\nEstimated parameters:")
print("Untreated: r =", r_fit, "K =", K_fit)
print("Treated: r =", r_treat, "K =", K_treat, "c =", c_fit)

