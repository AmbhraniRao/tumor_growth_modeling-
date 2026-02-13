import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import curve_fit

# True parameters (unknown in real life)
r_true = 0.35
K_true = 1200
c_true = 0.25
N0 = 20

t_data = np.linspace(0, 40, 15)


def logistic_growth(t, N, r, K):
    return r * N * (1 - N / K)


def treatment_model(t, N, r, K, c):
    return r * N * (1 - N / K) - c * N


def simulate(model, params, t):
    sol = solve_ivp(model, (t[0], t[-1]), [N0], args=params, t_eval=t)
    return sol.y[0]


# Generate synthetic noisy data
untreated_clean = simulate(logistic_growth, (r_true, K_true), t_data)
treated_clean = simulate(treatment_model, (r_true, K_true, c_true), t_data)

noise_level = 50
untreated_data = untreated_clean + np.random.normal(0, noise_level, len(t_data))
treated_data = treated_clean + np.random.normal(0, noise_level, len(t_data))


def logistic_solution(t, r, K):
    return simulate(logistic_growth, (r, K), t)


def treatment_solution(t, r, K, c):
    return simulate(treatment_model, (r, K, c), t)


popt_log, _ = curve_fit(logistic_solution, t_data, untreated_data, p0=[0.3, 1000])
r_fit, K_fit = popt_log

popt_treat, _ = curve_fit(treatment_solution, t_data, treated_data, p0=[0.3, 1000, 0.2])
r_treat, K_treat, c_fit = popt_treat


t_smooth = np.linspace(0, 40, 400)
log_sim = logistic_solution(t_smooth, r_fit, K_fit)
treat_sim = treatment_solution(t_smooth, r_treat, K_treat, c_fit)


plt.figure(figsize=(10,6))
plt.scatter(t_data, untreated_data, label="Untreated Data")
plt.scatter(t_data, treated_data, label="Treated Data")
plt.plot(t_smooth, log_sim, label="Fitted Logistic")
plt.plot(t_smooth, treat_sim, label="Fitted Treatment Model")
plt.xlabel("Time")
plt.ylabel("Tumor Size")
plt.legend()
plt.grid(True)
plt.show()


print("True parameters:")
print("r =", r_true, "K =", K_true, "c =", c_true)

print("\nEstimated parameters:")
print("Untreated → r =", r_fit, "K =", K_fit)
print("Treated → r =", r_treat, "K =", K_treat, "c =", c_fit)

