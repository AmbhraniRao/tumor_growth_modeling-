import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import curve_fit

data = pd.read_csv("tumor_data.csv")

t_data = data["time"].values
untreated_data = data["untreated"].values
treated_data = data["treated"].values


def logistic_growth(t, N, r, K):
    return r * N * (1 - N / K)


def treatment_model(t, N, r, K, c):
    return r * N * (1 - N / K) - c * N


def logistic_solution(t, r, K):
    sol = solve_ivp(
        logistic_growth,
        (t[0], t[-1]),
        [untreated_data[0]],
        args=(r, K),
        t_eval=t
    )
    return sol.y[0]


def treatment_solution(t, r, K, c):
    sol = solve_ivp(
        treatment_model,
        (t[0], t[-1]),
        [treated_data[0]],
        args=(r, K, c),
        t_eval=t
    )
    return sol.y[0]


popt_log, _ = curve_fit(logistic_solution, t_data, untreated_data, p0=[0.3, 1000])
r_fit, K_fit = popt_log

popt_treat, _ = curve_fit(treatment_solution, t_data, treated_data, p0=[0.3, 1000, 0.1])
r_treat, K_treat, c_fit = popt_treat


t_smooth = np.linspace(t_data[0], t_data[-1], 500)

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


print("Untreated Model Parameters:")
print("r =", r_fit)
print("K =", K_fit)

print("\nTreatment Model Parameters:")
print("r =", r_treat)
print("K =", K_treat)
print("c =", c_fit)

