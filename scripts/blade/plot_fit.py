import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from blade_data import angles, mean, std

angles_arr = np.array(angles, dtype=float)
mean_arr = np.array(mean, dtype=float)
std_arr = np.array(std, dtype=float)

angles_rad = np.deg2rad(angles_arr)

def model(angle_rad, A):
    beta = np.abs(angle_rad)
    alpha = beta - 0.5 * np.arccos(np.tanh(np.log(2) + np.arctanh(np.cos(2 * beta))))
    res = np.sign(angle_rad) * np.sin(2*alpha)
    return A * res

popt, pcov = curve_fit(model, angles_rad, mean_arr)
A_fit = popt[0]

fit_angles_deg = np.linspace(-45, 45, 300)
fit_angles_rad = np.deg2rad(fit_angles_deg)
fit_y = model(fit_angles_rad, A_fit)

fig, ax = plt.subplots(figsize=(8, 6))

# 95% confidence interval (1.96 * std)
ax.errorbar(
    angles_arr, mean_arr,
    yerr=1.96 * std_arr,
    fmt='o', color='steelblue', capsize=6, capthick=2,
    markersize=8, linewidth=2, label='Data (95% CI)'
)

ax.plot(fit_angles_deg, fit_y, color='tomato', linewidth=2.5,
        label=f'Predicted')

ax.set_xlabel('Blade Angle (°)', fontsize=16)
ax.set_ylabel('g-factor', fontsize=16)
ax.tick_params(axis='both', labelsize=14)
ax.legend(fontsize=14)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('fitted_blade_chirality.png', dpi=200)
print(f"Saved fitted_blade_chirality.png  (A = {A_fit:.6f})")
