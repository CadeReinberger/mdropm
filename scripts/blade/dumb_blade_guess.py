import numpy as np
import matplotlib.pyplot as plt

beta_data = np.array([45, 40, 15, 0, -45])
alpha_data = np.array([23.2, 13, 11.7, 0, -23.08])

beta_theory = np.linspace(0, 60, 500)
b = np.deg2rad(beta_theory)
alpha_theory = np.rad2deg(b - 0.5 * np.arccos(np.tanh(np.log(2) + np.arctanh(np.cos(2 * b)))))

beta_theory = np.concatenate((-beta_theory[::-1], beta_theory))
alpha_theory = np.concatenate((-alpha_theory[::-1], alpha_theory))

fig, ax = plt.subplots(figsize=(7, 5))
ax.plot(beta_theory, alpha_theory, label='Theory', color='steelblue', linewidth=2)
ax.scatter(beta_data, alpha_data, label='Data', color='tomato', s=60, zorder=5)
ax.set_xlabel(r'$\beta$ (degrees)', fontsize=13)
ax.set_ylabel(r'$\alpha$ (degrees)', fontsize=13)
ax.set_title('Predicted vs Observed Offset Angles For Blade Coating', fontsize=12)
ax.legend(fontsize=11)
ax.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig('blade_angle.png', dpi=200)
print(f"Saved blade_angle.png")
