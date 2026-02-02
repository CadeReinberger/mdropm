# =============================================================================
# A note on the MMP map orientation:  By Default matplotlib sets the (0,0) 
# coordinate to the bottom left. However the beamline maps from the top left.
# This has the effect to horizontally mirror the mapping.  Rather than resetting
# the coordinate in the plot and then rotating 90 degrees, the numpy array has
# just been transposed prior to plotting
# =============================================================================

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d#, RectBivariateSpline
from scipy.ndimage import map_coordinates
import pickle as pkl

# border_width = 3
# labeflsize = 24
# ticklabelsize = 20
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ['Arial']
# g_range = 0.15

# Load data from CSV file
# Option 1: CSV with columns [x, y, value]
# Uncomment and modify the filename:
# import pandas as pd
# df = pd.read_csv('your_data.csv')
# x_data = df['x'].values  # or use column index df.iloc[:, 0]
# y_data = df['y'].values
# z_data = df['value'].values
# 
# # Create regular grid (if data is not already gridded)
# from scipy.interpolate import griddata
# x = np.linspace(x_data.min(), x_data.max(), 100)
# y = np.linspace(y_data.min(), y_data.max(), 100)
# X, Y = np.meshgrid(x, y)
# Z = griddata((x_data, y_data), z_data, (X, Y), method='cubic')

# Option 2: CSV is already a 2D grid (rows=y, columns=x)
# Uncomment and modify:
#Z = np.loadtxt('Tom_Science_symmetric_linear_transposed?.csv', delimiter = ',')#'Tom_Science_symmetric_linear_transposed?.csv', delimiter=',')

Z = None
with open('data.pkl', 'rb') as file:
    Z = pkl.load(file)
    Z = np.flipud(Z)
    print('We got Z or something jfc')

x = np.linspace(0, 4, Z.shape[1])  # Adjust range as needed
y = np.linspace(0, 4, Z.shape[0])


#%%
# Example: Create sample heatmap data
# x = np.linspace(0, 10, 100)
# y = np.linspace(0, 10, 100)
# X, Y = np.meshgrid(x, y)
# Z = np.sin(X) * np.cos(Y) + 0.5 * np.exp(-((X-5)**2 + (Y-5)**2) / 4)

# Define a curved path using control points
# Method 1: Simple interpolation through points
control_points = np.array([
    [0.0, 3.4],
    [2.0, 3.90],
    # [1, 2],
    # [2, 2],
    [4.0, 2.9]
]) - np.array([0, .4])

# Create smooth curve through control points

t = np.linspace(0, 1, len(control_points))
t_fine = np.linspace(0, 1, 500)  # More points for smooth curve

fx = interp1d(t, control_points[:, 0], kind='quadratic')
fy = interp1d(t, control_points[:, 1], kind='quadratic')

curve_x = fx(t_fine)
curve_y = fy(t_fine)

# # Calculate distance along curve for x-axis
# distances = np.zeros(len(curve_x))
# for i in range(1, len(curve_x)):
#     dx = curve_x[i] - curve_x[i-1]
#     dy = curve_y[i] - curve_y[i-1]
#     distances[i] = distances[i-1] + np.sqrt(dx**2 + dy**2)

def create_scans_from_path(ycurve):
    # Sample the heatmap along the curve
    # Convert curve coordinates to array indices
    x_idx = np.interp(curve_x, x, np.arange(len(x)))
    y_idx = np.interp(ycurve, y, np.arange(len(y)))
    
    # Extract values using map_coordinates (handles interpolation)
    coords = np.array([y_idx, x_idx])  # Note: row, col order
    line_scan = map_coordinates(Z, coords, order=1)
    
    return line_scan

curves = {}
scans = {}
for i in range(4):
    curves['y'+str(i)] = curve_y - i*0.5
    scans['scan'+str(i)] = create_scans_from_path(curves['y'+str(i)])
    
    
#%%
#plotting parameters
border_width = 3
labeflsize = 28
colors = ['c','m','y','g']
g_range = 2 # 0.15
lwidth = 4


# # Plot results
# fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig,ax1 = plt.subplots(figsize=(7.5,6))
# Plot heatmap with curve overlay
im = ax1.imshow(Z, extent=[x.min(), x.max(), y.min(), y.max()],
                    cmap='seismic', aspect='equal', vmin = -g_range, vmax = g_range)
ax1.set_xticks([])
ax1.set_yticks([])
for location in ["top","bottom","left","right"]:
    ax1.spines[location].set_linewidth(border_width)
    
i=0
for curve in curves:
    ax1.plot(curve_x, curves[curve], color = colors[i], linewidth=lwidth)
    i+=1
# ax1.plot(control_points[:, 0], control_points[:, 1], 'co', 
#              markersize=8, label='Control points')
# ax1.set_xlabel('X')
# ax1.set_ylabel('Y')
# ax1.set_title('Heatmap with Curved Scan Path')
cbar = fig.colorbar(im, pad = 0.02)
cbar.outline.set_linewidth(border_width)
cbar.ax.tick_params(width = border_width, length = 6)
cbar.ax.set_yticklabels([])

# cbar.set_label('g-factor', fontsize = labeflsize)


plt.tight_layout()
plt.show()

#%%

# Plot line scan
fig,ax2 = plt.subplots(figsize=(6,5))

i=0
for scan in scans:
    ax2.plot(curve_x, scans[scan], color = colors[i], linewidth=lwidth)
    i+=1
    
ax2.set(xlim = [0,4], ylim = [-g_range,g_range])
ax2.set_xticks([0,1,2,3,4])
ax2.yaxis.tick_right()
ax2.set_xticklabels([])
ax2.set_yticklabels([])
# ax2.set_xlabel('Distance along curve')
# ax2.set_ylabel('Value')
# ax2.set_title('Line Scan Profile')
# ax2.grid(True, alpha=0.3)
for location in ["top","bottom","left","right"]:
    ax2.spines[location].set_linewidth(border_width)
ax2.tick_params(width = border_width, length = 6)

plt.tight_layout()
plt.show()

#%%
# # Optional: Use parametric equations for more complex curves
# # Example: Sinusoidal curve
# def create_sinusoidal_path(amplitude=2, frequency=2, x_range=(1, 9), n_points=500):
#     x_curve = np.linspace(x_range[0], x_range[1], n_points)
#     y_curve = 5 + amplitude * np.sin(frequency * np.pi * (x_curve - x_range[0]) / (x_range[1] - x_range[0]))
#     return x_curve, y_curve

# # Uncomment to use sinusoidal path instead:
# # curve_x, curve_y = create_sinusoidal_path()
