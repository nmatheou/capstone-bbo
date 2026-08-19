import numpy as np
import matplotlib.pyplot as plt

# Load F8 data
X = np.load(r'C:\Users\NtecD\OneDrive\AI\Imperial\Capstone\antigravity\next_week_data\function_8\initial_inputs.npy')
y = np.load(r'C:\Users\NtecD\OneDrive\AI\Imperial\Capstone\antigravity\next_week_data\function_8\initial_outputs.npy')

# Create a 2x4 grid of subplots
fig, axes = plt.subplots(2, 4, figsize=(16, 8))
fig.suptitle('F8: Dimension vs Oracle Score (y)', fontsize=16)
dims = ['d1', 'd2', 'd3', 'd4', 'd5', 'd6', 'd7', 'd8']

for i, ax in enumerate(axes.flatten()):
    ax.scatter(X[:, i], y, alpha=0.7, color='blue', edgecolors='k')
    
    # Label titles based on our identified trends
    if i < 4:
        title = f'{dims[i]} (Push to 0)'
    elif i in [4, 7]:
        title = f'{dims[i]} (Push to 1)'
    else:
        title = f'{dims[i]} (Interior)'
        
    ax.set_title(title)
    ax.set_xlabel(dims[i])
    ax.set_ylabel('Oracle y')
    ax.grid(True, linestyle='--', alpha=0.6)

plt.tight_layout()
plt.show()
#plt.savefig('f8_slopes.png')
print("Plot saved to f8_slopes.png")
