# Import matplotlib and numpy libraries
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.image as mpimg

# Import ImageGrid class
from mpl_toolkits.axes_grid1 import ImageGrid

import pickle
import matplotlib.pyplot as plt
import numpy as np



pickle_file = "outputs/interpretations/gridsearch/grid_search_images.pickle"
with open(pickle_file, 'rb') as f:
    grid_search_images = pickle.load(f)

data = grid_search_images

lr_params = np.unique([item[4] for item in data])
wd_params = np.unique([item[5] for item in data])

# reverse lr_params
lr_params = lr_params[::-1]

print(lr_params)
print(wd_params)

# Create a grid of subplots
num_lrs = len(lr_params)
num_wds = len(wd_params)

fig, axes = plt.subplots(num_lrs, num_wds, figsize=(12, 8), sharex='all', sharey='all')

# Iterate through data and populate the grid
for lr_idx, lr in enumerate(lr_params):
    for wd_idx, wd in enumerate(wd_params):
        image_data = [item[0] for item in data if item[4] == lr and item[5] == wd]
        if image_data:
            image_path = image_data[0]  # Select the first image for the given lr and wd
            image = plt.imread(image_path)
            axes[lr_idx, wd_idx].imshow(image)

        # Set axis labels
        if lr_idx == num_lrs - 1:
            axes[lr_idx, wd_idx].set_xlabel(f'wd = {wd}')
        if wd_idx == 0:
            axes[lr_idx, wd_idx].set_ylabel(f'lr = {lr}')

        # Remove ticks and labels
        axes[lr_idx, wd_idx].set_xticks([])
        axes[lr_idx, wd_idx].set_yticks([])

# Add a big title at the top
# plt.suptitle('Grid of Images with Hyperparameter Combinations', fontsize=16)

# Adjust layout
# plt.tight_layout(rect=[0, 0.03, 1, 0.95])

# Show the plot
plt.show()

# Show the figure
# plt.show()
# Save the figure
plt.savefig('outputs/interpretations/gridsearch/grid_search_images.png')