"""

"""

from dataset import train_dataloader, test_dataloader, split_train, split_val
import matplotlib.pyplot as plt 
import torchvision
import numpy as np


import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


dataloader = split_train(64)

# Get a single batch from the training loader
data_iter = iter(dataloader)
images, labels = next(data_iter)

# Select the first image from the batch
single_image = images[0]
single_label = labels[0]

# Print size information
print(f"Image tensor shape: {single_image.shape}")
print(f"Image dimensions: {single_image.shape[1]} x {single_image.shape[2]} pixels")
print(f"Number of channels: {single_image.shape[0]}")
print(f"Label: {single_label.item()}")

print(single_image.max())
print(single_image.min())

# Plot the image
plt.figure()
image_data = single_image.squeeze().numpy()

# image_min, image_max = image_data.min(), image_data.max()
# if image_max > image_min:
#     image_norm = (image_data - image_min) / (image_max - image_min)
# else:
#     image_norm = image_data  # In case it's a flat image

plt.imshow(image_data, cmap='gray')

# Show axis with pixel coordinates
plt.title(f"Label: {single_label.item()}")
plt.xlabel('Width (pixels)')
plt.ylabel('Height (pixels)')

# Set ticks to show the actual pixel coordinates
#plt.xticks(range(0, 256,20))  # Show every 4th pixel on x-axis
#plt.yticks(range(0, 256,20))  # Show every 4th pixel on y-axis

# Add grid to better visualize the pixel structure
#plt.grid(True, alpha=0.3, linewidth=0.5)



plt.show()



