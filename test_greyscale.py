import cv2
import numpy as np
from matplotlib import pyplot as plt

# Load image in color
image_path = "input_gcode/rect.jpg"
img = cv2.imread(image_path)
# Define black color range (BGR)
lower_black = np.array([0, 0, 0])
upper_black = np.array([50, 50, 50])  # Cho phép một chút gần đen

# Create mask: giữ pixel nằm trong vùng "đen"
mask_black = cv2.inRange(img, lower_black, upper_black)

# Invert to get binary: black stays black (0), others become white (255)
binary = cv2.bitwise_not(mask_black)

# Show result
plt.subplot(1, 2, 1)
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.title("Original")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(binary, cmap="gray")
plt.title("Better Black Preservation")
plt.axis("off")

plt.tight_layout()
plt.show()

