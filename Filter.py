import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load ảnh
img = cv2.imread(r'C:\D\image2Gcode\output_remove\test_filtered.jpg', cv2.IMREAD_GRAYSCALE)

# Binarize và đảo màu
_, binary = cv2.threshold(img, 180, 255, cv2.THRESH_BINARY)
inverted = cv2.bitwise_not(binary)

# Morphological closing để nối nét
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
morph = cv2.morphologyEx(inverted, cv2.MORPH_CLOSE, kernel, iterations=1)

# Tìm các connected components
nlabels, labels, stats, centroids = cv2.connectedComponentsWithStats(morph, connectivity=8)
h, w = img.shape
center_x, center_y = w // 2, h // 2

filtered = np.zeros_like(morph)

for i in range(1, nlabels):  # Bỏ vùng nền
    area = stats[i, cv2.CC_STAT_AREA]
    x, y, bw, bh = stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP], stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]
    cx, cy = centroids[i]
    dist = np.sqrt((cx - center_x) ** 2 + (cy - center_y) ** 2)

    # Điều kiện giữ lại:
    if area >= 80:  # tăng diện tích để loại nhiễu nhỏ
        filtered[labels == i] = 255
    elif area > 20 and dist < 300 and bw / bh < 3:  # giữ lại nét trong vùng trung tâm nếu hợp lý
        filtered[labels == i] = 255

# # === Tìm khung hình chữ nhật (contour lớn nhất có 4 cạnh) ===
# contours, _ = cv2.findContours(filtered, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# approx_rect = None
# max_area = 0
#
# for cnt in contours:
#     approx = cv2.approxPolyDP(cnt, epsilon=5, closed=True)
#     if len(approx) == 4:
#         area = cv2.contourArea(cnt)
#         if area > max_area:
#             max_area = area
#             approx_rect = approx
#
# # Nếu tìm được khung → vẽ lại nét vuông 1px
# if approx_rect is not None:
#     x, y, w, h = cv2.boundingRect(approx_rect)
#     cv2.rectangle(filtered, (x, y), (x + w, y + h), 255, 1)  # white line = 255 (vì nền đen)

# Đảo lại màu về ảnh nét đen, nền trắng
result = cv2.bitwise_not(filtered)

# Hiển thị
plt.figure(figsize=(10, 10))
plt.imshow(result, cmap='gray')
plt.title("Lọc ảnh")
plt.axis('off')
plt.show()