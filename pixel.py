import cv2
import numpy as np
import matplotlib.pyplot as plt

# ==== 1. Load ảnh ====
img1_path = r'C:\D\image2Gcode\input_framePicture\test.jpg'
img2_path = r'/test_anh/6_gcode_100_khung.jpg'

img1 = cv2.imread(img1_path, cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread(img2_path, cv2.IMREAD_GRAYSCALE)
print(img1)
# ==== 2. Resize nếu kích thước khác nhau ====
if img1.shape != img2.shape:
    img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))

# ==== 3. Nhị phân hóa ====
_, img1_bin = cv2.threshold(img1, 150, 255, cv2.THRESH_BINARY)
_, img2_bin = cv2.threshold(img2, 125, 255, cv2.THRESH_BINARY)

print(img1_bin)
#print(img2_bin)
# ==== 4. Tính hiệu tuyệt đối ====
diff = cv2.absdiff(img1_bin, img2_bin)

# ==== 5. Tính thống kê ====
num_diff_pixels = np.count_nonzero(diff)
total_pixels = diff.size
percent_diff = (num_diff_pixels / total_pixels) * 100

# ==== 6. In kết quả ra console ====
print("===== Kết quả so sánh hai ảnh =====")
print(f"Số pixel khác biệt: {num_diff_pixels}")
print(f"Tổng số pixel:      {total_pixels}")
print(f"Tỷ lệ khác biệt:     {percent_diff:.2f}%")

# ==== 7. Hiển thị ảnh diff với nền trắng, nét đen ====
# plt.figure(figsize=(6, 6))
# plt.imshow(255 - diff, cmap='gray', vmin=0, vmax=255)
#   # nền trắng, nét đen
# plt.title("Ảnh khác biệt (nền trắng, nét đen)")
# plt.axis('off')
# # plt.show()

plt.figure(figsize=(6, 6))
plt.imshow(img1_bin, cmap='gray', vmin=0, vmax=255)
  # nền trắng, nét đen
plt.title("1")
plt.axis('off')
plt.show()

# plt.figure(figsize=(6, 6))
# plt.imshow(img2_bin, cmap='gray', vmin=0, vmax=255)
#   # nền trắng, nét đen
# plt.title("2")
# plt.axis('off')
# plt.show()