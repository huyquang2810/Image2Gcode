import cv2
import numpy as np
import os

# ======== Cấu hình ========
mockup_path = r'/test_anh/6_gcode_100_khung.jpg'  # ảnh mô phỏng 320x240 có khung
image_to_crop = r'C:\D\image2Gcode\640x480.jpg'     # ảnh cần cắt theo vị trí tương ứng
output_dir_mockup = 'output_crop'
output_dir_target = 'output_target'
os.makedirs(output_dir_mockup, exist_ok=True)
os.makedirs(output_dir_target, exist_ok=True)

# ======== Đọc ảnh mô phỏng và nhúng vào canvas ========
mockup = cv2.imread(mockup_path, cv2.IMREAD_GRAYSCALE)
if mockup.shape != (240, 320):
    raise ValueError("Ảnh mô phỏng phải có kích thước 320x240.")

canvas = np.ones((480, 640), dtype=np.uint8) * 255
x_offset, y_offset = 160, 120
canvas[y_offset:y_offset+240, x_offset:x_offset+320] = mockup
cv2.imwrite(os.path.join(output_dir_mockup, 'mockup_640x480.jpg'), canvas)
# ======== Tìm contour và tọa độ cắt (loại bỏ viền) ========
_, binary = cv2.threshold(canvas, 127, 255, cv2.THRESH_BINARY_INV)
contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# Tìm contour lớn nhất (là khung)
x, y, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))

margin = 8
x_crop = x + margin
y_crop = y + margin
w_crop = w - 2 * margin #kích thước hình sau khi cắt
h_crop = h - 2 * margin # kích thuocs hình sau khi cắt

# ======== Cắt ảnh mô phỏng (bỏ viền) và lưu vào output_crop ========
crop_mockup = canvas[y_crop:y_crop+h_crop, x_crop:x_crop+w_crop]
cv2.imwrite(os.path.join(output_dir_mockup, 'mockup_crop_inside_frame.jpg'), crop_mockup)

# ======== Cắt ảnh thực tế theo cùng vị trí và lưu vào output_target ========
target_img = cv2.imread(image_to_crop, cv2.IMREAD_GRAYSCALE)
if target_img.shape[:2] != (480, 640):
    raise ValueError("Ảnh cần cắt phải có kích thước 640x480.")

crop_target = target_img[y_crop:y_crop+h_crop, x_crop:x_crop+w_crop]
cv2.imwrite(os.path.join(output_dir_target, 'cropped_from_target.jpg'), crop_target)

print("✅ Cắt xong:")

