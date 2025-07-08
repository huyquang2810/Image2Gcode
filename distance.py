import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

# === STEP 1: Load image ===
img_path = r"C:\D\image2Gcode\output_PLOT\6_gcode_100.nc.jpg"
img = cv2.imread(img_path)

# === Config: Độ dày khung viền ngoài ===
border_thickness = 1  # <-- bạn chỉ cần đổi số này nếu cần viền dày hơn
# === Cấu hình DPI (Dots Per Inch) ===
DPI = 96  # DPI của ảnh (300 DPI là tiêu chuẩn cho in ảnh chất lượng cao)

# === Hàm tìm bounding box khuôn mặt ===
def find_face_bbox(img, threshold=200):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    return cv2.boundingRect(np.vstack(contours))  # (x, y, w, h)

# === Tìm bounding box và vẽ ===
bbox = find_face_bbox(img)
if bbox:
    x, y, w, h = bbox
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 1)  # Bounding box màu xanh lá

    # 4 điểm góc
    top_left = (x, y)
    top_right = (x + w - 1, y)
    bottom_left = (x, y + h - 1)
    bottom_right = (x + w - 1, y + h - 1)

    # Vẽ chấm đỏ tại 4 góc
    for pt in [top_left, top_right, bottom_left, bottom_right]:
        cv2.circle(img, pt, 3, (0, 0, 255), -1)

    # === In phần 1: Tọa độ các điểm ===
    print("[INFO] Tọa độ 4 góc bounding box:")
    print(f"  Top-Left     : {top_left}")
    print(f"  Top-Right    : {top_right}")
    print(f"  Bottom-Left  : {bottom_left}")
    print(f"  Bottom-Right : {bottom_right}")

else:
    print("[WARNING] Không tìm thấy contour.")

# # === Vẽ khung viền đen quanh ảnh ===
# height, width = img.shape[:2]
# cv2.rectangle(img, (0, 0), (width - 1, height - 1), (0, 0, 0), border_thickness)

# === Tạo thư mục nếu chưa có ===
output_dir = "output_distance"
os.makedirs(output_dir, exist_ok=True)

# === Lưu ảnh ra thư mục ===
output_path = os.path.join(output_dir, "bounding_box_result.jpg")
cv2.imwrite(output_path, img)
print(f"[INFO] Ảnh đã được lưu tại: {output_path}")

