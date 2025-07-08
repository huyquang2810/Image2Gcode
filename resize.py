import os
import cv2
import numpy as np

# === STEP 1: Load images ===
img1_path = r"C:\D\image2Gcode\input_resize\6_gcode_100.nc.jpg"
img2_path = r"C:\D\image2Gcode\output_remove\F1500_cleaned.jpg"
img1 = cv2.imread(img1_path)
img2 = cv2.imread(img2_path)

# === Hàm tìm bounding box khuôn mặt ===
def find_face_bbox(img, threshold=200):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    return cv2.boundingRect(np.vstack(contours))  # x, y, w, h

# === Tìm bounding box cho cả 2 ảnh ===
bbox1 = find_face_bbox(img1)
bbox2 = find_face_bbox(img2)

# === So sánh kích thước bbox và chọn khung lớn hơn làm chuẩn ===
w1, h1 = bbox1[2], bbox1[3]
w2, h2 = bbox2[2], bbox2[3]
target_w = max(w1, w2)
target_h = max(h1, h2)
target_size = (target_w, target_h)

# === Hàm resize giữ nguyên tỉ lệ nhưng fit vào target_size ===
def crop_and_resize_keep_ratio(img, bbox, target_size):
    x, y, w, h = bbox
    face = img[y:y+h, x:x+w]
    fh, fw = face.shape[:2]
    target_w, target_h = target_size

    # Scale giữ nguyên tỉ lệ
    scale = min(target_w / fw, target_h / fh)
    new_w = int(fw * scale)
    new_h = int(fh * scale)

    # Resize giữ tỉ lệ
    resized = cv2.resize(face, (new_w, new_h), interpolation=cv2.INTER_CUBIC)

    # Đặt giữa canvas trắng đúng target_size
    canvas = np.ones((target_h, target_w, 3), dtype=np.uint8) * 255
    x_offset = (target_w - new_w) // 2
    y_offset = (target_h - new_h) // 2
    canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized

    return canvas

# === Resize từng mặt giữ nguyên tỉ lệ, dùng cùng target_size ===
face1_resized = crop_and_resize_keep_ratio(img1, bbox1, target_size)
face2_resized = crop_and_resize_keep_ratio(img2, bbox2, target_size)

# === Lưu kết quả vào thư mục output_resize ===
output_dir = r"C:\D\image2Gcode\output_resize"
os.makedirs(output_dir, exist_ok=True)
cv2.imwrite(os.path.join(output_dir, "aligned_img1.jpg"), face1_resized)
cv2.imwrite(os.path.join(output_dir, "aligned_img2.jpg"), face2_resized)

print("✅ Đã lưu ảnh đã resize giữ nguyên tỉ lệ vào:", output_dir)
