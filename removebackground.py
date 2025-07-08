
import os
import cv2
import numpy as np

def filter_black_color(image):
    """Lọc vùng đen trong ảnh màu, trả về ảnh nhị phân"""
    lower_black = np.array([0, 0, 0])
    upper_black = np.array([150, 150, 150]) # chỉnh thông số ở đây tùy ảnh để lọc màu
    mask_black = cv2.inRange(image, lower_black, upper_black)
    binary_image = cv2.bitwise_not(mask_black)
    return binary_image

def morphological_cleanup(image_gray):
    """Binarize, đảo màu và thực hiện closing để nối nét"""
    _, binary = cv2.threshold(image_gray, 180, 255, cv2.THRESH_BINARY)
    inverted = cv2.bitwise_not(binary)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    morph = cv2.morphologyEx(inverted, cv2.MORPH_CLOSE, kernel, iterations=1)
    return morph

def filter_connected_components(morph_image):
    """Giữ lại các vùng có diện tích và vị trí hợp lý (lọc nhiễu)"""
    nlabels, labels, stats, centroids = cv2.connectedComponentsWithStats(morph_image, connectivity=8)
    h, w = morph_image.shape
    center_x, center_y = w // 2, h // 2

    filtered = np.zeros_like(morph_image)
    for i in range(1, nlabels):  # Bỏ vùng nền
        area = stats[i, cv2.CC_STAT_AREA]
        x, y = stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP]
        bw, bh = stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]
        cx, cy = centroids[i]
        dist = np.sqrt((cx - center_x) ** 2 + (cy - center_y) ** 2)

        if area >= 80:
            filtered[labels == i] = 255
        elif area > 20 and dist < 300 and bw / bh < 3:
            filtered[labels == i] = 255
    return filtered

def process_image(image_path, output_folder='output_remove'):
    """Thực hiện toàn bộ pipeline xử lý và lưu ảnh"""
    if not os.path.exists(image_path):
        print(f"❌ File không tồn tại: {image_path}")
        return

    # ✅ Tạo thư mục nếu chưa có
    try:
        os.makedirs(output_folder, exist_ok=True)
    except Exception as e:
        print(f"❌ Không thể tạo thư mục {output_folder}: {e}")
        return

    image = cv2.imread(image_path)
    if image is None:
        print("❌ Không đọc được ảnh. Kiểm tra lại đường dẫn hoặc định dạng.")
        return

    # Xử lý từng bước
    filtered_color = filter_black_color(image)
    morph = morphological_cleanup(filtered_color)
    final_filtered = filter_connected_components(morph)
    result = cv2.bitwise_not(final_filtered)

    # Lưu ảnh kết quả
    filename = os.path.basename(image_path)
    filename_no_ext = os.path.splitext(filename)[0]
    output_path = os.path.join(output_folder, filename_no_ext + '_cleaned.jpg')

    success = cv2.imwrite(output_path, result)
    if success:
        print(f"✅ Đã lưu ảnh sau xử lý tại: {output_path}")
    else:
        print("❌ Lỗi khi lưu ảnh, kiểm tra lại đường dẫn và quyền ghi.")

def main():
    image_path = r'C:\D\image2Gcode\test_anh\F1500.jpg'  # <-- Cập nhật đường dẫn ảnh tại đây
    process_image(image_path)

if __name__ == "__main__":
    main()

