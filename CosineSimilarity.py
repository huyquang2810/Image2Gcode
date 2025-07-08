import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model
from scipy.spatial.distance import cosine
import os

# === Hàm xử lý ảnh để đưa vào VGG16 ===
def preprocess_image(image_path):
    img = image.load_img(image_path, target_size=(224, 224))
    img_data = image.img_to_array(img)
    # thêm khúc chỉnh threshold để tăng độ chính xác
    _, img_data = cv2.threshold(img_data, 125, 255, cv2.THRESH_BINARY)
    img_data = np.expand_dims(img_data, axis=0)
    img_data = preprocess_input(img_data)

    return img_data

# === Hàm chính: tính cosine similarity và hiển thị ảnh GUI ===
def calculate_feature_similarity(image_path1, image_path2):
    # Load pre-trained VGG16 (không gồm tầng fully-connected)
    base_model = VGG16(weights='imagenet', include_top=False)
    model = Model(inputs=base_model.input, outputs=base_model.get_layer('block5_pool').output)

    # Tiền xử lý ảnh
    img1 = preprocess_image(image_path1)
    img2 = preprocess_image(image_path2)

    # Trích xuất đặc trưng
    features_img1 = model.predict(img1).flatten()
    features_img2 = model.predict(img2).flatten()

    # Tính cosine similarity
    similarity = 1 - cosine(features_img1, features_img2)

    # === Hiển thị GUI với ảnh ===
    img1_display = cv2.cvtColor(cv2.imread(image_path1), cv2.COLOR_BGR2RGB)
    #_, img1_display = cv2.threshold(cv2.imread(image_path1), 125, 255, cv2.THRESH_BINARY)
    img2_display = cv2.cvtColor(cv2.imread(image_path2), cv2.COLOR_BGR2RGB)
    #_, img2_display = cv2.threshold(cv2.imread(image_path2), 125, 255, cv2.THRESH_BINARY)
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))  # Tạo 1 hàng 2 cột

    # Ảnh gốc
    axs[0].imshow(img1_display)
    axs[0].set_title("Original Image", fontsize=14, pad=30, weight='bold')

    axs[0].axis("off")

    # Ảnh tối ưu + text
    axs[1].imshow(img2_display)
    axs[1].set_title("Actual Image (Feed rate 1500)", fontsize=14, pad=30, weight='bold')
    axs[1].axis("off")

    # # Thêm similarity dưới ảnh tối ưu
    # axs[1].text(0.5, -0.12, f"Cosine Similarity: {similarity:.4f}",
    #             fontsize=12, color="black", ha='center', transform=axs[1].transAxes)
    #
    # fig.suptitle("Feature-based Cosine Similarity", fontsize=16)
    # plt.tight_layout(rect=[0, 0, 1, 0.95])  # chừa khoảng cho tiêu đề
    # plt.show()

    # === Tiêu đề trên cùng: Cosine Similarity in đậm ===
    fig.suptitle(f"Cosine Similarity: {similarity:.4f}", fontsize=16, weight='bold')

    plt.tight_layout(rect=[0, 0, 1, 0.88])
    plt.show()

    return similarity

# === Hàm main để chạy ===
def main():
    # Thay đổi đường dẫn cho phù hợp với ảnh của bạn
    original_image_path = r'C:\D\image2Gcode\output_resize\aligned_img1.jpg'
    optimized_image_paths = [
        r'C:\D\image2Gcode\output_resize\aligned_img2.jpg',
        # Thêm nhiều ảnh nếu cần
    ]

    for compare_path in optimized_image_paths:
        if os.path.exists(original_image_path) and os.path.exists(compare_path):
            similarity = calculate_feature_similarity(original_image_path, compare_path)
            print(f"[INFO] Similarity between:\n- {os.path.basename(original_image_path)}\n- {os.path.basename(compare_path)}\n→ Cosine similarity = {similarity:.4f}")
        else:
            print(f"[ERROR] File không tồn tại:\n- {original_image_path}\n- {compare_path}")

# === Chạy script nếu gọi trực tiếp ===
if __name__ == "__main__":
    main()


