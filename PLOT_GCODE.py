import matplotlib.pyplot as plt
import math
import os
from PIL import Image
from PIL import ImageDraw
# Hàm xoay điểm x, y quanh một điểm gốc (origin)
def rotate_point(x, y, angle_deg, origin=(0, 0)):
    angle_rad = math.radians(angle_deg)
    ox, oy = origin
    tx, ty = x - ox, y - oy
    rx = tx * math.cos(angle_rad) - ty * math.sin(angle_rad)
    ry = tx * math.sin(angle_rad) + ty * math.cos(angle_rad)
    return rx + ox, ry + oy

# Hàm để phân tích các lệnh G-code và trích xuất giá trị X và Y
def parse_gcode_line(line):
    x = y = None
    if line.startswith(('G0', 'G1')):  # Xử lý các lệnh G0 và G1
        parts = line.split()
        for part in parts:
            if part.startswith('X'):
                x = float(part[1:])
            elif part.startswith('Y'):
                y = float(part[1:])
    return x, y

# Hàm resize ảnh giữ nguyên tỷ lệ
def resize_image_with_aspect_ratio(img, target_size=(320, 240)):
    # Lấy kích thước gốc của ảnh
    img_w, img_h = img.size
    target_w, target_h = target_size

    # Tính tỷ lệ để giữ nguyên aspect ratio
    scale_w = target_w / img_w
    scale_h = target_h / img_h
    scale = min(scale_w, scale_h)  # Lấy tỷ lệ nhỏ nhất để ảnh không bị co kéo

    # Tính kích thước mới của ảnh
    new_w = int(img_w * scale)
    new_h = int(img_h * scale)

    # Resize ảnh giữ nguyên tỷ lệ
    resized_img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)

    new_img = Image.new("RGB", target_size, (255, 255, 255))
    # new_img = Image.new("RGB", target_size, (139, 69, 19)) # màu gỗ
    # new_img = Image.new("RGB", target_size, (255, 0, 0))  # Nền màu đỏ
    offset = ((target_w - new_w) // 2, (target_h - new_h) // 2)  # Căn giữa ảnh
    new_img.paste(resized_img, offset)

    return new_img

# Hàm để vẽ G-code và lưu ảnh
def draw_gcode_and_save(file_path, output_image_path="output.png", angle_deg=0):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    segments = []
    colors = []
    current_x, current_y = 0, 0
    # wood_color = (139 / 255, 69 / 255, 19 / 255)  # Màu gỗ (SaddleBrown)

    for line in lines:
        line = line.strip()
        if line.startswith('G1'):  # Vẽ các lệnh G1
            x, y = parse_gcode_line(line)
            x = x if x is not None else current_x
            y = y if y is not None else current_y
            segments.append(((current_x, current_y), (x, y)))
            colors.append('black')  # Màu đen cho lệnh G1
            current_x, current_y = x, y
        elif line.startswith('G0'):  # Vẽ lệnh G0 với độ mờ thấp
            x, y = parse_gcode_line(line)
            x = x if x is not None else current_x
            y = y if y is not None else current_y
            segments.append(((current_x, current_y), (x, y)))
            colors.append('white')  # Lệnh G0 sẽ có màu trắng, không hiển thị rõ ràng
            # colors.append(wood_color)
            current_x, current_y = x, y

    # Xoay quanh tâm
    all_x = [pt[0] for seg in segments for pt in seg]
    all_y = [pt[1] for seg in segments for pt in seg]
    center_x = (max(all_x) + min(all_x)) / 2
    center_y = (max(all_y) + min(all_y)) / 2

    # Tạo hình ảnh với kích thước 100mm x 100mm (tương đương 3.937 inch x 3.937 inch)
    fig, ax = plt.subplots(figsize=(3.937, 3.937))  # Kích thước 100mm x 100mm (3.937 inch)

    # Vẽ các đoạn thẳng với màu sắc và độ mờ phù hợp
    for ((x0, y0), (x1, y1)), color in zip(segments, colors):
        x0r, y0r = rotate_point(x0, y0, angle_deg, (center_x, center_y))
        x1r, y1r = rotate_point(x1, y1, angle_deg, (center_x, center_y))
        ax.plot([x0r, x1r], [y0r, y1r], color=color, alpha=0.1 if color == 'white' else 1.0)

    # ax.set_aspect('equal', adjustable='box')
    # ax.axis('off')  # Không hiển thị trục
    # plt.tight_layout()

    # # Set the background to red
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis('off')
    ax.set_xticks([])
    ax.set_yticks([])

    # fig.patch.set_facecolor('red')  # Đặt nền figure thành màu đỏ
    # fig.patch.set_facecolor((139 / 255, 69 / 255, 19 / 255))
    plt.gca().set_aspect('equal', adjustable='box')
    plt.tight_layout()

    # Lưu ảnh với DPI 300 để đảm bảo độ phân giải cho in ấn
    plt.savefig(output_image_path, dpi=300, bbox_inches='tight', pad_inches=0)

    # Sử dụng PIL để resize ảnh với giữ tỷ lệ
    img = Image.open(output_image_path)
    img = img.convert("RGB")  # Đảm bảo là định dạng 24-bit RGB
    img_resized = resize_image_with_aspect_ratio(img, target_size=(320, 240))  # Resize giữ tỷ lệ
    # === Vẽ khung đen bao quanh ảnh 320x240 ===
    draw = ImageDraw.Draw(img_resized)
    draw.rectangle([(0, 0), (319, 239)], outline=(0, 0, 0), width=2)  # Khung viền đen 1px
    # Lưu ảnh đã căn giữa vào A4
    img_resized.save(output_image_path)  # Lưu lại ảnh

    plt.close()

# Hàm xử lý tất cả các file G-code trong thư mục
def process_all_gcodes_in_directory(input_dir, output_dir, angle_deg=0):
    # Tạo thư mục output nếu chưa tồn tại
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Duyệt tất cả các file trong thư mục input_dir
    for file_name in os.listdir(input_dir):
        file_path = os.path.join(input_dir, file_name)

        # Kiểm tra nếu là file G-code (tệp .gcode hoặc .nc)
        if os.path.isfile(file_path) and file_name.lower().endswith(('.gcode', '.nc')):
            output_image_path = os.path.join(output_dir, file_name + '.jpg')
            draw_gcode_and_save(file_path, output_image_path, angle_deg)
            print(f"Đã xuất ảnh cho {file_name} vào {output_image_path}")

# Đường dẫn tới thư mục chứa các file G-code
input_dir = "C:\D\image2Gcode\input_scalePLOT"  # Thay đổi đường dẫn theo thư mục của bạn
output_dir = "C:/D/image2Gcode/output_PLOT"  # Thư mục lưu ảnh

# Gọi hàm để xử lý tất cả các file G-code trong thư mục
process_all_gcodes_in_directory(input_dir, output_dir)


