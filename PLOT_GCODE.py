import matplotlib.pyplot as plt
import math
import os
from PIL import Image
from PIL import ImageDraw
from matplotlib.patches import Rectangle
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

    new_img = Image.new("RGB", target_size, (255, 255, 255)) # màu đỏ
    # new_img = Image.new("RGB", target_size, (255, 127, 39))# màu cam
    # new_img = Image.new("RGB", target_size, (139, 69, 19)) # màu gỗ
    offset = ((target_w - new_w) // 2, (target_h - new_h) // 2)  # Căn giữa ảnh
    new_img.paste(resized_img, offset)
    return new_img

# Hàm để vẽ G-code và lưu ảnh
def draw_gcode_and_save(file_path, output_image_path="output.png", angle_deg=0):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    segments = []
    colors = []
    current_x = current_y = None

    for line in lines:
        line = line.strip()
        if line.startswith(('G0', 'G1')):
            x, y = parse_gcode_line(line)

            if current_x is None or current_y is None:
                current_x, current_y = x, y
                continue

            x = x if x is not None else current_x
            y = y if y is not None else current_y

            segments.append(((current_x, current_y), (x, y)))
            colors.append('black' if line.startswith('G1') else 'white')
            current_x, current_y = x, y

    if not segments:
        print(f"[WARNING] Không có đoạn G-code hợp lệ trong {file_path}")
        return

    all_x = [pt[0] for seg in segments for pt in seg]
    all_y = [pt[1] for seg in segments for pt in seg]
    center_x = (max(all_x) + min(all_x)) / 2
    center_y = (max(all_y) + min(all_y)) / 2

    fig, ax = plt.subplots(figsize=(3.937, 3.937))  # 100mm x 100mm

    rotated_segments = []
    for ((x0, y0), (x1, y1)), color in zip(segments, colors):
        x0r, y0r = rotate_point(x0, y0, angle_deg, (center_x, center_y))
        x1r, y1r = rotate_point(x1, y1, angle_deg, (center_x, center_y))
        rotated_segments.append(((x0r, y0r), (x1r, y1r)))
        ax.plot([x0r, x1r], [y0r, y1r], color=color, alpha=0.1 if color == 'white' else 1.0)

    all_rx = [pt[0] for seg in rotated_segments for pt in seg]
    all_ry = [pt[1] for seg in rotated_segments for pt in seg]
    x_min, x_max = min(all_rx), max(all_rx)
    y_min, y_max = min(all_ry), max(all_ry)

    bbox_w = x_max - x_min
    bbox_h = y_max - y_min
    ax.add_patch(Rectangle((x_min, y_min), bbox_w, bbox_h,
                           edgecolor='green', facecolor='none', linewidth=1.5, linestyle='--'))

    # === Gán tên đúng theo hệ tọa độ matplotlib (y tăng lên) ===
    top_y = y_max
    bottom_y = 0
    left_x = x_min
    right_x = x_max

    # === Tạo tọa độ 4 góc rõ ràng ===
    corner_coords = {
        "top_left": (left_x, top_y),
        "top_right": (right_x, top_y),
        "bottom_left": (left_x, bottom_y),
        "bottom_right": (right_x, bottom_y),
    }

    # === In ra tọa độ rõ ràng ===
    print("\n[INFO] Tọa độ 4 góc bounding box theo hệ trục x-y (mm):")
    for name, (x, y) in corner_coords.items():
        print(f"{name:>13}: ({x:.2f}, {y:.2f})")

    # === Vẽ chấm đỏ tại 4 góc ===
    for (x, y) in corner_coords.values():
        ax.plot(x, y, 'ro', markersize=4)

    # ax.set_aspect('equal', adjustable='box')
    # ax.axis('off')  # Không hiển thị trục
    # plt.tight_layout()
    # === Vẽ trục tọa độ và lưới chia 10mm ===
    ax.add_patch(Rectangle((0, 0), 100, 100, edgecolor='black', facecolor='none', linewidth=1))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    # ax.axis('off')
    # ax.set_xticks([])
    # ax.set_yticks([])
    ax.axis('on')
    ax.set_xticks(range(0, 110, 10))
    ax.set_yticks(range(0, 110, 10))
    ax.grid(True, linestyle='--', alpha=0.3)

    # fig.patch.set_facecolor((237 / 255, 28 / 255, 36 / 255))
    # fig.patch.set_facecolor((255/255, 127/255, 39/255))  # Đặt nền figure thành màu cam
    # fig.patch.set_facecolor((139 / 255, 69 / 255, 19 / 255))
    plt.gca().set_aspect('equal', adjustable='box')
    plt.tight_layout()

    # Lưu ảnh với DPI 300 để đảm bảo độ phân giải cho in ấn
    #plt.savefig(output_image_path, dpi=300, bbox_inches='tight', pad_inches=0)
    plt.savefig(output_image_path, dpi=300, bbox_inches=None, pad_inches=0)

    # Sử dụng PIL để resize ảnh với giữ tỷ lệ
    img = Image.open(output_image_path)
    img = img.convert("RGB")  # Đảm bảo là định dạng 24-bit RGB
    img_resized = resize_image_with_aspect_ratio(img, target_size=(320, 240))  # Resize giữ tỷ lệ
    # === Vẽ khung đen bao quanh ảnh 320x240 ===
    # draw = ImageDraw.Draw(img_resized)
    # draw.rectangle([(0, 0), (319, 239)], outline=(0, 0, 0), width=2)  # Khung viền đen 1px
    # Lưu ảnh đã căn giữa vào A4
    img_resized.save(output_image_path)  # Lưu lại ảnh

    plt.close()

# def draw_gcode_and_save(file_path, output_image_path="output.png", angle_deg=0):
#     with open(file_path, 'r') as file:
#         lines = file.readlines()
#
#     segments = []
#     colors = []
#     current_x = current_y = None
#
#     for line in lines:
#         line = line.strip()
#         if line.startswith(('G0', 'G1')):
#             x, y = parse_gcode_line(line)
#
#             if current_x is None or current_y is None:
#                 current_x, current_y = x, y
#                 continue
#
#             x = x if x is not None else current_x
#             y = y if y is not None else current_y
#
#             segments.append(((current_x, current_y), (x, y)))
#             colors.append('black' if line.startswith('G1') else 'white')
#             current_x, current_y = x, y
#
#     if not segments:
#         print(f"[WARNING] Không có đoạn G-code hợp lệ trong {file_path}")
#         return
#
#     all_x = [pt[0] for seg in segments for pt in seg]
#     all_y = [pt[1] for seg in segments for pt in seg]
#     center_x = (max(all_x) + min(all_x)) / 2
#     center_y = (max(all_y) + min(all_y)) / 2
#
#     fig, ax = plt.subplots(figsize=(3.937, 3.937))  # 100mm x 100mm
#
#     rotated_segments = []
#     for ((x0, y0), (x1, y1)), color in zip(segments, colors):
#         x0r, y0r = rotate_point(x0, y0, angle_deg, (center_x, center_y))
#         x1r, y1r = rotate_point(x1, y1, angle_deg, (center_x, center_y))
#         rotated_segments.append(((x0r, y0r), (x1r, y1r)))
#         ax.plot([x0r, x1r], [y0r, y1r], color=color, alpha=0.1 if color == 'white' else 1.0)
#
#     all_rx = [pt[0] for seg in rotated_segments for pt in seg]
#     all_ry = [pt[1] for seg in rotated_segments for pt in seg]
#      # === Vẽ trục tọa độ và lưới chia 10mm ===
#     ax.add_patch(Rectangle((0, 0), 100, 100, edgecolor='black', facecolor='none', linewidth=2))
#     ax.set_xlim(0, 100)
#     ax.set_ylim(0, 100)
#     ax.axis('off')
#     ax.set_xticks([])
#     ax.set_yticks([])
#     # ax.axis('on')
#     # ax.set_xticks(range(0, 110, 10))
#     # ax.set_yticks(range(0, 110, 10))
#     # ax.grid(True, linestyle='--', alpha=0.3)
#
#     # fig.patch.set_facecolor((237 / 255, 28 / 255, 36 / 255))
#     # fig.patch.set_facecolor((255/255, 127/255, 39/255))  # Đặt nền figure thành màu cam
#     # fig.patch.set_facecolor((139 / 255, 69 / 255, 19 / 255))
#     plt.gca().set_aspect('equal', adjustable='box')
#     plt.tight_layout()
#
#     # Lưu ảnh với DPI 300 để đảm bảo độ phân giải cho in ấn
#     #plt.savefig(output_image_path, dpi=300, bbox_inches='tight', pad_inches=0)
#     plt.savefig(output_image_path, dpi=300, bbox_inches=None, pad_inches=0)
#
#     # Sử dụng PIL để resize ảnh với giữ tỷ lệ
#     img = Image.open(output_image_path)
#     img = img.convert("RGB")  # Đảm bảo là định dạng 24-bit RGB
#     img_resized = resize_image_with_aspect_ratio(img, target_size=(320, 240))  # Resize giữ tỷ lệ
#     # === Vẽ khung đen bao quanh ảnh 320x240 ===
#     # draw = ImageDraw.Draw(img_resized)
#     # draw.rectangle([(0, 0), (319, 239)], outline=(0, 0, 0), width=2)  # Khung viền đen 1px
#     # Lưu ảnh đã căn giữa vào A4
#     img_resized.save(output_image_path)  # Lưu lại ảnh
#
#     plt.close()


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


