import insightface
import cv2
import numpy as np
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
import rembg
from PIL import Image
import os
from datetime import datetime
import re
from openpyxl import Workbook
from openpyxl.drawing.image import Image as XLImage
from PIL import Image as PILImage
from openpyxl.styles import Font
import matplotlib.pyplot as plt

# --- Class xử lý ảnh và sinh G-code ---
class Picture:
    def __init__(self, filepath, x_max=40, y_max=40):
        self.img = Image.open(filepath).convert("RGB")
        self.img = np.array(self.img)
        self.h, self.w, self.c = self.img.shape
        self.pre = np.ones(self.img.shape)
        self.gcode = ['G28']
        self.x_max = x_max
        self.y_max = y_max

    def gray_scale(self):
        gray = cv2.cvtColor(self.img, cv2.COLOR_RGB2GRAY)
        self.gray = gray / 255.0
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)
        edges = cv2.Canny(blurred, threshold1=50, threshold2=150)
        binary = (edges / 255.0).astype(float)
        self.pre = np.stack([binary] * 3, axis=-1)
        return self.pre

    def save_gray(self, output):
        if hasattr(self, 'gray') and self.gray is not None:
            plt.imshow(self.gray, cmap='gray')
            plt.axis('off')
            plt.imsave(output + '_gray.jpg', self.gray, cmap='gray')
            print('✅ Saved ' + output + '_gray.jpg')

    def save_binary(self, output):
        plt.imshow(self.pre, cmap='gray')
        plt.axis('off')
        plt.imsave(output + '_binary.jpg', self.pre)
        print('✅ Saved ' + output + '_binary.jpg')

    def gen_gcode(self):
        binary = (self.pre[:, :, 0] > 0.5).astype(np.uint8) * 255
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        ratio = self.x_max / max(self.w, self.h)
        total_points = 0
        for contour in contours:
            if len(contour) < 2:
                continue
            total_points += len(contour)
            x0, y0 = contour[0][0]
            y0_flipped = self.h - y0
            self.gcode.append(f"G0 X{x0 * ratio:.4f} Y{y0_flipped * ratio:.4f}")
            for pt in contour[1:]:
                x, y = pt[0]
                y_flipped = self.h - y
                self.gcode.append(f"G1 X{x * ratio:.4f} Y{y_flipped * ratio:.4f}")
        return self.gcode, total_points

    def save_gcode(self, output_name):
        os.makedirs(os.path.dirname(output_name), exist_ok=True)
        with open(f'{output_name}_gcode.nc', 'w') as f:
            for line in self.gcode:
                f.write(f'{line}\n')
        print(f'✅ Saved {output_name}_gcode.nc')

# --- Tiện ích resize ảnh trước khi gắn vào Excel ---
def resize_and_save_temp(image_path, output_path, max_size=(100, 100)):
    try:
        img = PILImage.open(image_path)
        img.thumbnail(max_size)
        img.save(output_path)
        return True
    except Exception as e:
        print(f"⚠️ Lỗi khi resize ảnh: {image_path} → {e}")
        return False

# --- Xử lý khuôn mặt ---
def init_face_analyzer():
    print("Initializing face analysis...")
    face_analyzer = insightface.app.FaceAnalysis(name='buffalo_l')
    try:
        face_analyzer.prepare(ctx_id=0)
    except Exception:
        print("GPU initialization failed. Switching to CPU...")
        face_analyzer.prepare(ctx_id=-1)
    return face_analyzer

def align_face(image, face):
    left_eye = tuple(face.kps[0].astype(int))
    right_eye = tuple(face.kps[1].astype(int))
    eyes_center = tuple(map(int, ((left_eye[0] + right_eye[0]) / 2, (left_eye[1] + right_eye[1]) / 2)))
    dx = right_eye[0] - left_eye[0]
    dy = right_eye[1] - left_eye[1]
    angle = np.degrees(np.arctan2(dy, dx))
    rotation_matrix = cv2.getRotationMatrix2D(eyes_center, angle, 1)
    aligned = cv2.warpAffine(image.astype(np.float32), rotation_matrix, (image.shape[1], image.shape[0]),
                             flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))
    return np.clip(aligned, 0, 255).astype(np.uint8)

def resize_to_a4(image, target_width=210, target_height=297):
    h, w = image.shape[:2]
    scale = min(target_width / w, target_height / h)
    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    canvas = np.ones((target_height, target_width, 3), dtype=np.uint8) * 255
    top = (target_height - new_h) // 2
    left = (target_width - new_w) // 2
    canvas[top:top + new_h, left:left + new_w] = resized
    return canvas

def extract_faces_to_A4(input_folder, output_folder, face_analyzer):
    os.makedirs(output_folder, exist_ok=True)
    filenames = sorted(
        [f for f in os.listdir(input_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))],
        key=lambda x: int(re.search(r'\d+', x).group()) if re.search(r'\d+', x) else float('inf')
    )

    for file in filenames:
        path = os.path.join(input_folder, file)
        image = cv2.imread(path)
        if image is None: continue

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(image_rgb)
        removed = rembg.remove(pil_img)
        rgba = removed.convert("RGBA")
        np_img = np.array(rgba)
        white_bg = np.ones_like(np_img[:, :, :3], dtype=np.uint8) * 255
        alpha = np_img[:, :, 3:4] / 255.0
        blend = (np_img[:, :, :3] * alpha + white_bg * (1 - alpha)).astype(np.uint8)

        faces = face_analyzer.get(blend)
        faces = face_analyzer.get(blend)
        if not faces:
            print("❌ Không phát hiện được khuôn mặt trong ảnh:", os.path.basename(path))
            continue

        face = faces[0]
        aligned = align_face(blend, face)
        x1, y1, x2, y2 = face.bbox.astype(int)
        h, w = aligned.shape[:2]
        top = np.clip(y1 - int(0.45 * (y2 - y1)), 0, h)
        bottom = np.clip(y2 + int(0.05 * (y2 - y1)), 0, h)
        left = np.clip(x1 - int(0.3 * (x2 - x1)), 0, w)
        right = np.clip(x2 + int(0.3 * (x2 - x1)), 0, w)
        crop = aligned[top:bottom, left:right]
        crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        a4 = resize_to_a4(crop_rgb)
        save_path = os.path.join(output_folder, os.path.splitext(file)[0] + "_a4.jpg")
        cv2.imwrite(save_path, a4)
        print(f"✅ Saved A4 image: {os.path.basename(save_path)}")

# === Main pipeline ===
if __name__ == '__main__':
    face_model = init_face_analyzer()
    print("Chọn chế độ:\n1: Folder\n2: Webcam")
    mode = input("Nhập chế độ: ")

    if mode == '1':
        input_face_dir = "Input_Folder"
    elif mode == '2':
        input_face_dir = "Cam_Input"
        os.makedirs(input_face_dir, exist_ok=True)
        cap = cv2.VideoCapture(0, cv2.CAP_ANY)
        if not cap.isOpened():
            print("❌ Webcam không thể truy cập.")
            exit()
        count = 1
        print("Nhấn Enter để chụp, ESC để thoát.")
        while True:
            ret, frame = cap.read()
            if not ret:
                print("❌ Không lấy được frame.")
                break
            cv2.imshow("Webcam", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == 13:
                save_path = os.path.join(input_face_dir, f"cam_{count}.jpg")
                cv2.imwrite(save_path, frame)
                print(f"📸 Đã chụp: {save_path}")
                count += 1
            elif key == 27:
                break
        cap.release()
        cv2.destroyAllWindows()
    else:
        print("Chế độ không hợp lệ.")
        exit()

    aligned_face_dir = "img"
    extract_faces_to_A4(input_face_dir, aligned_face_dir, face_model)

    output_folder = 'out'
    excel_folder = 'excel'
    thumb_folder = os.path.join(excel_folder, 'thumbs')
    os.makedirs(output_folder, exist_ok=True)
    os.makedirs(thumb_folder, exist_ok=True)
    os.makedirs(excel_folder, exist_ok=True)

    files = sorted([f for f in os.listdir(aligned_face_dir) if f.lower().endswith(('.jpg', '.jpeg'))],
                   key=lambda x: int(re.search(r'\d+', x).group()) if re.search(r'\d+', x) else float('inf'))

    wb = Workbook()
    ws = wb.active
    ws.title = "Thời gian xử lý ảnh"
    ws.append(["Tên file", "Input", "Thời gian (s)", "Output", "Số điểm G-code"])
    ws.column_dimensions['A'].width = 25
    ws.column_dimensions['B'].width = 14
    ws.column_dimensions['C'].width = 16
    ws.column_dimensions['D'].width = 14
    ws.column_dimensions['E'].width = 14

    for idx, filename in enumerate(files):
        input_path = os.path.join(aligned_face_dir, filename)
        base_name = os.path.splitext(filename)[0]
        output_name = os.path.join(output_folder, base_name)  # ← Không thêm số thứ tự nữa

        print(f"\n🔧 Processing {filename}...")
        start_time = datetime.now()

        pic = Picture(input_path)
        pic.gray_scale()
        pic.save_gray(output_name)
        pic.save_binary(output_name)
        gcode, num_points = pic.gen_gcode()
        pic.save_gcode(output_name)

        duration = (datetime.now() - start_time).total_seconds()
        ws.append([filename, "", round(duration, 3), "", num_points])
        current_row = ws.max_row

        # Đặt chiều cao hàng tương ứng với thumbnail
        ws.row_dimensions[current_row].height = 85

        thumb1 = os.path.join(thumb_folder, f'{base_name}_thumb1.jpg')
        thumb2 = os.path.join(thumb_folder, f'{base_name}_thumb2.jpg')
        if resize_and_save_temp(input_path, thumb1):
            ws.add_image(XLImage(thumb1), f'B{current_row}')
        if resize_and_save_temp(f'{output_name}_binary.jpg', thumb2):
            ws.add_image(XLImage(thumb2), f'D{current_row}')

    total_time = sum(ws.cell(row=r, column=3).value for r in range(2, ws.max_row + 1))
    ws.append(["", "", f"TỔNG: {total_time:.3f} giây", "", ""])
    ws[f"C{ws.max_row}"].font = Font(bold=True)

    wb.save(os.path.join(excel_folder, 'time_processing.xlsx'))
    print("🎉 Xử lý hoàn tất!")
