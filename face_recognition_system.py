#!/usr/bin/env python3
"""
适配 RTX 4090 的人脸识别系统 (增强版)
检测：InsightFace (buffalo_l)
识别：ArcFace (facenet-pytorch)
绘图：PIL + Noto Serif CJK (支持中文显示)
"""

import os
import cv2
import numpy as np
import argparse
from pathlib import Path
from tqdm import tqdm

# 关键导入
from insightface.app import FaceAnalysis
import torch
import torch.nn.functional as F
from facenet_pytorch import InceptionResnetV1
from PIL import Image, ImageDraw, ImageFont, ImageColor
import torchvision.transforms as transforms

class FaceRecognizer:
    def __init__(self, dataset_path, threshold=0.6, arcface_threshold=0.6, device=None):
        self.threshold = threshold
        self.arcface_threshold = arcface_threshold
        
        # 1. 设备初始化
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        print(f"🔧 使用设备: {self.device}")

        # 2. 字体加载逻辑 (Noto Serif CJK Regular)
        # 常见路径: Ubuntu 的 opentype 或 truetype 目录
        font_paths = [
            "/usr/share/fonts/opentype/noto/NotoSerifCJK-Regular.ttc",
            "/usr/share/fonts/truetype/noto/NotoSerifCJK-Regular.ttc",
            "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
            "NotoSerifCJK-Regular.ttc" # 如果你在当前目录放了字体文件
        ]
        self.font = None
        self.font_size = 24
        for p in font_paths:
            if os.path.exists(p):
                self.font = ImageFont.truetype(p, self.font_size)
                print(f"✅ 成功加载字体: {p}")
                break
        if self.font is None:
            print("⚠️ 未找到 NotoSerif 字体，将使用系统默认字体（可能不支持中文）")
            self.font = ImageFont.load_default()

        # 3. 初始化 RetinaFace (buffalo_l)
        print("🔄 正在初始化 RetinaFace (buffalo_l)...")
        self.app = FaceAnalysis(
            name='buffalo_l', 
            allowed_modules=['detection'], 
            providers=['CUDAExecutionProvider']
        )
        self.app.prepare(ctx_id=0, det_size=(1280, 1280))

        # 4. 初始化 ArcFace 模型
        print("🔄 初始化 ArcFace 模型...")
        self.arcface_model = InceptionResnetV1(pretrained='vggface2').eval().to(self.device)

        # 数据预处理
        self.transform = transforms.Compose([
            transforms.Resize((160, 160)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

        # 5. 加载数据库
        self.dataset_path = dataset_path
        self.face_database = {}
        self._load_face_database()

    def _load_face_database(self):
        """加载人脸数据库并提取特征"""
        if not os.path.exists(self.dataset_path):
            print(f"❌ 数据库路径不存在: {self.dataset_path}")
            return

        person_folders = [f for f in os.listdir(self.dataset_path) 
                         if os.path.isdir(os.path.join(self.dataset_path, f))]
        
        for person_name in tqdm(person_folders, desc="提取库特征"):
            person_path = os.path.join(self.dataset_path, person_name)
            features = []
            img_list = list(Path(person_path).glob("*.jpg")) + list(Path(person_path).glob("*.png"))
            
            for img_path in img_list[:5]:
                try:
                    img = Image.open(img_path).convert('RGB')
                    img_tensor = self.transform(img).unsqueeze(0).to(self.device)
                    with torch.no_grad():
                        feature = self.arcface_model(img_tensor)
                        feature = F.normalize(feature, p=2, dim=1)
                        features.append(feature.cpu().numpy())
                except Exception as e:
                    print(f"⚠️ 库图片提取失败 {img_path}: {e}")
            
            if features:
                self.face_database[person_name] = {
                    'name': person_name,
                    'features': np.mean(features, axis=0)
                }
        print(f"✅ 数据库加载完成: {len(self.face_database)} 人")

    def detect_faces(self, image):
        try:
            faces = self.app.get(image)
            detected_faces = []
            for face in faces:
                if face.det_score < self.threshold: continue
                bbox = face.bbox.astype(int)
                x1, y1, x2, y2 = bbox
                h_img, w_img = image.shape[:2]
                x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w_img, x2), min(h_img, y2)
                face_img = image[y1:y2, x1:x2]
                if face_img.size > 0:
                    detected_faces.append({'bbox': (x1, y1, x2, y2), 'face_img': face_img})
            return detected_faces
        except Exception: return []

    def recognize_faces(self, image):
        """使用 PIL 绘制文本以支持中文和 Noto 字体"""
        faces = self.detect_faces(image)
        if not faces:
            return [], image

        # 转换为 PIL 图像进行绘制
        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)
        draw = ImageDraw.Draw(img_pil)
        
        results = []
        for i, face_info in enumerate(faces):
            bbox = face_info['bbox']
            face_img = face_info['face_img']
            
            # 特征识别
            face_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
            pil_face = Image.fromarray(face_rgb)
            img_tensor = self.transform(pil_face).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                feature = self.arcface_model(img_tensor)
                feature = F.normalize(feature, p=2, dim=1).cpu().numpy()

            best_match, max_sim = None, 0
            for person_name, info in self.face_database.items():
                sim = np.dot(feature, info['features'].T)[0][0]
                if sim > max_sim:
                    max_sim = sim
                    best_match = person_name if sim > self.arcface_threshold else None

            # 绘制逻辑
            x1, y1, x2, y2 = bbox
            # 颜色定义 (RGB)
            rect_color = (0, 255, 0) if best_match else (255, 0, 0)
            display_name = best_match if best_match else "未知"
            label = f"{display_name} {max_sim:.2f}"
            
            # 画框
            draw.rectangle([x1, y1, x2, y2], outline=rect_color, width=4)
            
            # 画文字背景块
            # 使用 textbbox 获取文字范围 (x0, y0, x1, y1)
            text_bbox = draw.textbbox((x1, y1 - self.font_size - 10), label, font=self.font)
            draw.rectangle(text_bbox, fill=rect_color)
            
            # 写字
            draw.text((x1, y1 - self.font_size - 10), label, font=self.font, fill=(255, 255, 255))
            
            results.append({'face_id': i+1, 'matched_person': best_match, 'score': float(max_sim)})

        # 转回 BGR 用于 OpenCV 保存
        final_img = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
        return results, final_img

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="/hdd/f25/zgq/retina_face_test/196")
    parser.add_argument("--dataset", default="/hdd/f25/zgq/retina_face_test/Dataset")
    parser.add_argument("--output", default="recognition_results")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    recognizer = FaceRecognizer(args.dataset)

    image_files = list(Path(args.input).glob("*.jpg")) + list(Path(args.input).glob("*.png"))
    for img_path in tqdm(image_files, desc="处理中"):
        img = cv2.imread(str(img_path))
        if img is None: continue
        _, vis_img = recognizer.recognize_faces(img)
        cv2.imwrite(os.path.join(args.output, f"res_{img_path.name}"), vis_img)
    print(f"🏁 完成！结果存放在: {args.output}")

if __name__ == "__main__":
    main()