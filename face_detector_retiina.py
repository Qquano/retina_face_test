#!/usr/bin/env python3
"""
使用RetinaFace检测人脸并保存框选图像
"""

import os
import cv2
import numpy as np
from retinaface import RetinaFace
from pathlib import Path
import argparse
from tqdm import tqdm

class FaceDetector:
    def __init__(self, threshold=0.5, save_boxes=True, show_score=True):
        """
        初始化人脸检测器
        
        Args:
            threshold: 置信度阈值，默认0.5
            save_boxes: 是否保存框选图像
            show_score: 是否在图像上显示置信度
        """
        self.threshold = threshold
        self.save_boxes = save_boxes
        self.show_score = show_score
        
        # 初始化检测器
        print("🔄 初始化RetinaFace检测器...")
        self.detector = RetinaFace.build_model()
        print("✅ 检测器初始化完成")
    
    def detect_faces_in_image(self, image_path):
        """
        检测单张图像中的人脸
        
        Args:
            image_path: 图像路径
            
        Returns:
            faces: 检测到的人脸信息
            image_with_boxes: 带框选的图像
        """
        # 读取图像
        img = cv2.imread(str(image_path))
        if img is None:
            print(f"❌ 无法读取图像: {image_path}")
            return None, None
        
        # 使用RetinaFace检测人脸
        faces = RetinaFace.detect_faces(img, threshold=self.threshold)
        
        # 复制图像用于绘制
        img_with_boxes = img.copy()
        
        if isinstance(faces, dict) and faces:
            # 检测到人脸
            face_count = len(faces)
            
            for face_id, face_info in faces.items():
                # 获取人脸框和置信度
                facial_area = face_info['facial_area']
                score = face_info['score']
                
                # 绘制人脸框
                x1, y1, x2, y2 = facial_area
                cv2.rectangle(img_with_boxes, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # 显示置信度
                if self.show_score:
                    cv2.putText(img_with_boxes, f'{score:.3f}', 
                              (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                              0.5, (0, 255, 0), 1)
                
                # 绘制关键点（可选）
                landmarks = face_info['landmarks']
                colors = {'left_eye': (255, 0, 0),   # 蓝色
                         'right_eye': (0, 0, 255),  # 红色
                         'nose': (0, 255, 0),       # 绿色
                         'mouth_left': (0, 255, 255),  # 黄色
                         'mouth_right': (255, 0, 255)}  # 紫色
                
                for landmark_name, point in landmarks.items():
                    color = colors.get(landmark_name, (255, 255, 255))
                    cv2.circle(img_with_boxes, 
                              (int(point[0]), int(point[1])), 
                              3, color, -1)
            
            # 添加统计信息
            cv2.putText(img_with_boxes, f'Faces: {face_count}', 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                       1, (0, 0, 255), 2)
        else:
            # 未检测到人脸
            cv2.putText(img_with_boxes, 'No Face Detected', 
                       (img.shape[1]//4, img.shape[0]//2), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            faces = None
        
        return faces, img_with_boxes
    
    def save_detection_result(self, original_image, image_with_boxes, faces, 
                             output_path, save_crops=False, crop_output_dir=None):
        """
        保存检测结果
        
        Args:
            original_image: 原始图像
            image_with_boxes: 带框选的图像
            faces: 检测到的人脸信息
            output_path: 输出路径
            save_crops: 是否保存裁剪的人脸
            crop_output_dir: 人脸裁剪保存目录
        """
        try:
            # 保存带框选的图像
            cv2.imwrite(output_path, image_with_boxes, 
                       [cv2.IMWRITE_JPEG_QUALITY, 95])
            
            # 如果需要保存裁剪的人脸
            if save_crops and faces and isinstance(faces, dict):
                os.makedirs(crop_output_dir, exist_ok=True)
                
                for i, (face_id, face_info) in enumerate(faces.items()):
                    facial_area = face_info['facial_area']
                    x1, y1, x2, y2 = facial_area
                    
                    # 裁剪人脸
                    face_crop = original_image[y1:y2, x1:x2]
                    
                    if face_crop.size > 0:
                        # 保存裁剪的人脸
                        crop_filename = f"face_{i+1:03d}.jpg"
                        crop_path = os.path.join(crop_output_dir, crop_filename)
                        cv2.imwrite(crop_path, face_crop, 
                                   [cv2.IMWRITE_JPEG_QUALITY, 95])
            
            return True
        except Exception as e:
            print(f"❌ 保存结果失败: {e}")
            return False

def process_folder(input_folder, output_folder, threshold=0.5, 
                   save_crops=False, crop_prefix="crop"):
    """
    处理整个文件夹的图像
    
    Args:
        input_folder: 输入文件夹路径
        output_folder: 输出文件夹路径
        threshold: 检测阈值
        save_crops: 是否保存裁剪的人脸
        crop_prefix: 人脸裁剪文件夹前缀
    """
    print("🎯 开始处理监控图像...")
    print(f"📁 输入文件夹: {input_folder}")
    print(f"📁 输出文件夹: {output_folder}")
    print(f"📊 检测阈值: {threshold}")
    print("-" * 60)
    
    # 检查输入文件夹
    if not os.path.exists(input_folder):
        print(f"❌ 输入文件夹不存在: {input_folder}")
        return
    
    # 创建输出文件夹
    os.makedirs(output_folder, exist_ok=True)
    
    # 初始化检测器
    detector = FaceDetector(threshold=threshold)
    
    # 支持的图像格式
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', 
                       '.JPG', '.JPEG', '.PNG', '.BMP'}
    
    # 获取所有图像文件
    image_files = []
    for ext in image_extensions:
        image_files.extend(Path(input_folder).glob(f'*{ext}'))
    
    if not image_files:
        print(f"❌ 未找到图像文件: {input_folder}")
        return
    
    print(f"📊 找到 {len(image_files)} 个图像文件")
    
    # 统计数据
    stats = {
        'total_images': 0,
        'detected_images': 0,
        'no_face_images': 0,
        'total_faces': 0,
        'failed_images': 0
    }
    
    # 处理每个图像
    for image_path in tqdm(image_files, desc="检测人脸"):
        stats['total_images'] += 1
        
        try:
            # 检测人脸
            faces, img_with_boxes = detector.detect_faces_in_image(image_path)
            
            if faces is None:
                stats['no_face_images'] += 1
            else:
                stats['detected_images'] += 1
                stats['total_faces'] += len(faces)
            
            # 构建输出路径
            output_filename = f"detected_{image_path.stem}.jpg"
            output_path = os.path.join(output_folder, output_filename)
            
            # 如果需要保存裁剪的人脸
            crop_output_dir = None
            if save_crops and faces:
                crop_dir_name = f"{crop_prefix}_{image_path.stem}"
                crop_output_dir = os.path.join(output_folder, crop_dir_name)
            
            # 读取原始图像用于保存裁剪
            img_original = cv2.imread(str(image_path))
            
            # 保存结果
            detector.save_detection_result(
                img_original, img_with_boxes, faces, 
                output_path, save_crops, crop_output_dir
            )
            
        except Exception as e:
            stats['failed_images'] += 1
            print(f"❌ 处理失败 {image_path.name}: {e}")
    
    # 打印统计报告
    print("\n" + "=" * 60)
    print("📊 检测完成！")
    print("=" * 60)
    print(f"📈 统计信息:")
    print(f"  📂 总图像数: {stats['total_images']}")
    print(f"  ✅ 检测到人脸的图像: {stats['detected_images']}")
    print(f"  ⚠️  未检测到人脸的图像: {stats['no_face_images']}")
    print(f"  👤 检测到的总人脸数: {stats['total_faces']}")
    print(f"  ❌ 处理失败的图像: {stats['failed_images']}")
    
    if stats['detected_images'] > 0:
        avg_faces = stats['total_faces'] / stats['detected_images']
        print(f"  📊 平均每张图像人脸数: {avg_faces:.1f}")
    
    print(f"\n📁 结果保存到: {os.path.abspath(output_folder)}")
    
    # 生成报告文件
    report_path = os.path.join(output_folder, "detection_report.txt")
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("人脸检测报告\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"输入文件夹: {input_folder}\n")
        f.write(f"输出文件夹: {output_folder}\n")
        f.write(f"检测阈值: {threshold}\n\n")
        f.write("统计信息:\n")
        f.write(f"  总图像数: {stats['total_images']}\n")
        f.write(f"  检测到人脸的图像: {stats['detected_images']}\n")
        f.write(f"  未检测到人脸的图像: {stats['no_face_images']}\n")
        f.write(f"  检测到的总人脸数: {stats['total_faces']}\n")
        f.write(f"  处理失败的图像: {stats['failed_images']}\n")
        if stats['detected_images'] > 0:
            f.write(f"  平均每张图像人脸数: {stats['total_faces']/stats['detected_images']:.1f}\n")
    
    print(f"📄 详细报告: {report_path}")

def single_image_demo(image_path, output_path=None, threshold=0.5):
    """
    单张图像演示
    
    Args:
        image_path: 图像路径
        output_path: 输出路径
        threshold: 检测阈值
    """
    print(f"🔍 单张图像演示: {image_path}")
    print(f"📊 检测阈值: {threshold}")
    print("-" * 60)
    
    if not os.path.exists(image_path):
        print(f"❌ 图像不存在: {image_path}")
        return
    
    # 初始化检测器
    detector = FaceDetector(threshold=threshold, show_score=True)
    
    # 检测人脸
    faces, img_with_boxes = detector.detect_faces_in_image(image_path)
    
    if faces is None:
        print("❌ 未检测到人脸")
    else:
        print(f"✅ 检测到 {len(faces)} 张人脸")
        
        # 显示详细信息
        for i, (face_id, face_info) in enumerate(faces.items()):
            facial_area = face_info['facial_area']
            score = face_info['score']
            x1, y1, x2, y2 = facial_area
            width = x2 - x1
            height = y2 - y1
            
            print(f"\n👤 人脸 {i+1}:")
            print(f"  🎯 置信度: {score:.4f}")
            print(f"  📍 位置: ({x1}, {y1}) - ({x2}, {y2})")
            print(f"  📏 大小: {width}×{height} 像素")
    
    # 保存或显示结果
    if output_path:
        # 保存结果
        img_original = cv2.imread(image_path)
        detector.save_detection_result(img_original, img_with_boxes, faces, output_path)
        print(f"✅ 结果已保存: {output_path}")
    else:
        # 显示结果
        cv2.imshow("Face Detection Result", img_with_boxes)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

def main():
    parser = argparse.ArgumentParser(description="使用RetinaFace检测人脸并保存框选图像")
    parser.add_argument("input", help="输入文件夹或图像路径"，default="196")
    parser.add_argument("-o", "--output", default="detection_results", 
                       help="输出文件夹路径，默认: detection_results")
    parser.add_argument("-t", "--threshold", type=float, default=0.6,
                       help="检测阈值，默认: 0.5")
    parser.add_argument("--save-crops", action="store_true",
                       help="保存裁剪的人脸图像")
    parser.add_argument("--crop-prefix", default="crop",
                       help="人脸裁剪文件夹前缀，默认: crop")
    parser.add_argument("--demo", action="store_true",
                       help="单张图像演示模式")
    
    args = parser.parse_args()
    
    # 检查依赖
    try:
        from retinaface import RetinaFace
    except ImportError:
        print("❌ 需要安装 retina-face")
        print("安装命令: pip install retina-face")
        exit(1)
    
    print("🎯 RetinaFace人脸检测器")
    print("=" * 60)
    
    if args.demo:
        # 单张图像演示模式
        if os.path.isfile(args.input):
            output_path = f"detected_{Path(args.input).stem}.jpg"
            single_image_demo(args.input, output_path, args.threshold)
        else:
            print(f"❌ 不是有效的图像文件: {args.input}")
    else:
        # 批量处理模式
        if os.path.isdir(args.input):
            process_folder(args.input, args.output, args.threshold, 
                         args.save_crops, args.crop_prefix)
        elif os.path.isfile(args.input):
            print("⚠️  输入是单个文件，使用 --demo 参数进行演示")
            print(f"  或将其放入文件夹中批量处理")
        else:
            print(f"❌ 输入路径不存在: {args.input}")

if __name__ == "__main__":
    main()