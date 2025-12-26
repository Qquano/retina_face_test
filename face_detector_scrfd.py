#!/usr/bin/env python3
"""
使用SCRFD检测监控图像中的人脸
SCRFD: Sample and Computation Redistribution for Efficient Face Detection
特点：轻量、快速、准确，适合监控场景
"""

import os
import cv2
import numpy as np
from pathlib import Path
import argparse
from tqdm import tqdm

class SCRFDFaceDetector:
    """SCRFD人脸检测器"""
    
    def __init__(self, model_path=None, conf_threshold=0.5, nms_threshold=0.5, 
                 use_gpu=False, model_name='scrfd_10g', input_size=(1280, 1280)):
        """
        初始化SCRFD检测器
        
        Args:
            model_path: 模型文件路径，如果为None则使用默认模型
            conf_threshold: 置信度阈值
            nms_threshold: NMS阈值
            use_gpu: 是否使用GPU
            model_name: 模型名称，可选: 'scrfd_500m', 'scrfd_2.5g', 'scrfd_10g'
            input_size: 输入尺寸 (width, height)
        """
        self.conf_threshold = conf_threshold
        self.nms_threshold = nms_threshold
        self.use_gpu = use_gpu
        self.model_name = model_name
        self.fixed_input_size = input_size  # 保存固定的输入尺寸
        self.model_path = model_path  # 保存模型路径
        
        print(f"🔄 初始化SCRFD检测器 (模型: {model_name}, 输入尺寸: {input_size})...")
        self.net = self._load_model()
        print("✅ SCRFD检测器初始化完成")
    
    def _load_model(self):
        """
        加载SCRFD模型
        """
        try:
            import onnxruntime as ort
            
            # 确定模型路径
            model_path = self.model_path
            
            # 如果没有指定模型路径，使用内置的模型名称
            if model_path is None:
                # 本地模型文件路径
                model_dir = os.path.join(os.path.expanduser("~"), ".scrfd_models")
                os.makedirs(model_dir, exist_ok=True)
                model_path = os.path.join(model_dir, f"{self.model_name}.onnx")
                
                if not os.path.exists(model_path):
                    print(f"⚠️  模型文件不存在: {model_path}")
                    print(f"💡 请从以下地址下载模型:")
                    print(f"   https://github.com/deepinsight/insightface/releases/download/v0.7/scrfd_10g.onnx")
                    print(f"💡 或运行: wget https://github.com/deepinsight/insightface/releases/download/v0.7/scrfd_10g.onnx -O {model_path}")
                    raise FileNotFoundError(f"模型文件不存在: {model_path}")
            
            # 检查模型文件
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"模型文件不存在: {model_path}")
            
            print(f"📁 加载模型: {os.path.basename(model_path)}")
            
            # 配置ONNX Runtime
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if self.use_gpu else ['CPUExecutionProvider']
            
            # 创建推理会话
            session = ort.InferenceSession(model_path, providers=providers)
            
            # 获取输入信息
            self.input_name = session.get_inputs()[0].name
            
            # 重要：使用固定的输入尺寸，不要从模型中获取动态尺寸
            # SCRFD模型支持动态输入，但我们需要指定固定尺寸
            print(f"✅ 模型加载成功，使用固定输入尺寸: {self.fixed_input_size}")
            
            return session
            
        except ImportError:
            print("❌ 需要安装 onnxruntime")
            print("安装命令: pip install onnxruntime")
            if self.use_gpu:
                print("GPU版本: pip install onnxruntime-gpu")
            raise
        except Exception as e:
            print(f"❌ 加载模型失败: {e}")
            raise
    
    def _preprocess(self, image):
        """预处理图像"""
        # 确保输入尺寸是有效的整数
        target_width = int(self.fixed_input_size[0])
        target_height = int(self.fixed_input_size[1])
        
        # 调整尺寸
        img_resized = cv2.resize(image, (target_width, target_height))
        
        # 转换为RGB
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        
        # 归一化 (SCRFD使用的归一化方式)
        img_normalized = img_rgb.astype(np.float32)
        img_normalized = (img_normalized - 127.5) / 128.0
        
        # 调整维度顺序: HWC -> NCHW
        img_transposed = np.transpose(img_normalized, (2, 0, 1))
        img_batch = np.expand_dims(img_transposed, axis=0)
        
        return img_batch, img_resized
    
    def _postprocess(self, outputs, original_size, resized_size):
        """后处理检测结果"""
        # SCRFD输出有9个，分别是不同尺度的分类分数、边界框和关键点
        # 我们需要处理所有尺度的输出
        
        all_detections = []
        
        # SCRFD输出索引：3个尺度，每个尺度有3个输出（分类、边界框、关键点）
        # 共9个输出：[score1, score2, score3, bbox1, bbox2, bbox3, landmark1, landmark2, landmark3]
        
        # 遍历3个尺度
        for scale_idx in range(3):
            score_idx = scale_idx  # 分类分数索引
            bbox_idx = scale_idx + 3  # 边界框索引
            landmark_idx = scale_idx + 6  # 关键点索引
            
            scores = outputs[score_idx][0]  # 形状: [N, 1]
            bboxes = outputs[bbox_idx][0]  # 形状: [N, 4]
            
            # 将scores从[N, 1]转换为[N]
            scores = scores.flatten()
            
            # 过滤低置信度的检测
            keep_indices = scores > self.conf_threshold
            
            if not np.any(keep_indices):
                continue
            
            scale_scores = scores[keep_indices]
            scale_bboxes = bboxes[keep_indices]
            
            # 应用NMS
            indices = self._nms(scale_bboxes, scale_scores)
            
            for idx in indices:
                score = scale_scores[idx]
                bbox = scale_bboxes[idx]
                
                # 将边界框从resized尺寸映射回原始尺寸
                x1, y1, x2, y2 = bbox[:4]
                
                # 计算缩放比例
                scale_x = original_size[0] / resized_size[0]
                scale_y = original_size[1] / resized_size[1]
                
                # 映射到原始图像
                x1 = int(x1 * scale_x)
                y1 = int(y1 * scale_y)
                x2 = int(x2 * scale_x)
                y2 = int(y2 * scale_y)
                
                # 确保边界框在图像范围内
                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(original_size[0], x2)
                y2 = min(original_size[1], y2)
                
                # 确保边界框有效
                if x2 > x1 and y2 > y1:
                    all_detections.append({
                        'bbox': (x1, y1, x2, y2),
                        'score': float(score),
                        'width': x2 - x1,
                        'height': y2 - y1
                    })
        
        return all_detections
    
    def _nms(self, boxes, scores):
        """非极大值抑制"""
        if len(boxes) == 0:
            return []
        
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]
        
        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        order = scores.argsort()[::-1]
        
        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
            
            w = np.maximum(0.0, xx2 - xx1 + 1)
            h = np.maximum(0.0, yy2 - yy1 + 1)
            inter = w * h
            
            ovr = inter / (areas[i] + areas[order[1:]] - inter)
            
            inds = np.where(ovr <= self.nms_threshold)[0]
            order = order[inds + 1]
        
        return keep
    
    def detect(self, image):
        """
        检测单张图像中的人脸
        
        Args:
            image: 输入图像 (BGR格式)
            
        Returns:
            detections: 检测结果列表，每个元素包含bbox, score等信息
        """
        original_size = (image.shape[1], image.shape[0])  # (width, height)
        
        # 预处理
        input_data, resized_img = self._preprocess(image)
        resized_size = (resized_img.shape[1], resized_img.shape[0])
        
        # 推理
        outputs = self.net.run(None, {self.input_name: input_data})
        
        # 后处理
        detections = self._postprocess(outputs, original_size, resized_size)
        
        return detections
    
    def detect_from_file(self, image_path):
        """
        从文件检测人脸
        
        Args:
            image_path: 图像文件路径
            
        Returns:
            detections: 检测结果
            image: 原始图像
        """
        # 读取图像
        image = cv2.imread(str(image_path))
        if image is None:
            print(f"❌ 无法读取图像: {image_path}")
            return None, None
        
        # 检测人脸
        detections = self.detect(image)
        
        return detections, image
    
    def draw_detections(self, image, detections):
        """
        在图像上绘制检测结果
        
        Args:
            image: 原始图像
            detections: 检测结果
            
        Returns:
            绘制了检测框的图像
        """
        img_with_boxes = image.copy()
        
        if detections:
            for i, det in enumerate(detections):
                bbox = det['bbox']
                score = det['score']
                x1, y1, x2, y2 = bbox
                
                # 绘制边界框
                color = (0, 255, 0)  # 绿色
                thickness = 2
                cv2.rectangle(img_with_boxes, (x1, y1), (x2, y2), color, thickness)
                
                # 绘制置信度
                label = f"{score:.3f}"
                label_size, base_line = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                y_label = max(y1, label_size[1] + 10)
                
                # 绘制背景
                cv2.rectangle(img_with_boxes, 
                            (x1, y_label - label_size[1] - 10),
                            (x1 + label_size[0], y_label + base_line - 10),
                            color, cv2.FILLED)
                
                # 绘制文本
                cv2.putText(img_with_boxes, label, 
                          (x1, y_label - 7), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
            
            # 添加统计信息
            cv2.putText(img_with_boxes, f'Faces: {len(detections)}', 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                       1, (0, 0, 255), 2)
        else:
            # 未检测到人脸
            cv2.putText(img_with_boxes, 'No Face Detected', 
                       (image.shape[1]//4, image.shape[0]//2), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        return img_with_boxes

def process_monitoring_images(input_folder, output_folder, 
                            conf_threshold=0.5, nms_threshold=0.5,
                            save_crops=False, crop_prefix="face",
                            model_name='scrfd_10g', use_gpu=False,
                            input_size=(1280,1280), model_path=None):
    """
    处理监控图像文件夹
    
    Args:
        input_folder: 输入文件夹路径
        output_folder: 输出文件夹路径
        conf_threshold: 置信度阈值
        nms_threshold: NMS阈值
        save_crops: 是否保存裁剪的人脸
        crop_prefix: 人脸裁剪文件名前缀
        model_name: SCRFD模型名称
        use_gpu: 是否使用GPU
        input_size: 输入图像尺寸 (width, height)
        model_path: 模型文件路径
    """
    print("🎯 开始处理监控图像...")
    print(f"📁 输入文件夹: {input_folder}")
    print(f"📁 输出文件夹: {output_folder}")
    print(f"📊 置信度阈值: {conf_threshold}")
    print(f"📊 NMS阈值: {nms_threshold}")
    print(f"🤖 模型: {model_name}")
    if model_path:
        print(f"📁 模型路径: {model_path}")
    print(f"📏 输入尺寸: {input_size}")
    print(f"⚡ GPU加速: {'是' if use_gpu else '否'}")
    print("-" * 60)
    
    # 检查输入文件夹
    if not os.path.exists(input_folder):
        print(f"❌ 输入文件夹不存在: {input_folder}")
        return
    
    # 创建输出文件夹
    os.makedirs(output_folder, exist_ok=True)
    
    # 初始化检测器
    try:
        detector = SCRFDFaceDetector(
            model_path=model_path,
            conf_threshold=conf_threshold,
            nms_threshold=nms_threshold,
            use_gpu=use_gpu,
            model_name=model_name,
            input_size=input_size
        )
    except Exception as e:
        print(f"❌ 无法初始化检测器: {e}")
        print("\n💡 解决方案:")
        print("1. 安装依赖: pip install onnxruntime opencv-python")
        print("2. 确保模型文件存在")
        if model_path:
            print(f"   指定路径: {model_path}")
        else:
            print("   默认路径: ~/.scrfd_models/scrfd_10g.onnx")
        return
    
    # 支持的图像格式
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff',
                       '.JPG', '.JPEG', '.PNG', '.BMP', '.TIFF'}
    
    # 获取所有图像文件
    image_files = []
    for ext in image_extensions:
        image_files.extend(Path(input_folder).glob(f'*{ext}'))
        image_files.extend(Path(input_folder).glob(f'*{ext.lower()}'))
    
    # 去重
    image_files = list(set(image_files))
    
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
            detections, image = detector.detect_from_file(image_path)
            
            if image is None:
                stats['failed_images'] += 1
                continue
            
            if not detections:
                stats['no_face_images'] += 1
            else:
                stats['detected_images'] += 1
                stats['total_faces'] += len(detections)
            
            # 绘制检测结果
            image_with_boxes = detector.draw_detections(image, detections)
            
            # 保存结果图像
            output_filename = f"detected_{image_path.stem}.jpg"
            output_path = os.path.join(output_folder, output_filename)
            cv2.imwrite(output_path, image_with_boxes, [cv2.IMWRITE_JPEG_QUALITY, 95])
            
            # 保存裁剪的人脸
            if save_crops and detections:
                crop_dir = os.path.join(output_folder, f"{crop_prefix}_{image_path.stem}")
                os.makedirs(crop_dir, exist_ok=True)
                
                for i, det in enumerate(detections):
                    x1, y1, x2, y2 = det['bbox']
                    face_crop = image[y1:y2, x1:x2]
                    
                    if face_crop.size > 0:
                        crop_filename = f"{crop_prefix}_{i+1:03d}.jpg"
                        crop_path = os.path.join(crop_dir, crop_filename)
                        cv2.imwrite(crop_path, face_crop, [cv2.IMWRITE_JPEG_QUALITY, 95])
            
        except Exception as e:
            stats['failed_images'] += 1
            print(f"❌ 处理失败 {image_path.name}: {e}")
    
    # 打印统计报告
    print("\n" + "=" * 60)
    print("📊 SCRFD检测完成！")
    print("=" * 60)
    print(f"📈 统计信息:")
    print(f"  总图像数: {stats['total_images']}")
    print(f"  检测到人脸的图像: {stats['detected_images']}")
    print(f"  未检测到人脸的图像: {stats['no_face_images']}")
    print(f"  检测到的总人脸数: {stats['total_faces']}")
    print(f"  处理失败的图像: {stats['failed_images']}")
    
    if stats['detected_images'] > 0:
        avg_faces = stats['total_faces'] / stats['detected_images']
        print(f"  平均每张图像人脸数: {avg_faces:.1f}")
    
    print(f"\n📁 结果保存到: {os.path.abspath(output_folder)}")
    
    # 生成报告文件
    report_path = os.path.join(output_folder, "scrfd_detection_report.txt")
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("SCRFD人脸检测报告\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"输入文件夹: {input_folder}\n")
        f.write(f"输出文件夹: {output_folder}\n")
        f.write(f"模型: {model_name}\n")
        if model_path:
            f.write(f"模型路径: {model_path}\n")
        f.write(f"输入尺寸: {input_size}\n")
        f.write(f"置信度阈值: {conf_threshold}\n")
        f.write(f"NMS阈值: {nms_threshold}\n")
        f.write(f"GPU加速: {use_gpu}\n\n")
        f.write("统计信息:\n")
        f.write(f"  总图像数: {stats['total_images']}\n")
        f.write(f"  检测到人脸的图像: {stats['detected_images']}\n")
        f.write(f"  未检测到人脸的图像: {stats['no_face_images']}\n")
        f.write(f"  检测到的总人脸数: {stats['total_faces']}\n")
        f.write(f"  处理失败的图像: {stats['failed_images']}\n")
        if stats['detected_images'] > 0:
            f.write(f"  平均每张图像人脸数: {stats['total_faces']/stats['detected_images']:.1f}\n")
    
    print(f"📄 详细报告: {report_path}")

def main():
    parser = argparse.ArgumentParser(description="使用SCRFD检测监控图像中的人脸")
    parser.add_argument("input", help="监控图像文件夹路径", nargs='?', default="196")
    parser.add_argument("-o", "--output", default="scrfd_results", 
                       help="输出文件夹路径，默认: scrfd_results")
    parser.add_argument("-c", "--conf", type=float, default=0.5,
                       help="置信度阈值，默认: 0.5")
    parser.add_argument("-n", "--nms", type=float, default=0.5,
                       help="NMS阈值，默认: 0.5")
    parser.add_argument("-m", "--model", default="scrfd_10g",
                       choices=['scrfd_500m', 'scrfd_2.5g', 'scrfd_10g'], 
                       help="SCRFD模型，默认: scrfd_10g")
    parser.add_argument("--model-path", default="/hdd/f25/zgq/retina_face_test/scrfd_10g_bnkps.onnx",
                       help="SCRFD模型文件路径，默认: /hdd/f25/zgq/retina_face_test/scrfd_10g_bnkps.onnx")
    parser.add_argument("--input-size", type=int, nargs=2, default=[1280, 1280],
                       metavar=('WIDTH', 'HEIGHT'),
                       help="输入图像尺寸，默认: 640 640")
    parser.add_argument("--gpu", action="store_true",
                       help="使用GPU加速（需要onnxruntime-gpu）")
    parser.add_argument("--save-crops", action="store_true",
                       help="保存裁剪的人脸图像")
    
    args = parser.parse_args()
    
    print("🤖 SCRFD人脸检测器 - 专为监控场景优化")
    print("=" * 60)
    
    # 检查输入路径
    if not os.path.exists(args.input):
        print(f"❌ 输入路径不存在: {args.input}")
        return
    
    # 处理图像
    if os.path.isdir(args.input):
        process_monitoring_images(
            input_folder=args.input,
            output_folder=args.output,
            conf_threshold=args.conf,
            nms_threshold=args.nms,
            save_crops=args.save_crops,
            model_name=args.model,
            use_gpu=args.gpu,
            input_size=tuple(args.input_size),
            model_path=args.model_path
        )
    elif os.path.isfile(args.input):
        # 单张图像处理
        print(f"🔍 处理单张图像: {args.input}")
        
        # 初始化检测器
        detector = SCRFDFaceDetector(
            model_path=args.model_path,  # 使用args.model_path
            conf_threshold=args.conf,
            nms_threshold=args.nms,
            use_gpu=args.gpu,
            model_name=args.model,
            input_size=tuple(args.input_size)
        )
        
        # 检测人脸
        detections, image = detector.detect_from_file(args.input)
        
        if image is None:
            print("❌ 无法读取图像")
            return
        
        if detections:
            print(f"✅ 检测到 {len(detections)} 张人脸")
            for i, det in enumerate(detections):
                print(f"  👤 人脸 {i+1}: 置信度={det['score']:.3f}, 位置={det['bbox']}, 大小={det['width']}x{det['height']}")
        else:
            print("❌ 未检测到人脸")
        
        # 保存结果
        output_path = f"detected_{Path(args.input).stem}.jpg"
        image_with_boxes = detector.draw_detections(image, detections)
        cv2.imwrite(output_path, image_with_boxes, [cv2.IMWRITE_JPEG_QUALITY, 95])
        print(f"✅ 结果已保存: {output_path}")
    else:
        print(f"❌ 无效的输入路径: {args.input}")

if __name__ == "__main__":
    # 检查依赖
    try:
        import onnxruntime
    except ImportError:
        print("❌ 需要安装 onnxruntime")
        print("安装命令:")
        print("  CPU版本: pip install onnxruntime")
        print("  GPU版本: pip install onnxruntime-gpu")
        exit(1)
    
    main()