# 人脸数据集

## 目录结构
```
Dataset/
├── Person_Name_1/
│   ├── crop_001.jpg
│   ├── crop_002.jpg
│   └── ref_high_res.jpg  # 来自tczs_staff的高清参考图
├── Person_Name_2/
│   └── ...
└── ...
```

## 文件说明
- `crop_*.jpg`: 从原始大图中裁剪的人脸图像
- `ref_high_res.jpg`: 来自tczs_staff文件夹的高清人脸参考图

## 数据来源
- 原始标注: labels_my-project-name_2025-12-17-01-57-45_all.csv
- 高清参考图: tczs_staff_all
- 构建时间: 2025-12-25 16:33:30
