#!/usr/bin/env python3
"""
生成训练图片集脚本
基于现有图片生成随机涂鸦图片
"""

import os
import sys
import random
import argparse

try:
    from PIL import Image, ImageDraw, ImageFilter
except ImportError:
    print("错误: 需要安装 Pillow 库")
    print("请运行: pip install Pillow")
    sys.exit(1)

def random_scribble(img):
    """
    在图片上随机涂鸦
    添加随机线条、形状等
    """
    img_copy = img.copy()
    draw = ImageDraw.Draw(img_copy)
    width, height = img_copy.size
    
    # 随机生成多条涂鸦线条
    num_lines = random.randint(5, 20)
    for _ in range(num_lines):
        # 随机颜色（可以是任意颜色）
        color = (
            random.randint(0, 255),
            random.randint(0, 255),
            random.randint(0, 255)
        )
        
        # 随机线条宽度
        line_width = random.randint(1, 5)
        
        # 随机起点和终点
        x1 = random.randint(0, width)
        y1 = random.randint(0, height)
        x2 = random.randint(0, width)
        y2 = random.randint(0, height)
        
        draw.line([(x1, y1), (x2, y2)], fill=color, width=line_width)
    
    # 随机添加一些圆形或矩形
    num_shapes = random.randint(3, 10)
    for _ in range(num_shapes):
        color = (
            random.randint(0, 255),
            random.randint(0, 255),
            random.randint(0, 255)
        )
        
        shape_type = random.choice(['circle', 'rectangle'])
        if shape_type == 'circle':
            center_x = random.randint(0, width)
            center_y = random.randint(0, height)
            radius = random.randint(5, min(width, height) // 10)
            bbox = (center_x - radius, center_y - radius, 
                   center_x + radius, center_y + radius)
            draw.ellipse(bbox, fill=color, outline=color)
        else:
            x1 = random.randint(0, width)
            y1 = random.randint(0, height)
            x2 = random.randint(0, width)
            y2 = random.randint(0, height)
            # 确保矩形坐标顺序正确（左上角到右下角）
            x_min, x_max = min(x1, x2), max(x1, x2)
            y_min, y_max = min(y1, y2), max(y1, y2)
            draw.rectangle([x_min, y_min, x_max, y_max], fill=color, outline=color)
    
    return img_copy

def main():
    parser = argparse.ArgumentParser(description="生成训练图片集")
    parser.add_argument("--source_image", type=str, 
                       default="./asset/images_noise_1x/train/class1/train_image1.png",
                       help="源图片路径")
    parser.add_argument("--output_dir", type=str,
                       default="./asset/images_noise_1x/train/class1",
                       help="输出目录")
    parser.add_argument("--num_scribble", type=int, default=100,
                       help="生成的随机涂鸦图片数量")
    parser.add_argument("--seed", type=int, default=42,
                       help="随机种子")
    args = parser.parse_args()
    
    # 设置随机种子
    random.seed(args.seed)
    
    # 检查源图片是否存在
    if not os.path.exists(args.source_image):
        print(f"错误: 源图片不存在: {args.source_image}")
        sys.exit(1)
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 读取源图片
    print(f"读取源图片: {args.source_image}")
    try:
        source_img = Image.open(args.source_image).convert("RGB")
        print(f"源图片尺寸: {source_img.size}")
    except Exception as e:
        print(f"错误: 无法读取源图片: {e}")
        sys.exit(1)
    
    # 生成随机涂鸦图片
    print(f"\n生成 {args.num_scribble} 张随机涂鸦图片...")
    for i in range(args.num_scribble):
        scribbled = random_scribble(source_img)
        output_path = os.path.join(args.output_dir, f"scribble_{i+1:03d}.png")
        scribbled.save(output_path)
        if (i + 1) % 10 == 0:
            print(f"  已生成 {i + 1}/{args.num_scribble} 张")
    
    print(f"\n完成! 所有图片已保存到: {args.output_dir}")
    print(f"  - 随机涂鸦图片: {args.num_scribble} 张")

if __name__ == "__main__":
    main()
