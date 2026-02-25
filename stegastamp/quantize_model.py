#!/usr/bin/env python3
"""
模型量化脚本
将训练好的 StegaStamp 模型进行量化，减小模型大小（通常可减少 50-75%）
对模型准确性的影响通常很小（<1%）
"""

import argparse
import os
import sys
import torch
import torch.quantization

# 将项目根目录添加到 Python 路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from stegastamp.models import StegaStampEncoder, StegaStampDecoder


def quantize_model(
    input_path: str,
    output_path: str,
    height: int = 224,
    width: int = 224,
    secret_size: int = 64,
    verify: bool = True
):
    """
    量化 StegaStamp 模型
    
    Args:
        input_path: 输入模型路径（.pth 文件）
        output_path: 输出量化模型路径
        height: 图像高度
        width: 图像宽度
        secret_size: 秘密信息大小
        verify: 是否验证量化后的模型
    """
    print("=" * 60)
    print("StegaStamp 模型量化工具")
    print("=" * 60)
    
    # 检查输入文件
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"模型文件不存在: {input_path}")
    
    # 检查原始文件大小
    original_size = os.path.getsize(input_path) / (1024 * 1024)  # MB
    print(f"\n原始模型文件: {input_path}")
    print(f"原始模型大小: {original_size:.2f} MB")
    
    # 加载模型
    print("\n[1/4] 加载模型...")
    try:
        ckpt = torch.load(input_path, map_location="cpu", weights_only=False)
    except Exception as e:
        raise RuntimeError(f"加载模型失败: {e}")
    
    # 创建模型结构
    print(f"[2/4] 创建模型结构 (height={height}, width={width}, secret_size={secret_size})...")
    encoder = StegaStampEncoder(height=height, width=width, secret_size=secret_size)
    decoder = StegaStampDecoder(secret_size=secret_size, height=height, width=width)
    
    # 加载权重
    if "encoder" in ckpt:
        encoder.load_state_dict(ckpt["encoder"])
        print("  ✓ 加载 encoder 权重")
    else:
        raise ValueError("模型文件中未找到 'encoder' 权重")
    
    if "decoder" in ckpt:
        decoder.load_state_dict(ckpt["decoder"])
        print("  ✓ 加载 decoder 权重")
    else:
        raise ValueError("模型文件中未找到 'decoder' 权重")
    
    # 设置为评估模式（量化前必须）
    encoder.eval()
    decoder.eval()
    
    # 量化模型
    print("\n[3/4] 量化模型（使用动态量化）...")
    print("  - 量化 encoder...")
    encoder_quantized = torch.quantization.quantize_dynamic(
        encoder,
        {torch.nn.Linear, torch.nn.Conv2d},  # 只量化这些层类型
        dtype=torch.qint8
    )
    
    print("  - 量化 decoder...")
    decoder_quantized = torch.quantization.quantize_dynamic(
        decoder,
        {torch.nn.Linear, torch.nn.Conv2d},
        dtype=torch.qint8
    )
    
    # 保存量化模型
    print("\n[4/4] 保存量化模型...")
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    
    quantized_ckpt = {
        "encoder": encoder_quantized.state_dict(),
        "decoder": decoder_quantized.state_dict(),
    }
    
    # 如果原模型有 step 信息，保留它
    if "step" in ckpt:
        quantized_ckpt["step"] = ckpt["step"]
        print(f"  - 保留训练步数: {ckpt['step']}")
    
    torch.save(quantized_ckpt, output_path)
    print(f"  ✓ 保存到: {output_path}")
    
    # 检查量化后文件大小
    quantized_size = os.path.getsize(output_path) / (1024 * 1024)  # MB
    compression_ratio = original_size / quantized_size if quantized_size > 0 else 0
    space_saved = (1 - quantized_size / original_size) * 100 if original_size > 0 else 0
    
    print("\n" + "=" * 60)
    print("量化完成！")
    print("=" * 60)
    print(f"原始模型大小: {original_size:.2f} MB")
    print(f"量化后大小:   {quantized_size:.2f} MB")
    print(f"压缩比:       {compression_ratio:.2f}x")
    print(f"节省空间:     {space_saved:.1f}%")
    print("=" * 60)
    
    # 验证量化模型（可选）
    if verify:
        print("\n验证量化模型...")
        try:
            # 创建测试输入
            test_secret = torch.randint(0, 2, (1, secret_size), dtype=torch.float32)
            test_image = torch.rand(1, 3, height, width)
            
            # 测试 encoder
            with torch.no_grad():
                residual = encoder_quantized(test_secret, test_image)
                print("  ✓ Encoder 测试通过")
            
            # 测试 decoder
            with torch.no_grad():
                encoded = torch.clamp(test_image + residual, 0.0, 1.0)
                logits = decoder_quantized(encoded)
                print("  ✓ Decoder 测试通过")
            
            print("\n✓ 量化模型验证成功！")
        except Exception as e:
            print(f"\n⚠ 警告: 量化模型验证失败: {e}")
            print("   模型已保存，但建议检查是否正常工作")
    
    return quantized_size, compression_ratio, space_saved


def main():
    parser = argparse.ArgumentParser(
        description="量化 StegaStamp 模型以减小文件大小",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 量化默认配置的模型
  python -m stegastamp.quantize_model \\
    --input checkpoints/exp_name/latest.pth \\
    --output stegastamp_quantized.pt

  # 指定模型参数
  python -m stegastamp.quantize_model \\
    --input checkpoints/exp_name/latest.pth \\
    --output stegastamp_quantized.pt \\
    --height 224 --width 224 --secret_size 64

  # 不进行验证（更快）
  python -m stegastamp.quantize_model \\
    --input checkpoints/exp_name/latest.pth \\
    --output stegastamp_quantized.pt \\
    --no-verify
        """
    )
    
    parser.add_argument(
        "--input", "-i",
        type=str,
        required=True,
        help="输入模型路径（.pth 文件）"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        required=True,
        help="输出量化模型路径"
    )
    parser.add_argument(
        "--height",
        type=int,
        default=224,
        help="图像高度（默认: 224）"
    )
    parser.add_argument(
        "--width",
        type=int,
        default=224,
        help="图像宽度（默认: 224）"
    )
    parser.add_argument(
        "--secret_size",
        type=int,
        default=64,
        help="秘密信息大小（默认: 64）"
    )
    parser.add_argument(
        "--no-verify",
        action="store_true",
        help="不验证量化后的模型（更快）"
    )
    
    args = parser.parse_args()
    
    try:
        quantize_model(
            input_path=args.input,
            output_path=args.output,
            height=args.height,
            width=args.width,
            secret_size=args.secret_size,
            verify=not args.no_verify
        )
    except Exception as e:
        print(f"\n❌ 错误: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
