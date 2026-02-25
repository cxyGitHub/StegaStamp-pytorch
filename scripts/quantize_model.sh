#!/usr/bin/env bash

# 模型量化脚本
# 用法: bash scripts/quantize_model.sh <input_model> <output_model> [height] [width] [secret_size]

set -euo pipefail

INPUT_MODEL=${1:-}
OUTPUT_MODEL=${2:-}
HEIGHT=${3:-224}
WIDTH=${4:-224}
SECRET_SIZE=${5:-64}

if [ -z "$INPUT_MODEL" ] || [ -z "$OUTPUT_MODEL" ]; then
    echo "用法: $0 <input_model> <output_model> [height] [width] [secret_size]"
    echo ""
    echo "示例:"
    echo "  $0 checkpoints/exp_name/latest.pth stegastamp_quantized.pt"
    echo "  $0 checkpoints/exp_name/latest.pth stegastamp_quantized.pt 224 224 64"
    exit 1
fi

# 解析脚本与工程根路径
SCRIPT_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/$(basename "${BASH_SOURCE[0]}")"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

echo "量化模型: $INPUT_MODEL"
echo "输出到: $OUTPUT_MODEL"
echo "配置: height=$HEIGHT, width=$WIDTH, secret_size=$SECRET_SIZE"
echo ""

python -m stegastamp.quantize_model \
    --input "$INPUT_MODEL" \
    --output "$OUTPUT_MODEL" \
    --height "$HEIGHT" \
    --width "$WIDTH" \
    --secret_size "$SECRET_SIZE"
