"""
将 nii.gz 文件转换为 npy 文件
用法:
    python nii2npy.py /path/to/file.nii.gz              # 单文件
    python nii2npy.py /path/to/folder                   # 整个文件夹
    python nii2npy.py /path/to/folder --recursive      # 递归处理子文件夹
"""

import os
import sys
import argparse
import numpy as np
import nibabel as nib
from pathlib import Path


def convert_nii_to_npy(nii_path, output_path=None):
    """转换单个 nii.gz 文件"""
    if output_path is None:
        output_path = str(nii_path).replace('.nii.gz', '.npy').replace('.nii', '.npy')
    
    try:
        data = nib.load(nii_path).get_fdata().astype(np.float32)
        np.save(output_path, data)
        return True, output_path
    except Exception as e:
        return False, str(e)


def process_folder(folder_path, recursive=False):
    """处理文件夹中的所有 nii.gz 文件"""
    folder = Path(folder_path)
    
    if recursive:
        nii_files = list(folder.rglob('*.nii.gz')) + list(folder.rglob('*.nii'))
    else:
        nii_files = list(folder.glob('*.nii.gz')) + list(folder.glob('*.nii'))
    
    print(f"找到 {len(nii_files)} 个文件")
    
    success, failed = 0, 0
    for i, nii_file in enumerate(nii_files):
        # 跳过已转换的
        npy_path = str(nii_file).replace('.nii.gz', '.npy').replace('.nii', '.npy')
        if os.path.exists(npy_path):
            print(f"[{i+1}/{len(nii_files)}] 跳过 (已存在): {nii_file.name}")
            continue
        
        ok, result = convert_nii_to_npy(nii_file)
        if ok:
            print(f"[{i+1}/{len(nii_files)}] ✓ {nii_file.name}")
            success += 1
        else:
            print(f"[{i+1}/{len(nii_files)}] ✗ {nii_file.name}: {result}")
            failed += 1
    
    print(f"\n完成: {success} 成功, {failed} 失败")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='将 nii.gz 转换为 npy')
    parser.add_argument('path', help='文件或文件夹路径')
    parser.add_argument('-r', '--recursive', action='store_true', help='递归处理子文件夹')
    parser.add_argument('-o', '--output', help='输出路径 (仅单文件时有效)')
    
    args = parser.parse_args()
    
    path = Path(args.path)
    
    if path.is_file():
        ok, result = convert_nii_to_npy(path, args.output)
        if ok:
            print(f"✓ 已保存: {result}")
        else:
            print(f"✗ 失败: {result}")
    elif path.is_dir():
        process_folder(path, args.recursive)
    else:
        print(f"路径不存在: {args.path}")
        sys.exit(1)
