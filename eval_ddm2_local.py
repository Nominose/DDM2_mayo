"""
DDM2 Evaluation Script - Local Version
本脚本用于在本地环境下评估 DDM2 模型的去噪效果。
它会计算 MAE、SSIM 和 LPIPS 三个指标，评估 DDM2 的去噪结果与 Noisy 图像和 N2N 结果的差异。

Usage:
    # 评估单个患者
    python eval_ddm2_local.py --patient_id 10461
"""

import nibabel as nb
import os
import lpips
import torch
import numpy as np
import pandas as pd
from skimage.metrics import structural_similarity
import argparse

# ============================================================================
# 路径配置 - 修改为你的本地路径
# ============================================================================
EXCEL_PATH = '/host/d/file/xingyi_datasets.xlsx'
BATCH_LIST = [5]

# 数据路径
GT_ROOT = '/host/d/file/gt/diffusion denoising/unsupervised_gaussian_2D_current_beta0/pred_images'
GT_EPOCH = 61
NOISY_ROOT = '/host/d/file/pre/noise2noise/pred_images'  # condition_img 所在目录
N2N_ROOT = '/host/d/file/pre/noise2noise/pred_images'
DDM2_ROOT = '/host/c/Users/ROG/Documents/GitHub/DDM/experiments/ct_denoise_260121_021441/inference'

N2N_EPOCH = 78

# 评估窗口
vmin = 0
vmax = 100

# ============================================================================
# 指标计算函数 
# ============================================================================
def calc_mae_with_ref_window(img, ref, vmin, vmax):
    maes = []
    for slice_num in range(0, img.shape[-1]):
        slice_img = img[:,:,slice_num]
        slice_ref = ref[:,:,slice_num]
        mask = np.where((slice_ref >= vmin) & (slice_ref <= vmax), 1, 0)
        if np.sum(mask) == 0:
            continue
        mae = np.sum(np.abs(slice_img - slice_ref) * mask) / np.sum(mask)
        maes.append(mae)

    return np.mean(maes), np.std(maes)


def calc_ssim_with_ref_window(img, ref, vmin, vmax):
    ssims = []
    for slice_num in range(0, img.shape[-1]):
        slice_img = img[:,:,slice_num]
        slice_ref = ref[:,:,slice_num]
        mask = np.where((slice_ref >= vmin) & (slice_ref <= vmax), 1, 0)
        if np.sum(mask) == 0:
            continue
        _, ssim_map = structural_similarity(slice_img, slice_ref, data_range=vmax - vmin, full=True)
        ssim = np.sum(ssim_map * mask) / np.sum(mask)
        ssims.append(ssim)

    return np.mean(ssims), np.std(ssims)


def calc_lpips(imgs1, imgs2, vmin, vmax, loss_fn):
    """注意：loss_fn 从外部传入，避免重复加载"""
    device = next(loss_fn.parameters()).device
    
    lpipss = []
    for slice_num in range(0, imgs1.shape[-1]):
        slice1 = imgs1[:,:,slice_num]
        slice2 = imgs2[:,:,slice_num]

        slice1 = np.clip(slice1, vmin, vmax).astype(np.float32)
        slice2 = np.clip(slice2, vmin, vmax).astype(np.float32)

        slice1 = (slice1 - vmin) / (vmax - vmin) * 2 - 1
        slice2 = (slice2 - vmin) / (vmax - vmin) * 2 - 1

        slice1 = np.stack([slice1, slice1, slice1], axis=-1)
        slice2 = np.stack([slice2, slice2, slice2], axis=-1)

        slice1 = np.transpose(slice1, (2, 0, 1))[np.newaxis, ...]
        slice2 = np.transpose(slice2, (2, 0, 1))[np.newaxis, ...]

        slice1 = torch.from_numpy(slice1).to(device)
        slice2 = torch.from_numpy(slice2).to(device)

        lpips_val = loss_fn(slice1, slice2)
        lpipss.append(lpips_val.item())

    return np.mean(lpipss), np.std(lpipss)


def get_gt_path(gt_root, pid_str, psid_str, random_n, gt_epoch):
    """根据患者 ID 构建 GT 路径"""
    # 格式: GT_ROOT/pid_str/psid_str/random_0/epoch61_1/gt_img.nii.gz
    return os.path.join(
        gt_root, 
        pid_str, 
        psid_str, 
        f'random_{random_n}', 
        f'epoch{gt_epoch}_1', 
        'gt_img.nii.gz'
    )


# ============================================================================
# 主程序
# ============================================================================
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--excel', type=str, default=EXCEL_PATH, help='Excel file path')
    parser.add_argument('--batch', type=int, nargs='+', default=BATCH_LIST, help='Batch list')
    parser.add_argument('--patient_id', type=int, default=None, help='Specific patient ID')
    parser.add_argument('--gt_root', type=str, default=GT_ROOT, help='GT root directory')
    parser.add_argument('--gt_epoch', type=int, default=GT_EPOCH, help='GT epoch number')
    parser.add_argument('--noisy_root', type=str, default=NOISY_ROOT, help='Noisy root directory')
    parser.add_argument('--n2n_root', type=str, default=N2N_ROOT, help='N2N root directory')
    parser.add_argument('--ddm2_root', type=str, default=DDM2_ROOT, help='DDM2 root directory')
    parser.add_argument('--output', type=str, default='/host/d/file/ddm2_results.xlsx', help='Output Excel file')
    args = parser.parse_args()
    
    # 加载 LPIPS 模型（只加载一次）
    print("Loading LPIPS model...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loss_fn = lpips.LPIPS().to(device)
    print("LPIPS model loaded.")
    
    # 读取患者列表
    df = pd.read_excel(args.excel)
    df = df[df['batch'].isin(args.batch)]
    df = df[df['random_num'] == 0]  # 只用 random_0
    
    # 如果指定了 patient_id，过滤
    if args.patient_id is not None:
        df = df[df['Patient_ID'] == args.patient_id]
    
    print(f"Batch {args.batch}: {len(df)} patients")
    
    results = []
    
    for i in range(len(df)):
        row = df.iloc[i]
        patient_id = row['Patient_ID']
        patient_subid = row['Patient_subID']
        random_n = row['random_num']
        
        # 格式化路径 (8位/10位补零，用于N2N/GT等)
        pid_str = f"{int(patient_id):08d}"
        psid_str = f"{int(patient_subid):010d}"
        # DDM2 路径不补零
        pid_raw = str(int(patient_id))
        psid_raw = str(int(patient_subid))
        
        print(f"\n[{i+1}/{len(df)}] {pid_str} / {psid_str} / random_{random_n}")
        
        # ========== 加载数据 ==========
        # GT (ground truth) - 根据患者 ID 查找
        gt_file = get_gt_path(args.gt_root, pid_str, psid_str, random_n, args.gt_epoch)
        if not os.path.exists(gt_file):
            print(f"  [SKIP] GT not found: {gt_file}")
            continue
        gt_img = nb.load(gt_file).get_fdata()
        
        # Noisy (condition) image
        noisy_file = os.path.join(args.noisy_root, pid_str, psid_str, f'random_{random_n}', f'epoch{N2N_EPOCH}', 'condition_img.nii.gz')
        if not os.path.exists(noisy_file):
            print(f"  [SKIP] Noisy not found: {noisy_file}")
            continue
        noisy_img = nb.load(noisy_file).get_fdata()
        
        # N2N result
        n2n_file = os.path.join(args.n2n_root, pid_str, psid_str, f'random_{random_n}', f'epoch{N2N_EPOCH}', 'pred_img.nii.gz')
        if not os.path.exists(n2n_file):
            print(f"  [SKIP] N2N not found: {n2n_file}")
            continue
        n2n_img = nb.load(n2n_file).get_fdata()
        
        # DDM2 First Step (不补零)
        ddm2_first_file = os.path.join(args.ddm2_root, pid_raw, psid_raw, 'ddm2_first_step.nii.gz')
        ddm2_first_img = None
        if os.path.exists(ddm2_first_file):
            ddm2_first_img = nb.load(ddm2_first_file).get_fdata()
        else:
            print(f"  [WARNING] DDM2 First not found: {ddm2_first_file}")
        
        # DDM2 Final (不补零)
        ddm2_final_file = os.path.join(args.ddm2_root, pid_raw, psid_raw, 'ddm2_final.nii.gz')
        ddm2_final_img = None
        if os.path.exists(ddm2_final_file):
            ddm2_final_img = nb.load(ddm2_final_file).get_fdata()
        else:
            print(f"  [WARNING] DDM2 Final not found: {ddm2_final_file}")
        
        # ========== 计算指标 ==========
        # MAE
        mae_noisy, _ = calc_mae_with_ref_window(noisy_img, gt_img, vmin, vmax)
        mae_n2n, _ = calc_mae_with_ref_window(n2n_img, gt_img, vmin, vmax)
        mae_ddm2_first = np.nan
        mae_ddm2_final = np.nan
        if ddm2_first_img is not None:
            mae_ddm2_first, _ = calc_mae_with_ref_window(ddm2_first_img, gt_img, vmin, vmax)
        if ddm2_final_img is not None:
            mae_ddm2_final, _ = calc_mae_with_ref_window(ddm2_final_img, gt_img, vmin, vmax)
        
        # SSIM
        ssim_noisy, _ = calc_ssim_with_ref_window(noisy_img, gt_img, vmin, vmax)
        ssim_n2n, _ = calc_ssim_with_ref_window(n2n_img, gt_img, vmin, vmax)
        ssim_ddm2_first = np.nan
        ssim_ddm2_final = np.nan
        if ddm2_first_img is not None:
            ssim_ddm2_first, _ = calc_ssim_with_ref_window(ddm2_first_img, gt_img, vmin, vmax)
        if ddm2_final_img is not None:
            ssim_ddm2_final, _ = calc_ssim_with_ref_window(ddm2_final_img, gt_img, vmin, vmax)
        
        # LPIPS
        lpips_noisy, _ = calc_lpips(noisy_img, gt_img, vmin, vmax, loss_fn)
        lpips_n2n, _ = calc_lpips(n2n_img, gt_img, vmin, vmax, loss_fn)
        lpips_ddm2_first = np.nan
        lpips_ddm2_final = np.nan
        if ddm2_first_img is not None:
            lpips_ddm2_first, _ = calc_lpips(ddm2_first_img, gt_img, vmin, vmax, loss_fn)
        if ddm2_final_img is not None:
            lpips_ddm2_final, _ = calc_lpips(ddm2_final_img, gt_img, vmin, vmax, loss_fn)
        
        # 打印结果
        print(f"  Noisy:       MAE={mae_noisy:.4f}, SSIM={ssim_noisy:.4f}, LPIPS={lpips_noisy:.4f}")
        print(f"  N2N:         MAE={mae_n2n:.4f}, SSIM={ssim_n2n:.4f}, LPIPS={lpips_n2n:.4f}")
        if ddm2_first_img is not None:
            print(f"  DDM2_First:  MAE={mae_ddm2_first:.4f}, SSIM={ssim_ddm2_first:.4f}, LPIPS={lpips_ddm2_first:.4f}")
        if ddm2_final_img is not None:
            print(f"  DDM2_Final:  MAE={mae_ddm2_final:.4f}, SSIM={ssim_ddm2_final:.4f}, LPIPS={lpips_ddm2_final:.4f}")
        
        # 收集结果
        results.append([
            patient_id, patient_subid, random_n,
            mae_noisy, mae_n2n, mae_ddm2_first, mae_ddm2_final,
            ssim_noisy, ssim_n2n, ssim_ddm2_first, ssim_ddm2_final,
            lpips_noisy, lpips_n2n, lpips_ddm2_first, lpips_ddm2_final,
        ])
    
    # 保存结果
    if results:
        columns = [
            'patient_id', 'patient_subid', 'random_n',
            'mae_noisy', 'mae_n2n', 'mae_ddm2_first', 'mae_ddm2_final',
            'ssim_noisy', 'ssim_n2n', 'ssim_ddm2_first', 'ssim_ddm2_final',
            'lpips_noisy', 'lpips_n2n', 'lpips_ddm2_first', 'lpips_ddm2_final',
        ]
        result_df = pd.DataFrame(results, columns=columns)
        
        # 打印汇总
        print("\n" + "=" * 60)
        print("Summary")
        print("=" * 60)
        print(f"{'Method':<15} {'MAE ↓':>10} {'SSIM ↑':>10} {'LPIPS ↓':>10}")
        print("-" * 60)
        print(f"{'Noisy':<15} {result_df['mae_noisy'].mean():>10.4f} {result_df['ssim_noisy'].mean():>10.4f} {result_df['lpips_noisy'].mean():>10.4f}")
        print(f"{'N2N':<15} {result_df['mae_n2n'].mean():>10.4f} {result_df['ssim_n2n'].mean():>10.4f} {result_df['lpips_n2n'].mean():>10.4f}")
        print(f"{'DDM2_First':<15} {result_df['mae_ddm2_first'].mean():>10.4f} {result_df['ssim_ddm2_first'].mean():>10.4f} {result_df['lpips_ddm2_first'].mean():>10.4f}")
        print(f"{'DDM2_Final':<15} {result_df['mae_ddm2_final'].mean():>10.4f} {result_df['ssim_ddm2_final'].mean():>10.4f} {result_df['lpips_ddm2_final'].mean():>10.4f}")
        print("=" * 60)
        
        # 保存 Excel
        result_df.to_excel(args.output, index=False)
        print(f"\nResults saved to: {args.output}")
    
    print("\nDone!")
