"""
DDM2 Evaluation Script for Mayo CT Dataset
评估 DDM2 模型的去噪效果，计算 MAE、SSIM 和 LPIPS

Usage:
    # 评估单个患者
    python eval_mayo.py --patient_id L143
    
    # 评估所有val患者
    python eval_mayo.py --batch val
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
EXCEL_PATH = '/host/d/file/new/mayo/Patient_lists/mayo_low_dose_CT_gaussian_simulation_v2.xlsx'
DATA_ROOT = '/host/d/file/new/mayo/'  # 用于替换 /host/d/Data/

# N2N 结果路径
N2N_ROOT = '/host/d/file/new/mayo/noise2noise data/gaussian_noise/pred_images_input_both'
N2N_EPOCH = 70

# DDM2 结果路径
DDM2_ROOT = 'experiments/mayo_ct_denoise/inference'

# GT 路径
GT_ROOT = '/host/d/file/new/mayo/original_imgs'

# 评估窗口
vmin = -160
vmax = 240



# ============================================================================
# 指标计算函数 
# ============================================================================
def calc_mae_with_ref_window(img, ref, vmin, vmax):
    """计算MAE，只在参考图像的[vmin, vmax]窗口内计算"""
    maes = []
    for slice_num in range(img.shape[-1]):
        slice_img = img[:,:,slice_num]
        slice_ref = ref[:,:,slice_num]
        mask = np.where((slice_ref >= vmin) & (slice_ref <= vmax), 1, 0)
        if np.sum(mask) == 0:
            continue
        mae = np.sum(np.abs(slice_img - slice_ref) * mask) / np.sum(mask)
        maes.append(mae)
    
    if len(maes) == 0:
        return np.nan, np.nan
    return np.mean(maes), np.std(maes)


def calc_ssim_with_ref_window(img, ref, vmin, vmax):
    """计算SSIM，只在参考图像的[vmin, vmax]窗口内计算"""
    ssims = []
    for slice_num in range(img.shape[-1]):
        slice_img = img[:,:,slice_num]
        slice_ref = ref[:,:,slice_num]
        mask = np.where((slice_ref >= vmin) & (slice_ref <= vmax), 1, 0)
        if np.sum(mask) == 0:
            continue
        _, ssim_map = structural_similarity(slice_img, slice_ref, data_range=vmax - vmin, full=True)
        ssim = np.sum(ssim_map * mask) / np.sum(mask)
        ssims.append(ssim)
    
    if len(ssims) == 0:
        return np.nan, np.nan
    return np.mean(ssims), np.std(ssims)


def calc_lpips(imgs1, imgs2, vmin, vmax, loss_fn):
    """计算LPIPS"""
    device = next(loss_fn.parameters()).device
    
    lpipss = []
    for slice_num in range(imgs1.shape[-1]):
        slice1 = imgs1[:,:,slice_num]
        slice2 = imgs2[:,:,slice_num]

        slice1 = np.clip(slice1, vmin, vmax).astype(np.float32)
        slice2 = np.clip(slice2, vmin, vmax).astype(np.float32)

        # 归一化到 [-1, 1]
        slice1 = (slice1 - vmin) / (vmax - vmin) * 2 - 1
        slice2 = (slice2 - vmin) / (vmax - vmin) * 2 - 1

        # 扩展为3通道
        slice1 = np.stack([slice1, slice1, slice1], axis=-1)
        slice2 = np.stack([slice2, slice2, slice2], axis=-1)

        slice1 = np.transpose(slice1, (2, 0, 1))[np.newaxis, ...]
        slice2 = np.transpose(slice2, (2, 0, 1))[np.newaxis, ...]

        slice1 = torch.from_numpy(slice1).to(device)
        slice2 = torch.from_numpy(slice2).to(device)

        lpips_val = loss_fn(slice1, slice2)
        lpipss.append(lpips_val.item())

    if len(lpipss) == 0:
        return np.nan, np.nan
    return np.mean(lpipss), np.std(lpipss)


def fix_path(path, data_root):
    """替换路径前缀"""
    if path and '/host/d/Data/' in path:
        return path.replace('/host/d/Data/', data_root)
    return path


# ============================================================================
# 主程序
# ============================================================================
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--excel', type=str, default=EXCEL_PATH, help='Excel file path')
    parser.add_argument('--batch', type=str, nargs='+', default=['val', 'test'], help='Batch list')
    parser.add_argument('--patient_id', type=str, default=None, help='Specific patient ID (e.g., L333)')
    parser.add_argument('--data_root', type=str, default=DATA_ROOT, help='Data root directory')
    parser.add_argument('--gt_root', type=str, default=GT_ROOT, help='GT root directory')
    parser.add_argument('--n2n_root', type=str, default=N2N_ROOT, help='N2N root directory')
    parser.add_argument('--n2n_epoch', type=int, default=N2N_EPOCH, help='N2N epoch')
    parser.add_argument('--ddm2_root', type=str, default=DDM2_ROOT, help='DDM2 root directory')
    parser.add_argument('--output', type=str, default='mayo_ddm2_results.xlsx', help='Output Excel file')
    args = parser.parse_args()
    
    # 加载 LPIPS 模型
    print("Loading LPIPS model...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loss_fn = lpips.LPIPS().to(device)
    print(f"LPIPS model loaded on {device}")
    
    # 读取患者列表
    df = pd.read_excel(args.excel)
    df = df[df['batch'].isin(args.batch)]
    
    # 如果指定了 patient_id，过滤
    if args.patient_id is not None:
        df = df[df['Patient_ID'] == args.patient_id]
    
    print(f"Batch {args.batch}: {len(df)} patients")
    print(f"Patients: {df['Patient_ID'].tolist()}")
    print(f"Evaluation window: [{vmin}, {vmax}]")
    
    results = []
    
    for i in range(len(df)):
        row = df.iloc[i]
        patient_id = row['Patient_ID']
        
        print(f"\n[{i+1}/{len(df)}] Patient: {patient_id}")
        
        # ========== 构建路径 ==========
        # GT (ground truth): {gt_root}/{patient_id}/img_sliced.nii.gz
        gt_file = os.path.join(args.gt_root, str(patient_id), 'img_sliced.nii.gz')
        if not os.path.exists(gt_file):
            print(f"  [SKIP] GT not found: {gt_file}")
            continue
        
        # Noisy (simulation input)
        noisy_file = fix_path(row['noise_file'], args.data_root)
        if not os.path.exists(noisy_file):
            print(f"  [SKIP] Noisy not found: {noisy_file}")
            continue
        
        # N2N result: {n2n_root}/{patient_id}/random_0/epoch{N}/pred_img.nii.gz
        n2n_file = os.path.join(args.n2n_root, str(patient_id), 'random_0', f'epoch{args.n2n_epoch}', 'pred_img.nii.gz')
        if not os.path.exists(n2n_file):
            print(f"  [SKIP] N2N not found: {n2n_file}")
            continue
        
        # DDM2 results: {ddm2_root}/{patient_id}/ddm2_*.nii.gz
        ddm2_first_file = os.path.join(args.ddm2_root, str(patient_id), 'ddm2_first_step.nii.gz')
        ddm2_final_file = os.path.join(args.ddm2_root, str(patient_id), 'ddm2_final.nii.gz')
        
        # ========== 加载数据 ==========
        print(f"  Loading GT: {gt_file}")
        gt_img = nb.load(gt_file).get_fdata().astype(np.float32)
        
        print(f"  Loading Noisy: {noisy_file}")
        noisy_img = nb.load(noisy_file).get_fdata().astype(np.float32)
        
        print(f"  Loading N2N: {n2n_file}")
        n2n_img = nb.load(n2n_file).get_fdata().astype(np.float32)
        
        HU_MIN_EVAL = -200
        HU_MAX_EVAL = 250
        gt_img = np.clip(gt_img, HU_MIN_EVAL, HU_MAX_EVAL)
        noisy_img = np.clip(noisy_img, HU_MIN_EVAL, HU_MAX_EVAL)
        n2n_img = np.clip(n2n_img, HU_MIN_EVAL, HU_MAX_EVAL)

        ddm2_first_img = None
        ddm2_final_img = None
        
        if os.path.exists(ddm2_first_file):
            print(f"  Loading DDM2 First: {ddm2_first_file}")
            ddm2_first_img = nb.load(ddm2_first_file).get_fdata().astype(np.float32)
        else:
            print(f"  [WARNING] DDM2 First not found: {ddm2_first_file}")
        
        if os.path.exists(ddm2_final_file):
            print(f"  Loading DDM2 Final: {ddm2_final_file}")
            ddm2_final_img = nb.load(ddm2_final_file).get_fdata().astype(np.float32)
        else:
            print(f"  [WARNING] DDM2 Final not found: {ddm2_final_file}")
        
        # 检查shape
        print(f"  Shapes - GT: {gt_img.shape}, Noisy: {noisy_img.shape}, N2N: {n2n_img.shape}")
        if ddm2_first_img is not None:
            print(f"           DDM2_First: {ddm2_first_img.shape}, DDM2_Final: {ddm2_final_img.shape if ddm2_final_img is not None else 'N/A'}")
        
        # 确保slice数量一致（取最小值）
        min_slices = min(gt_img.shape[-1], noisy_img.shape[-1], n2n_img.shape[-1])
        if ddm2_first_img is not None:
            min_slices = min(min_slices, ddm2_first_img.shape[-1])
        if ddm2_final_img is not None:
            min_slices = min(min_slices, ddm2_final_img.shape[-1])
        
        gt_img = gt_img[:,:,:min_slices]
        noisy_img = noisy_img[:,:,:min_slices]
        n2n_img = n2n_img[:,:,:min_slices]
        if ddm2_first_img is not None:
            ddm2_first_img = ddm2_first_img[:,:,:min_slices]
        if ddm2_final_img is not None:
            ddm2_final_img = ddm2_final_img[:,:,:min_slices]
        
        print(f"  Using {min_slices} slices for evaluation")
        
        # ========== 计算指标 ==========
        print("  Computing metrics...")
        
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
        print(f"  Results:")
        print(f"    Noisy:       MAE={mae_noisy:.4f}, SSIM={ssim_noisy:.4f}, LPIPS={lpips_noisy:.4f}")
        print(f"    N2N:         MAE={mae_n2n:.4f}, SSIM={ssim_n2n:.4f}, LPIPS={lpips_n2n:.4f}")
        if ddm2_first_img is not None:
            print(f"    DDM2_First:  MAE={mae_ddm2_first:.4f}, SSIM={ssim_ddm2_first:.4f}, LPIPS={lpips_ddm2_first:.4f}")
        if ddm2_final_img is not None:
            print(f"    DDM2_Final:  MAE={mae_ddm2_final:.4f}, SSIM={ssim_ddm2_final:.4f}, LPIPS={lpips_ddm2_final:.4f}")
        
        # 收集结果
        results.append({
            'patient_id': patient_id,
            'batch': row['batch'],
            'num_slices': min_slices,
            'mae_noisy': mae_noisy,
            'mae_n2n': mae_n2n,
            'mae_ddm2_first': mae_ddm2_first,
            'mae_ddm2_final': mae_ddm2_final,
            'ssim_noisy': ssim_noisy,
            'ssim_n2n': ssim_n2n,
            'ssim_ddm2_first': ssim_ddm2_first,
            'ssim_ddm2_final': ssim_ddm2_final,
            'lpips_noisy': lpips_noisy,
            'lpips_n2n': lpips_n2n,
            'lpips_ddm2_first': lpips_ddm2_first,
            'lpips_ddm2_final': lpips_ddm2_final,
        })
    
    # 保存结果
    if results:
        result_df = pd.DataFrame(results)
        
        # 打印汇总
        print("\n" + "=" * 70)
        print("Summary")
        print("=" * 70)
        print(f"{'Method':<15} {'MAE ↓':>12} {'SSIM ↑':>12} {'LPIPS ↓':>12}")
        print("-" * 70)
        print(f"{'Noisy':<15} {result_df['mae_noisy'].mean():>12.4f} {result_df['ssim_noisy'].mean():>12.4f} {result_df['lpips_noisy'].mean():>12.4f}")
        print(f"{'N2N':<15} {result_df['mae_n2n'].mean():>12.4f} {result_df['ssim_n2n'].mean():>12.4f} {result_df['lpips_n2n'].mean():>12.4f}")
        if not result_df['mae_ddm2_first'].isna().all():
            print(f"{'DDM2_First':<15} {result_df['mae_ddm2_first'].mean():>12.4f} {result_df['ssim_ddm2_first'].mean():>12.4f} {result_df['lpips_ddm2_first'].mean():>12.4f}")
        if not result_df['mae_ddm2_final'].isna().all():
            print(f"{'DDM2_Final':<15} {result_df['mae_ddm2_final'].mean():>12.4f} {result_df['ssim_ddm2_final'].mean():>12.4f} {result_df['lpips_ddm2_final'].mean():>12.4f}")
        print("=" * 70)
        
        # 计算相对改进
        if not result_df['mae_ddm2_final'].isna().all():
            mae_improve = (result_df['mae_n2n'].mean() - result_df['mae_ddm2_final'].mean()) / result_df['mae_n2n'].mean() * 100
            ssim_improve = (result_df['ssim_ddm2_final'].mean() - result_df['ssim_n2n'].mean()) / result_df['ssim_n2n'].mean() * 100
            lpips_improve = (result_df['lpips_n2n'].mean() - result_df['lpips_ddm2_final'].mean()) / result_df['lpips_n2n'].mean() * 100
            print(f"\nDDM2 vs N2N improvement:")
            print(f"  MAE:   {mae_improve:+.1f}% {'↓ better' if mae_improve > 0 else '↑ worse'}")
            print(f"  SSIM:  {ssim_improve:+.1f}% {'↑ better' if ssim_improve > 0 else '↓ worse'}")
            print(f"  LPIPS: {lpips_improve:+.1f}% {'↓ better' if lpips_improve > 0 else '↑ worse'}")
        
        # 保存 Excel
        result_df.to_excel(args.output, index=False)
        print(f"\nResults saved to: {args.output}")
    else:
        print("\nNo results to save!")
    
    print("\nDone!")
