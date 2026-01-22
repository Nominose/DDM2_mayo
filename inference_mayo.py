"""
DDM2 Inference Script for Mayo CT Dataset
生成完整的 100 slice nii.gz 文件

Usage:
    python inference_mayo.py -c config/mayo_ct_denoise.json --patient_id L291
    python inference_mayo.py -c config/mayo_ct_denoise.json --patient_idx 0
"""

import argparse
import os
import sys
import json
import numpy as np
import torch
import nibabel as nib
from tqdm import tqdm

sys.path.insert(0, '.')

import data as Data
import model as Model
import core.logger as Logger


def parse_args():
    parser = argparse.ArgumentParser(description='DDM2 Inference for Mayo CT')
    parser.add_argument('-c', '--config', type=str, required=True, help='Config file path')
    parser.add_argument('--checkpoint', type=str, default=None, help='Checkpoint path')
    parser.add_argument('--patient_id', type=str, default=None, help='Patient ID (e.g., L333)')
    parser.add_argument('--patient_idx', type=int, default=0, help='Patient index (if patient_id not specified)')
    parser.add_argument('--output_dir', type=str, default=None, help='Output directory')
    parser.add_argument('--save_first', action='store_true', default=True, help='Save first-step')
    parser.add_argument('--save_final', action='store_true', default=True, help='Save final')
    parser.add_argument('--gpu', type=int, default=0, help='GPU id')
    return parser.parse_args()


def find_latest_checkpoint(experiments_dir='experiments'):
    """自动查找最新的 checkpoint"""
    latest_dir = None
    latest_time = 0
    
    for d in os.listdir(experiments_dir):
        if 'mayo' in d.lower():
            ckpt_path = os.path.join(experiments_dir, d, 'checkpoint', 'latest_gen.pth')
            if os.path.exists(ckpt_path):
                mtime = os.path.getmtime(ckpt_path)
                if mtime > latest_time:
                    latest_time = mtime
                    latest_dir = os.path.join(experiments_dir, d, 'checkpoint', 'latest')
    
    return latest_dir


def main():
    args = parse_args()
    
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 加载 config
    with open(args.config, 'r') as f:
        opt = json.load(f)
    
    HU_MIN = opt['datasets']['val'].get('HU_MIN', -1000.0)
    HU_MAX = opt['datasets']['val'].get('HU_MAX', 2000.0)
    
    print("=" * 60)
    print("DDM2 Inference - Mayo CT")
    print("=" * 60)
    print(f"Config: {args.config}")
    print(f"HU range: [{HU_MIN}, {HU_MAX}]")
    
    # 查找 checkpoint
    checkpoint = args.checkpoint
    if checkpoint is None:
        checkpoint = find_latest_checkpoint()
    
    if checkpoint is None:
        print("[ERROR] No checkpoint found!")
        return
    
    print(f"Checkpoint: {checkpoint}")
    
    # 设置输出目录
    if args.output_dir is None:
        args.output_dir = os.path.dirname(checkpoint).replace('/checkpoint', '/inference')
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Output dir: {args.output_dir}")
    
    # 创建数据集
    print("\n[1/3] Loading dataset...")
    val_opt = opt['datasets']['val'].copy()
    
    # 如果指定了patient_id，覆盖配置
    if args.patient_id:
        val_opt['patient_ids'] = [args.patient_id]
        patient_id = args.patient_id
    else:
        # 用patient_idx，需要先创建一个临时数据集获取patient_id
        import pandas as pd
        df = pd.read_excel(val_opt['dataroot'])
        # 按batch筛选
        val_batches = val_opt.get('val_batches', ['val'])
        df = df[df['batch'].isin(val_batches)].reset_index(drop=True)
        if args.patient_idx >= len(df):
            print(f"[ERROR] patient_idx {args.patient_idx} out of range (max: {len(df)-1})")
            return
        patient_id = df.iloc[args.patient_idx]['Patient_ID']
        val_opt['patient_ids'] = [patient_id]
    
    print(f"Patient ID: {patient_id}")
    
    val_opt['val_volume_idx'] = 0  # 因为只选了一个patient
    val_opt['val_slice_idx'] = 'all'
    
    val_set = Data.create_dataset(val_opt, 'val', stage2_file=opt.get('stage2_file'))
    
    num_slices = len(val_set)
    print(f"Total slices: {num_slices}")
    
    # 加载模型
    print("\n[2/3] Loading model...")
    opt_model = Logger.dict_to_nonedict(opt)
    opt_model['path']['resume_state'] = checkpoint
    
    diffusion = Model.create_model(opt_model)
    diffusion.set_new_noise_schedule(opt_model['model']['beta_schedule']['val'], schedule_phase='val')
    print("Model loaded!")
    
    # 推理
    print("\n[3/3] Running inference...")
    
    first_results = []
    final_results = []
    noisy_inputs = []
    
    for idx in tqdm(range(num_slices), desc="Inference"):
        sample = val_set[idx]
        
        batch = {k: v.unsqueeze(0) if isinstance(v, torch.Tensor) else v 
                 for k, v in sample.items()}
        
        diffusion.feed_data(batch)
        diffusion.test(continous=True)
        visuals = diffusion.get_current_visuals()
        
        all_imgs = visuals['denoised'].numpy()
        
        # 从 [-1, 1] 转换到 [0, 1]
        noisy = (all_imgs[0].squeeze() + 1) / 2
        first = (all_imgs[1].squeeze() + 1) / 2
        final = (all_imgs[-1].squeeze() + 1) / 2
        
        # 转换到 HU 值
        noisy_hu = noisy * (HU_MAX - HU_MIN) + HU_MIN
        first_hu = first * (HU_MAX - HU_MIN) + HU_MIN
        final_hu = final * (HU_MAX - HU_MIN) + HU_MIN
        
        noisy_inputs.append(noisy_hu)
        first_results.append(first_hu)
        final_results.append(final_hu)
    
    # 堆叠成 3D volume
    noisy_volume = np.stack(noisy_inputs, axis=-1).astype(np.float32)
    first_volume = np.stack(first_results, axis=-1).astype(np.float32)
    final_volume = np.stack(final_results, axis=-1).astype(np.float32)
    
    print(f"\nVolume shape: {first_volume.shape}")
    
    # 获取 affine（尝试从原始文件获取）
    affine = np.eye(4)
    if hasattr(val_set, 'data_list') and len(val_set.data_list) > 0:
        noise_path = val_set.data_list[0].get('noise_file')
        if noise_path and hasattr(val_set, '_fix_path'):
            noise_path = val_set._fix_path(noise_path)
        if noise_path and os.path.exists(noise_path):
            try:
                orig_nii = nib.load(noise_path)
                affine = orig_nii.affine
                print(f"Loaded affine from: {noise_path}")
            except:
                pass
    
    # 保存文件
    output_subdir = os.path.join(args.output_dir, str(patient_id))
    os.makedirs(output_subdir, exist_ok=True)
    
    # 保存 noisy input
    noisy_nii = nib.Nifti1Image(noisy_volume, affine)
    noisy_path = os.path.join(output_subdir, 'noisy_input.nii.gz')
    nib.save(noisy_nii, noisy_path)
    print(f"Saved: {noisy_path}")
    
    # 保存 first-step
    if args.save_first:
        first_nii = nib.Nifti1Image(first_volume, affine)
        first_path = os.path.join(output_subdir, 'ddm2_first_step.nii.gz')
        nib.save(first_nii, first_path)
        print(f"Saved: {first_path}")
    
    # 保存 final
    if args.save_final:
        final_nii = nib.Nifti1Image(final_volume, affine)
        final_path = os.path.join(output_subdir, 'ddm2_final.nii.gz')
        nib.save(final_nii, final_path)
        print(f"Saved: {final_path}")
    
    # 统计信息
    print("\n" + "=" * 60)
    print("Statistics (HU values)")
    print("-" * 60)
    print(f"{'Image':<20} {'Min':>10} {'Max':>10} {'Mean':>10} {'Std':>10}")
    print("-" * 60)
    print(f"{'Noisy Input':<20} {noisy_volume.min():>10.1f} {noisy_volume.max():>10.1f} {noisy_volume.mean():>10.1f} {noisy_volume.std():>10.1f}")
    print(f"{'DDM2 First-step':<20} {first_volume.min():>10.1f} {first_volume.max():>10.1f} {first_volume.mean():>10.1f} {first_volume.std():>10.1f}")
    print(f"{'DDM2 Final':<20} {final_volume.min():>10.1f} {final_volume.max():>10.1f} {final_volume.mean():>10.1f} {final_volume.std():>10.1f}")
    print("=" * 60)
    
    print("\nDone!")


if __name__ == '__main__':
    main()
