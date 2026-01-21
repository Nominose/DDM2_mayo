"""
Mayo CT Dataset for DDM2

新增 patient_ids 参数：指定用哪些患者训练
例如: patient_ids=["L333"] 只用L333这一个case
"""

import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import nibabel as nib


class MayoCTDataset(Dataset):
    def __init__(
        self,
        dataroot,
        phase='train',
        image_size=512,
        in_channel=1,
        val_volume_idx=0,
        val_slice_idx=25,
        padding=3,
        lr_flip=0.5,
        stage2_file=None,
        train_batches=('train',),
        val_batches=('val',),
        slice_range=None,
        HU_MIN=-1000.0,
        HU_MAX=2000.0,
        data_root=None,
        teacher_n2n_root=None,
        teacher_n2n_epoch=70,
        num_slices=100,
        # ====== 新增：指定患者 ======
        patient_ids=None,  # e.g. ["L333"] 或 ["L333", "L096"]
        **kwargs
    ):
        self.phase = phase
        self.image_size = image_size
        self.padding = padding // 2
        self.lr_flip = lr_flip
        self.HU_MIN = HU_MIN
        self.HU_MAX = HU_MAX
        self.data_root = data_root
        self.teacher_n2n_root = teacher_n2n_root
        self.teacher_n2n_epoch = teacher_n2n_epoch
        self.num_slices = num_slices
        self.slice_range = slice_range
        
        # 统计npy使用情况
        self.npy_count = 0
        self.nii_count = 0

        # 读取Excel
        self.df = pd.read_excel(dataroot)
        
        # 按batch筛选
        target_batches = train_batches if phase == 'train' else val_batches
        self.df = self.df[self.df['batch'].isin(target_batches)].reset_index(drop=True)
        
        # ====== 按patient_ids筛选 ======
        if patient_ids is not None:
            if isinstance(patient_ids, str):
                patient_ids = [patient_ids]
            self.df = self.df[self.df['Patient_ID'].isin(patient_ids)].reset_index(drop=True)
            print(f'[{phase}] Filtering by patient_ids: {patient_ids}')
        
        # 构建数据列表
        self.data_list = []
        for _, row in self.df.iterrows():
            self.data_list.append({
                'noise_file': row['noise_file'],
                'patient_id': row['Patient_ID'],
            })
        
        V = len(self.data_list)
        self.data_shape = (512, 512)
        
        # 验证集设置
        if val_volume_idx == 'all':
            self.val_volume_idx = list(range(V))
        elif isinstance(val_volume_idx, int):
            self.val_volume_idx = [val_volume_idx]
        else:
            self.val_volume_idx = list(val_volume_idx)
        self.val_volume_idx = [x for x in self.val_volume_idx if x < V]
        self.val_slice_idx = val_slice_idx

        # 构建样本索引
        self.samples = self._build_sample_indices()
        self.matched_state = self._parse_stage2_file(stage2_file) if stage2_file else None

        # 兼容性
        class FakeRawData:
            def __init__(self, shape):
                self.shape = shape
        self.raw_data = FakeRawData((512, 512, num_slices * V, V))

        # transforms
        if phase == 'train':
            self.transforms = transforms.Compose([
                transforms.ToTensor(),
                transforms.RandomVerticalFlip(lr_flip),
                transforms.RandomHorizontalFlip(lr_flip),
                transforms.Lambda(lambda t: (t * 2) - 1),
            ])
        else:
            self.transforms = transforms.Compose([
                transforms.ToTensor(),
                transforms.Lambda(lambda t: (t * 2) - 1),
            ])

        # 打印信息
        patient_list = [d['patient_id'] for d in self.data_list]
        print(f'[{phase}] MayoCTDataset: {V} patients {patient_list}, {len(self.samples)} samples, {num_slices} slices/patient')
        print(f'[{phase}] data_root: {data_root}')
        print(f'[{phase}] teacher_n2n_root: {teacher_n2n_root}')
        
        # 检测npy情况
        if self.data_list:
            self._check_npy_availability()

    def _fix_path(self, path):
        """替换路径前缀"""
        if self.data_root and path:
            if '/host/d/Data/' in path:
                return path.replace('/host/d/Data/', self.data_root)
        return path

    def _check_npy_availability(self):
        """检测npy文件可用性"""
        sample_noise = self._fix_path(self.data_list[0]['noise_file'])
        npy_noise = sample_noise.replace('.nii.gz', '.npy') if sample_noise else None
        if npy_noise and os.path.exists(npy_noise):
            print(f'[{self.phase}] ✓ Simulation NPY available: {npy_noise}')
        elif sample_noise and os.path.exists(sample_noise):
            print(f'[{self.phase}] ✗ Simulation using NII.GZ: {sample_noise}')
        else:
            print(f'[{self.phase}] ✗ Simulation file NOT FOUND: {sample_noise}')
        
        if self.teacher_n2n_root:
            sample_teacher = self._get_teacher_path(self.data_list[0]['patient_id'])
            npy_teacher = sample_teacher.replace('.nii.gz', '.npy') if sample_teacher else None
            if npy_teacher and os.path.exists(npy_teacher):
                print(f'[{self.phase}] ✓ Teacher N2N NPY available: {npy_teacher}')
            elif sample_teacher and os.path.exists(sample_teacher):
                print(f'[{self.phase}] ✗ Teacher N2N using NII.GZ: {sample_teacher}')
            else:
                print(f'[{self.phase}] ✗ Teacher N2N file NOT FOUND: {sample_teacher}')

    def _get_teacher_path(self, patient_id):
        """N2N结果路径: {root}/{patient_id}/random_0/epoch{N}/pred_img.nii.gz"""
        if not self.teacher_n2n_root:
            return None
        return os.path.join(self.teacher_n2n_root, str(patient_id),
                          "random_0", f"epoch{self.teacher_n2n_epoch}", "pred_img.nii.gz")

    def _build_sample_indices(self):
        samples = []
        start = self.slice_range[0] if self.slice_range else 0
        end = min(self.num_slices, self.slice_range[1]) if self.slice_range else self.num_slices
        
        if self.phase in ('train', 'test'):
            for vol_idx in range(len(self.data_list)):
                for s in range(start, end):
                    samples.append((vol_idx, s))
        else:  # val
            for vol_idx in self.val_volume_idx:
                if vol_idx >= len(self.data_list):
                    continue
                if self.val_slice_idx == 'all':
                    slices = range(start, end)
                elif isinstance(self.val_slice_idx, int):
                    slices = [self.val_slice_idx] if start <= self.val_slice_idx < end else []
                else:
                    slices = [s for s in self.val_slice_idx if start <= s < end]
                for s in slices:
                    samples.append((vol_idx, s))
        return samples

    def _parse_stage2_file(self, path):
        if not path or not os.path.exists(path):
            return None
        results = {}
        with open(path, 'r') as f:
            for line in f:
                info = line.strip().split('_')
                if len(info) >= 3:
                    v, s, t = int(info[0]), int(info[1]), int(info[2])
                    results.setdefault(v, {})[s] = t
        return results

    def _preprocess(self, img):
        img = np.clip(img.astype(np.float32), self.HU_MIN, self.HU_MAX)
        return (img - self.HU_MIN) / (self.HU_MAX - self.HU_MIN)

    def _load_slice(self, path, idx):
        if not path:
            return np.zeros(self.data_shape, dtype=np.float32)
        
        path = self._fix_path(path)
        npy_path = path.replace('.nii.gz', '.npy')
        
        if os.path.exists(npy_path):
            vol = np.load(npy_path, mmap_mode='r')
            idx = min(idx, vol.shape[2] - 1)
            img = np.array(vol[:, :, idx])
            self.npy_count += 1
        elif os.path.exists(path):
            nii = nib.load(path)
            idx = min(idx, nii.shape[2] - 1)
            img = np.asarray(nii.dataobj[:, :, idx])
            self.nii_count += 1
        else:
            print(f'[WARNING] File not found: {path}')
            return np.zeros(self.data_shape, dtype=np.float32)
        
        return self._preprocess(np.nan_to_num(img.astype(np.float32), 0))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        vol_idx, slice_idx = self.samples[idx]
        data = self.data_list[vol_idx]
        patient_id = data['patient_id']

        noisy = self._load_slice(data['noise_file'], slice_idx)

        teacher_path = self._get_teacher_path(patient_id)
        has_teacher = teacher_path and os.path.exists(self._fix_path(teacher_path))
        teacher = self._load_slice(teacher_path, slice_idx) if has_teacher else None

        cond_ch = 2 * self.padding if self.padding > 0 else 1
        channels = [noisy] * cond_ch + [noisy]
        if teacher is not None:
            channels.append(teacher)

        raw = self.transforms(np.stack(channels, axis=-1).astype(np.float32))

        if teacher is not None:
            denoised = raw[[-1], :, :]
            raw = raw[:-1, :, :]
        
        ret = {
            'X': raw[[-1], :, :].float(),
            'condition': raw[:-1, :, :].float(),
            'matched_state': torch.tensor([500.0])
        }

        if self.matched_state and vol_idx in self.matched_state and slice_idx in self.matched_state[vol_idx]:
            ret['matched_state'] = torch.tensor([float(self.matched_state[vol_idx][slice_idx])])

        if self.teacher_n2n_root:
            ret['denoised'] = denoised.float() if teacher is not None else ret['X'].clone()
            if teacher is None:
                ret['matched_state'] = torch.tensor([1.0])

        return ret
    
    def print_loading_stats(self):
        """打印加载统计"""
        total = self.npy_count + self.nii_count
        if total > 0:
            print(f'[{self.phase}] Loading stats: NPY={self.npy_count} ({100*self.npy_count/total:.1f}%), NII.GZ={self.nii_count} ({100*self.nii_count/total:.1f}%)')
