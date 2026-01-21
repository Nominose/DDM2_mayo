# DDM<sup>2</sup>: Self-Supervised Diffusion MRI Denoising with Generative Diffusion Models, ICLR 2023

1.修改config文件路径为本地路径
2.(可选但是加速很多)运行nii2npy.py文件处理噪声图像和n2n结果得到npy格式图像，加速训练和推理
3.终端: ./run_stage2.sh 得到stage2_matched.txt文件
4.终端：./run_stage3.sh  运行训练
5.终端：python inference_ddm2.py -c config/ct_denoise.json --patient_idx 0 得到推理结果，文件路径在experiment/inference/
6.终端：python eval_ddm2_local.py --patient_id 214878 
修改最后病人id为需要量化结果的id，得到量化结果，该步需要修改eval_ddm2_local.py内文件路径为本地路径


config:

"resume_state"：
{
  "name": "ct_denoise",
  "phase": "train",
  "gpu_ids": [
    0
  ],
  "path": {
    "log": "logs",
    "tb_logger": "tb_logger",
    "results": "results",
    "checkpoint": "checkpoint",
    "resume_state": null //断点重训，路径如"/host/xxx/experiments/ct_denoise_xxx/checkpoint/latest"
  },
  "datasets": {
    "train": {
      "name": "ct",
      "dataroot": "/host/d/file/fixedCT_static_simulation_train_test_gaussian_local.xlsx", //输入噪声的dataroot

      "data_root": "/host/d/file/simulation/", //噪声数据的实际存放位置
      "train_batches": [5], //训练集batch
      "val_batches": [5], //验证集batch
      "volume_mask": [14, 15], //训练用当前batch下的第几个case，从0开始当前为0，1，2...第14张
      "slice_range": [30, 80],
      "phase": "train",
      "padding": 3,
      "val_volume_idx": "all", //选验证集当前batch下的第几个的case验证
      "val_slice_idx": 25, //第几张slice
      "batch_size": 1,
      "in_channel": 1,
      "num_workers": 4,
      "use_shuffle": true,
      "image_size": 512,
      "lr_flip": 0.5,
      "HU_MIN": -1000.0, //HU设置
      "HU_MAX": 2000.0, 
      "histogram_equalization": true, //是否开启he
      "bins_file": "/host/d/file/histogram_equalization/bins.npy", //替换为本地路径
      "bins_mapped_file": "/host/d/file/histogram_equalization/bins_mapped.npy", //替换为本地路径
      "teacher_n2n_root": "/host/d/file/pre/noise2noise/pred_images/", //n2n结果路径
      "teacher_n2n_epoch": 78 //n2nepoch
    },
    "val": {
      "name": "ct",
      "dataroot": "/host/d/file/fixedCT_static_simulation_train_test_gaussian_local.xlsx", //输入噪声的dataroot

      "data_root": "/host/d/file/simulation/", //噪声数据的实际存放位置
      "train_batches": [5], //训练集batch
      "val_batches": [5], //验证集batch
      "volume_mask": [14, 15], //训练用当前batch下的第几个case，从0开始当前为0，1，2...第14张
      "slice_range": [30, 80],
      "phase": "val",
      "padding": 3,
      "val_volume_idx": "all", //选验证集当前batch下的第几个的case验证
      "val_slice_idx": 25,  //第几张slice
      "batch_size": 1,
      "in_channel": 1,
      "num_workers": 0,
      "image_size": 512,
      "lr_flip": 0.0,
      "HU_MIN": -1000.0, //HU设置
      "HU_MAX": 2000.0,
      "histogram_equalization": true, //是否开启he
      "bins_file": "/host/d/file/histogram_equalization/bins.npy", //替换为本地路径
      "bins_mapped_file": "/host/d/file/histogram_equalization/bins_mapped.npy", //替换为本地路径
      "teacher_n2n_root": "/host/d/file/pre/noise2noise/pred_images/", //n2n结果路径
      "teacher_n2n_epoch": 78,
      "data_len": 1
    }
  },
  "model": {
    "which_model_G": "mri",
    "finetune_norm": false,
    "drop_rate": 0.0,
    "unet": {
      "in_channel": 1,
      "out_channel": 1,
      "inner_channel": 32,
      "norm_groups": 32,
      "channel_multiplier": [1, 2, 4, 8, 8],
      "attn_res": [16],
      "res_blocks": 2,
      "dropout": 0.0,
      "version": "v1"
    },
    "beta_schedule": {
      "train": {
        "schedule": "rev_warmup70",
        "n_timestep": 1000,
        "linear_start": 5e-05,
        "linear_end": 0.01
      },
      "val": {
        "schedule": "rev_warmup70",
        "n_timestep": 1000,
        "linear_start": 5e-05,
        "linear_end": 0.01
      }
    },
    "diffusion": {
      "image_size": 512,
      "channels": 1,
      "conditional": true
    }
  },
  "train": {
    "n_iter": 100000, //iteration次数
    "val_freq": 1000, //验证频率
    "save_checkpoint_freq": 10000, //模型多少iter保存一次
    "print_freq": 100,
    "optimizer": {
      "type": "adam",
      "lr": 0.0001
    },
    "ema_scheduler": {
      "step_start_ema": 5000,
      "update_ema_every": 1,
      "ema_decay": 0.9999
    }
  }, //stage1设置无需修改
  "noise_model": {
    "resume_state": null,
    "initial_stage_file": null,
    "drop_rate": 0.0,
    "unet": {
      "in_channel": 2,
      "out_channel": 1,
      "inner_channel": 32,
      "norm_groups": 32,
      "channel_multiplier": [1, 2, 4, 8, 8],
      "attn_res": [16],
      "res_blocks": 2,
      "dropout": 0.0,
      "version": "v1"
    },
    "beta_schedule": {
      "linear_start": 5e-05,
      "linear_end": 0.01
    },
    "n_iter": 50000,
    "val_freq": 2000,
    "save_checkpoint_freq": 10000,
    "print_freq": 100,
    "optimizer": {
      "type": "adam",
      "lr": 0.0001
    }
  },
  "stage2_file": "experiments/ct_denoise_teacher/stage2_matched.txt" //提前设置好stage2_matched的生成路径
}
