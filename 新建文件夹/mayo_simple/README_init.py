# ============================================
# 在 data/__init__.py 中添加以下代码
# ============================================

# 找到创建dataset的位置，添加mayo分支:

if 'mayo' in dataset_opt.get('name', '').lower():
    from data.mayo_ct_dataset import MayoCTDataset
    dataset = MayoCTDataset(**dataset_opt)
else:
    # 原有逻辑...
    pass
