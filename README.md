# FarmSeg-Net

## 项目简介
FarmSeg-Net 是一个面向农田场景三维点云的语义分割网络，实现于 TensorFlow compat.v1 框架。项目在 RandLA-Net 的基础上，结合多层嵌套解码结构和注意力特征融合模块，针对大规模稀疏农田点云的地物分类（如建筑、农田、硬地、道路、树木等）进行优化。数据组织采用类似 S3DIS 的 Area_x/.../Annotations 风格，并配套提供从原始标注文本生成 PLY 点云、KDTree 与重投影索引的完整预处理流水线。为兼顾精度与效率，项目集成了 C++/Cython 实现的 KNN 搜索与网格下采样算子，支持在单 GPU 上对大规模农田场景进行端到端训练、测试与特征可视化（UMAP / PCA / t-SNE）。

## 环境与依赖
Name                      Version                  Build  Channel
absl-py                   0.15.0                   pypi_0    pypi
astunparse                1.6.3                    pypi_0    pypi
cached-property           1.5.2                    pypi_0    pypi
cachetools                4.2.4                    pypi_0    pypi
certifi                   2024.12.14               pypi_0    pypi
charset-normalizer        2.0.12                   pypi_0    pypi
clang                     5.0                      pypi_0    pypi
cycler                    0.11.0                   pypi_0    pypi
cython                    0.29.15                  pypi_0    pypi
dataclasses               0.8                      pypi_0    pypi
flatbuffers               1.12                     pypi_0    pypi
gast                      0.4.0                    pypi_0    pypi
google-auth               2.22.0                   pypi_0    pypi
google-auth-oauthlib      0.4.6                    pypi_0    pypi
google-pasta              0.2.0                    pypi_0    pypi
grpcio                    1.48.2                   pypi_0    pypi
h5py                      3.1.0                    pypi_0    pypi
idna                      3.10                     pypi_0    pypi
importlib-metadata        4.8.3                    pypi_0    pypi
importlib-resources       5.4.0                    pypi_0    pypi
joblib                    1.1.1                    pypi_0    pypi
keras                     2.6.0                    pypi_0    pypi
keras-preprocessing       1.1.2                    pypi_0    pypi
kiwisolver                1.3.1                    pypi_0    pypi
llvmlite                  0.36.0                   pypi_0    pypi
markdown                  3.3.7                    pypi_0    pypi
matplotlib                3.3.4                    pypi_0    pypi
numba                     0.53.1                   pypi_0    pypi
numpy                     1.19.5                   pypi_0    pypi
oauthlib                  3.2.2                    pypi_0    pypi
open3d-python             0.3.0.0                  pypi_0    pypi
openssl                   1.0.2l                        0    https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free
opt-einsum                3.3.0                    pypi_0    pypi
pandas                    0.25.3                   pypi_0    pypi
pillow                    8.4.0                    pypi_0    pypi
pip                       21.3.1                   pypi_0    pypi
plyfile                   0.8                      pypi_0    pypi
protobuf                  3.19.6                   pypi_0    pypi
pyasn1                    0.5.1                    pypi_0    pypi
pyasn1-modules            0.3.0                    pypi_0    pypi
pynndescent               0.5.13                   pypi_0    pypi
pyparsing                 3.0.7                    pypi_0    pypi
python                    3.6.2                         0    https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free
python-dateutil           2.9.0.post0              pypi_0    pypi
pytz                      2024.2                   pypi_0    pypi
pyyaml                    5.4                      pypi_0    pypi
readline                  6.2                           2    https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free
requests                  2.27.1                   pypi_0    pypi
requests-oauthlib         2.0.0                    pypi_0    pypi
rsa                       4.9                      pypi_0    pypi
scikit-learn              0.22.2                   pypi_0    pypi
scipy                     1.4.1                    pypi_0    pypi
seaborn                   0.11.2                   pypi_0    pypi
setuptools                59.6.0                   pypi_0    pypi
six                       1.15.0                   pypi_0    pypi
sqlite                    3.13.0                        0    https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free
tensorboard               2.10.1                   pypi_0    pypi
tensorboard-data-server   0.6.1                    pypi_0    pypi
tensorboard-plugin-wit    1.8.1                    pypi_0    pypi
tensorflow-estimator      2.8.0                    pypi_0    pypi
tensorflow-gpu            2.6.0                    pypi_0    pypi
termcolor                 1.1.0                    pypi_0    pypi
threadpoolctl             3.1.0                    pypi_0    pypi
tk                        8.5.18                        0    https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free
tqdm                      4.64.1                   pypi_0    pypi
typing-extensions         3.7.4.3                  pypi_0    pypi
umap-learn                0.5.7                    pypi_0    pypi
urllib3                   1.26.20                  pypi_0    pypi
werkzeug                  2.0.3                    pypi_0    pypi
wheel                     0.37.1                   pypi_0    pypi
wrapt                     1.12.1                   pypi_0    pypi
xz                        5.2.3                         0    https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free
zipp                      3.6.0                    pypi_0    pypi
zlib                      1.2.11                        0    https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free

## 数据预处理
### 数据准备
默认数据根目录为 `./data/S3DIS`（见 `main_S3DIS.py` 中 `self.path = './data/S3DIS'`）。

`utils/data_prepare_s3dis.py` 会读取：

- 原始标注目录：`./data/S3DIS/Stanford3dDataset_v1.2_Aligned_Version`
- 标注子目录列表：`utils/meta/anno_paths.txt`
- 类别名：`utils/meta/class_names.txt`
- 
将数据放到如下路径（或自行改 `utils/data_prepare_s3dis.py` 里的 `dataset_path`）：

```text
data/
  S3DIS/
    Stanford3dDataset_v1.2_Aligned_Version/
      Area_1/...
      Area_2/...
      ...
```
### 数据预处理
```bash
python utils/data_prepare_s3dis.py
```

它会在 `data/S3DIS/` 下生成：

- `original_ply/`：每个场景一个 `.ply`（含 `xyzrgb + class`）
- `input_0.040/`：下采样后的 `.ply`、KDTree（`*_KDTree.pkl`）以及投影索引（`*_proj.pkl`）

---

## 训练与测试
### 训练
以 `Area_5` 作为测试区域（其它 Area 用于训练）：
```bash
python -B main_S3DIS.py --gpu 0 --mode train --test_area 5
```
训练日志会输出到类似 `log_train_S3DISArea_5.txt`，并在 `results/Log_*/snapshots/` 下保存模型快照（`snap-*.meta` 等）。

### 测试

默认会自动选择 `results/` 下最新一次训练的最新快照进行测试：

```bash
python -B main_S3DIS.py --gpu 0 --mode test --test_area 5
```

也可以手动指定快照路径（不带 `.meta` 后缀）：

```bash
python -B main_S3DIS.py --gpu 0 --mode test --test_area 5 --model_path "results/Log_xxx/snapshots/snap-16001"
```

测试输出：
- 日志：`log_test_Area_5.txt`
- 预测结果：`test/Log_*/val_preds/*.ply`（每个文件包含 `pred` 与 `label` 字段）

---
### 特征可视化（UMAP/PCA/t-SNE）

`main_S3DIS.py` 的 `--mode vis` 会从验证集迭代器中抽取若干 batch，取网络中的 `feature_fused` 特征做降维并保存图片：

```bash
python -B main_S3DIS.py --gpu 0 --mode vis --test_area 5 --vis_method umap --vis_dim 2
```

常用参数：
- `--vis_method`：`umap` / `pca` / `tsne`
- `--vis_dim`：`2` 或 `3`
- `--max_points`：可视化最大点数（默认 400000）
- `--custom_colors`：启用预设类别颜色（见 `main_S3DIS.py`）

输出目录：`visualization/`（同时会保存 `visualization/feature_data/features.npy` 与 `labels.npy`）。

---







