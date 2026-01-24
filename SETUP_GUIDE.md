# LLaVA-KD 训练环境配置指南

本指南记录了修复 `train_qwen2_distill.sh` 脚本的所有步骤和创建的工具。

## 🔧 已修复的问题

### 1. 文件路径错误
**问题**: 脚本使用了错误的相对路径,导致找不到训练脚本文件。

**错误信息**:
```
can't open file 'llavakd/train/train_distill_qwen2.py': [Errno 2] No such file or directory
```

**解决方案**: 
- 修改了 `pretrain_qwen2_distill.sh`、`finetune_qwen2_sft.sh` 和 `finetune_distill_after_qwen2_sft.sh`
- 添加了 `cd "$(dirname "$0")/../../.."` 切换到项目根目录
- 更新了所有相对路径为从项目根目录开始

### 2. Python 模块导入错误
**问题**: Python 无法找到 `llavakd` 模块。

**错误信息**:
```
ModuleNotFoundError: No module named 'llavakd'
```

**解决方案**:
- 在脚本中添加了 `export PYTHONPATH="${PWD}:${PYTHONPATH}"`
- 确保 Python 可以从项目根目录导入模块

### 3. 缺少教师模型
**问题**: 知识蒸馏训练需要预训练的教师模型。

**解决方案**:
- 创建了 `download_teacher_model.py` 脚本
- 使用 Hugging Face 开源模型 `Zhang199/TinyLLaVA-Qwen2.5-3B-SigLIP`
- 创建了 `convert_teacher_model.py` 转换模型格式

### 4. 模型格式不兼容
**问题**: 下载的模型格式与训练脚本期望的格式不同。

**解决方案**:
- 修改了 `train_distill_qwen2.py` 中的教师模型加载方式
- 使用 `LLaVAKD.from_pretrained()` 直接加载 Hugging Face 格式的模型

## 📦 创建的工具

### 1. `download_teacher_model.py`
下载教师模型的脚本。

**使用方法**:
```bash
# 下载默认教师模型 (Zhang199/TinyLLaVA-Qwen2.5-3B-SigLIP)
python download_teacher_model.py

# 下载其他模型
python download_teacher_model.py --repo-id YOUR_REPO_ID

# 下载基础组件 (Vision Encoder, LLM)
python download_teacher_model.py --components
```

**功能**:
- 从 Hugging Face 自动下载教师模型
- 支持断点续传
- 提供详细的下载说明

### 2. `convert_teacher_model.py`
转换模型格式的脚本。

**使用方法**:
```bash
# 转换默认路径的模型
python convert_teacher_model.py

# 指定源目录和目标目录
python convert_teacher_model.py --source-dir /path/to/source --target-dir /path/to/target
```

**功能**:
- 将 safetensors 格式转换为 pytorch_model.bin
- 分离 vision_tower、connector 和 language_model 组件
- 创建训练脚本期望的目录结构

## 🚀 使用流程

### 步骤 1: 激活环境
```bash
conda activate llava-kd
```

### 步骤 2: 下载教师模型
```bash
cd /home/csgrad/jsun39/llava-kd
python download_teacher_model.py
```

### 步骤 3: 转换模型格式
```bash
python convert_teacher_model.py
```

### 步骤 4: 运行训练脚本
```bash
cd scripts/train/llava_kd_qwen2
bash train_qwen2_distill.sh
```

## 📁 目录结构

训练完成后的目录结构:
```
llava-kd/
├── pretrained_checkpoints/
│   └── LLaVA_KD_ckpts/
│       └── tiny-llava-Qwen2.5-3B-siglip-so400m-patch14-384-qwen2-0_5b_base-finetune/
│           ├── vision_tower/
│           │   └── pytorch_model.bin (817MB)
│           ├── connector/
│           │   └── pytorch_model.bin (13MB)
│           ├── language_model/
│           │   ├── pytorch_model.bin (6.4GB)
│           │   ├── config.json
│           │   └── [其他配置文件]
│           └── config.json
├── checkpoints/  # 训练输出目录
├── llavakd/  # 源代码
├── scripts/  # 训练脚本
├── download_teacher_model.py  # 下载工具
├── convert_teacher_model.py  # 转换工具
└── SETUP_GUIDE.md  # 本文档
```

## 🔍 修改的文件清单

### 训练脚本
1. `scripts/train/llava_kd_qwen2/pretrain_qwen2_distill.sh`
   - 添加了 `cd` 命令切换到项目根目录
   - 添加了 `PYTHONPATH` 环境变量
   - 修正了所有相对路径

2. `scripts/train/llava_kd_qwen2/finetune_qwen2_sft.sh`
   - 同上

3. `scripts/train/llava_kd_qwen2/finetune_distill_after_qwen2_sft.sh`
   - 同上

### Python 训练代码
4. `llavakd/train/train_distill_qwen2.py`
   - 修改了教师模型加载方式
   - 使用 `LLaVAKD.from_pretrained()` 替代手动加载组件

## 📝 注意事项

1. **数据集路径**: 确保数据集文件存在于正确的路径:
   - 预训练数据: `../../../dataset/text_files/blip_laion_cc_sbu_558k.json`
   - 预训练图像: `../../../dataset/llava/llava_pretrain/images`

2. **GPU 要求**: 脚本默认使用 4 个 GPU (localhost:0,1,2,3)

3. **模型大小**: 教师模型约 7.2GB,确保有足够的磁盘空间

4. **训练时间**: 知识蒸馏训练可能需要较长时间,建议使用 `nohup` 或 `screen` 在后台运行

## 🐛 常见问题

### Q: 如何更换教师模型?
A: 修改 `llavakd/train/train_distill_qwen2.py` 中的 `teacher_dir` 变量,或下载其他模型后使用 `convert_teacher_model.py` 转换。

### Q: 训练中断后如何恢复?
A: DeepSpeed 支持断点续传,检查 `--output_dir` 中的检查点文件。

### Q: 如何修改 GPU 数量?
A: 修改 `pretrain_qwen2_distill.sh` 中的 `--include localhost:0,1,2,3` 参数。

## 📚 参考资源

- 教师模型: [Zhang199/TinyLLaVA-Qwen2.5-3B-SigLIP](https://huggingface.co/Zhang199/TinyLLaVA-Qwen2.5-3B-SigLIP)
- TinyLLaVA Factory: [GitHub](https://github.com/TinyLLaVA/TinyLLaVA_Factory)
- LLaVA: [GitHub](https://github.com/haotian-liu/LLaVA)

## ✅ 验证清单

在运行训练之前,请确认:
- [ ] 已激活 `llava-kd` 环境
- [ ] 教师模型已下载并转换
- [ ] 数据集文件存在
- [ ] GPU 可用 (`nvidia-smi`)
- [ ] 有足够的磁盘空间 (至少 50GB)
- [ ] PYTHONPATH 已正确设置

---

**最后更新**: 2026-01-23
**作者**: AI Assistant
