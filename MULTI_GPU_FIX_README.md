# 多GPU训练和Checkpoint优化说明

## 修复的问题

### 1. 多GPU训练问题 ✅

**原因分析：**
- 原脚本使用 `--include localhost:0` 只指定了单个GPU
- Trainer中使用了 `torch.nn.DataParallel`，与DeepSpeed冲突
- DataParallel和DeepSpeed不能同时使用，会导致模型初始化和数据分发错误

**解决方案：**
1. **脚本修改** (`pretrain_qwen2_distill.sh`)：
   - 移除了 `--include localhost:0`
   - 现在使用 `deepspeed` 命令会自动检测所有可用GPU
   - 如需指定GPU，使用环境变量：`CUDA_VISIBLE_DEVICES=0,1,2,3 bash pretrain_qwen2_distill.sh ...`

2. **Trainer修改** (`tinyllava_distill_trainer.py`)：
   - 移除了 `torch.nn.DataParallel` 包装
   - DeepSpeed会自动处理多GPU分布
   - Teacher模型放置在每个GPU上以支持并行推理

**使用方法：**
```bash
# 使用所有可用GPU（推荐）
bash scripts/train/llava_kd_qwen2/pretrain_qwen2_distill.sh \
    <DATA_PATH> <IMAGE_PATH> <LLM_VERSION> <VT_VERSION> \
    <VT_VERSION2> <CN_VERSION> <VERSION> <TRAIN_RECIPE> <MODEL_MAX_LENGTH>

# 指定使用特定GPU（例如GPU 0,1,2,3）
CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/train/llava_kd_qwen2/pretrain_qwen2_distill.sh \
    <DATA_PATH> <IMAGE_PATH> <LLM_VERSION> <VT_VERSION> \
    <VT_VERSION2> <CN_VERSION> <VERSION> <TRAIN_RECIPE> <MODEL_MAX_LENGTH>
```

### 2. Checkpoint存储优化 ✅

**原因分析：**
- `trainer.save_state()` 会保存完整的优化器状态（可能有数GB）
- 保存了不必要的模型权重（LLM和Vision Tower可以从HuggingFace重新加载）

**解决方案：**
修改 `base.py` 中的 `save()` 方法：
1. **移除优化器状态保存**：不再调用 `trainer.save_state()`
2. **仅保存必要信息**：
   - ✅ Connector权重（唯一需要的训练权重）
   - ✅ Tokenizer配置
   - ✅ 模型配置
   - ✅ 轻量级训练状态（步数、epoch等元信息）
   - ❌ 优化器状态（占用大量空间）
   - ❌ LLM权重（从HuggingFace加载）
   - ❌ Vision Tower权重（从HuggingFace加载）

**存储空间对比：**
- 之前：~10-20GB per checkpoint（包含优化器状态）
- 现在：~50-200MB per checkpoint（仅connector权重）
- **节省空间：95%+**

**保存的文件结构：**
```
checkpoints/${VERSION}-pretrain/
├── config.json                    # 模型配置
├── tokenizer_config.json         # Tokenizer配置
├── tokenizer.json
├── trainer_state.json            # 轻量级训练状态（仅元信息）
└── connector/
    └── pytorch_model.bin         # Connector权重（~50-200MB）
```

## 注意事项

### 多GPU训练
1. **批次大小**：
   - 当前设置：`per_device_train_batch_size=1`，`gradient_accumulation_steps=16`
   - 有效批次大小 = `1 × 4 GPUs × 16 = 64`
   - 可根据显存调整 `per_device_train_batch_size`

2. **显存管理**：
   - Teacher模型（fp16）：~6GB
   - Student模型（fp16）：~1-2GB
   - 激活值和梯度：~4-8GB
   - 建议每个GPU至少16GB显存

3. **数据加载**：
   - `dataloader_num_workers=8`：每个GPU 8个worker
   - 总worker数 = 8 × 4 = 32（确保CPU和内存足够）

### Checkpoint恢复
如果需要从checkpoint恢复训练（虽然优化器状态不会保存）：
1. Connector权重会自动加载
2. 优化器会重新初始化（使用原始学习率）
3. 如果需要完整恢复，可以修改 `save()` 方法包含 `trainer.save_state()`

### DeepSpeed配置
当前使用 `zero2.json`（ZeRO Stage 2）：
- 优化器状态分片
- 梯度分片
- 参数不分片

如果显存不足，可以尝试 `zero3.json`（ZeRO Stage 3）：
```bash
--deepspeed scripts/zero3.json
```

## 测试建议

1. **单GPU测试**：
   ```bash
   CUDA_VISIBLE_DEVICES=0 bash scripts/train/llava_kd_qwen2/pretrain_qwen2_distill.sh ...
   ```

2. **双GPU测试**：
   ```bash
   CUDA_VISIBLE_DEVICES=0,1 bash scripts/train/llava_kd_qwen2/pretrain_qwen2_distill.sh ...
   ```

3. **四GPU训练**：
   ```bash
   CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/train/llava_kd_qwen2/pretrain_qwen2_distill.sh ...
   ```

4. **验证checkpoint大小**：
   ```bash
   du -sh checkpoints/${VERSION}-pretrain/
   du -sh checkpoints/${VERSION}-pretrain/connector/
   ```

## 常见问题

### Q: 多GPU训练时OOM（显存不足）
A: 尝试以下方法：
1. 减小 `per_device_train_batch_size`（例如从1改为1）
2. 增加 `gradient_accumulation_steps`保持有效批次大小
3. 使用 `zero3.json` 而不是 `zero2.json`
4. 减少 `dataloader_num_workers`

### Q: 多GPU训练速度没有明显提升
A: 检查：
1. `dataloader_num_workers` 是否足够（推荐4-8）
2. 数据加载是否成为瓶颈（使用 `lazy_preprocess=True` 已启用）
3. 网络带宽（多节点训练时）

### Q: 想保存优化器状态以便精确恢复
A: 修改 `base.py` 的 `save()` 方法，在保存connector权重后添加：
```python
trainer.save_state()
```
注意：这会增加~10-15GB per checkpoint的存储空间。
