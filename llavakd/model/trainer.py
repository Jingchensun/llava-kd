"""
MobileVLM 自定义训练器

扩展Transformers的Trainer类，专门用于视觉语言模型的训练。

主要功能：
1. 多模态数据的长度分组采样
2. 多种知识蒸馏损失函数（FKL, RKL, TVD, JS, CTKD, DKD, TAID等）
3. 多模态投影器的独立学习率设置
4. 损失加权策略：equal（等权重）/ uncertainty（不确定性自适应）
"""

import torch
from transformers import Trainer
from typing import List, Optional
from torch.utils.data import Sampler
from transformers.trainer import (ALL_LAYERNORM_LAYERS, ShardedDDPOption,
                                  get_parameter_names, has_length,
                                  is_sagemaker_mp_enabled, logger)
from transformers.utils import is_sagemaker_mp_enabled, is_apex_available
from transformers.modeling_utils import unwrap_model
from transformers.models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING_NAMES
import torch.nn.functional as F
from .distil_losses import (
    ForwardKL, ReverseKL, TVD, JS, AdaptiveKL, 
    SkewForwardKL, SkewReverseKL, CTKD, CTKDMLP, DKD, TAID, MSE_Probs, MSE_Logits, Cosine, CosineProbs
)
from .weighting import create_weighting_strategy
import wandb

if is_sagemaker_mp_enabled():
    from transformers.trainer_pt_utils import smp_forward_backward
if is_apex_available():
    from apex import amp
import os

def split_to_even_chunks(indices, lengths, num_chunks):
    """将索引列表按长度分割成大致相等的块"""
    if len(indices) % num_chunks != 0:
        return [indices[i::num_chunks] for i in range(num_chunks)]

    num_indices_per_chunk = len(indices) // num_chunks
    chunks = [[] for _ in range(num_chunks)]
    chunks_lengths = [0 for _ in range(num_chunks)]
    for index in indices:
        shortest_chunk = chunks_lengths.index(min(chunks_lengths))
        chunks[shortest_chunk].append(index)
        chunks_lengths[shortest_chunk] += lengths[index]
        if len(chunks[shortest_chunk]) == num_indices_per_chunk:
            chunks_lengths[shortest_chunk] = float("inf")
    return chunks


def get_modality_length_grouped_indices(lengths, batch_size, world_size, generator=None):
    """按模态和长度对数据进行分组索引（正数=多模态，负数=纯文本）"""
    assert all(l != 0 for l in lengths), "Should not have zero length."
    
    mm_samples = [(i, l) for i, l in enumerate(lengths) if l > 0]
    lang_samples = [(i, -l) for i, l in enumerate(lengths) if l < 0]
    
    if len(mm_samples) == 0 or len(lang_samples) == 0:
        return get_length_grouped_indices(lengths, batch_size, world_size, generator)
    
    mm_indices, mm_lengths = zip(*mm_samples)
    lang_indices, lang_lengths = zip(*lang_samples)

    assert len(mm_indices) > 0, "Should have at least one multimodal sample."
    assert len(lang_indices) > 0, "Should have at least one language sample."

    mm_shuffle = [mm_indices[i] for i in get_length_grouped_indices(mm_lengths, batch_size, world_size, generator=None)]
    lang_shuffle = [lang_indices[i] for i in get_length_grouped_indices(lang_lengths, batch_size, world_size, generator=None)]
    megabatch_size = world_size * batch_size
    mm_megabatches = [mm_shuffle[i : i + megabatch_size] for i in range(0, len(mm_shuffle), megabatch_size)]
    lang_megabatches = [lang_shuffle[i : i + megabatch_size] for i in range(0, len(lang_shuffle), megabatch_size)]

    last_mm = mm_megabatches[-1]
    last_lang = lang_megabatches[-1]
    additional_batch = last_mm + last_lang
    megabatches = mm_megabatches[:-1] + lang_megabatches[:-1]
    megabatch_indices = torch.randperm(len(megabatches), generator=generator)
    megabatches = [megabatches[i] for i in megabatch_indices]

    if len(additional_batch) >= megabatch_size:
        megabatches = [additional_batch[:megabatch_size]] + megabatches
        additional_batch = additional_batch[megabatch_size:]

    if len(additional_batch) > 0:
        megabatches.append(additional_batch)

    return [i for megabatch in megabatches for i in megabatch]


def get_length_grouped_indices(lengths, batch_size, world_size, generator=None, merge=True):
    # We need to use torch for the random part as a distributed sampler will set the random seed for torch.
    indices = torch.randperm(len(lengths), generator=generator)
    megabatch_size = world_size * batch_size
    megabatches = [indices[i : i + megabatch_size].tolist() for i in range(0, len(lengths), megabatch_size)]
    megabatches = [sorted(megabatch, key=lambda i: lengths[i], reverse=True) for megabatch in megabatches]
    megabatches = [split_to_even_chunks(megabatch, lengths, world_size) for megabatch in megabatches]

    return [i for megabatch in megabatches for batch in megabatch for i in batch]


class LengthGroupedSampler(Sampler):
    """按长度和模态分组采样，减少padding浪费"""
    
    def __init__(
        self,
        batch_size: int,
        world_size: int,
        lengths: Optional[List[int]] = None,
        generator=None,
        group_by_modality: bool = False,
    ):
        if lengths is None:
            raise ValueError("Lengths must be provided.")
        self.batch_size = batch_size
        self.world_size = world_size
        self.lengths = lengths
        self.generator = generator
        self.group_by_modality = group_by_modality

    def __len__(self):
        return len(self.lengths)

    def __iter__(self):
        if self.group_by_modality:
            indices = get_modality_length_grouped_indices(
                self.lengths, self.batch_size, self.world_size, generator=self.generator)
        else:
            indices = get_length_grouped_indices(
                self.lengths, self.batch_size, self.world_size, generator=self.generator)
        return iter(indices)

    
class VLMTrainer(Trainer):
    """视觉语言模型训练器，支持知识蒸馏和多模态数据处理"""
    
    def __init__(
        self,
        model=None,
        teacher=None,
        args=None,
        data_collator=None,
        train_dataset=None,
        eval_dataset=None,
        tokenizer=None,
        model_init=None,
        compute_metrics=None,
        callbacks=None,
        optimizers=(None, None),
        preprocess_logits_for_metrics=None,
    ):
        import warnings
        warnings.filterwarnings("ignore", message=".*ByteTensor.*")
        warnings.filterwarnings("ignore", message=".*Output attentions is not supported.*")
        
        super(VLMTrainer, self).__init__(
            model, args, data_collator, train_dataset, eval_dataset, 
            tokenizer, model_init, compute_metrics, callbacks, 
            optimizers, preprocess_logits_for_metrics)
        
        if args.distill == 1:
            self.teacher = teacher
            
            # 注意力适配器：32→16通道
            attn_adapter = torch.nn.Sequential(
                torch.nn.Conv2d(32, 16, 1),
            ).to(device=teacher.device, dtype=teacher.dtype).train()
            
            # 投影适配器：4096→2048维
            proj_adapter = torch.nn.Sequential(
                torch.nn.Conv1d(4096, 2048, 1),
            ).to(device=teacher.device, dtype=teacher.dtype).train()
            
            # 创建损失加权策略
            ratio_strategy = getattr(args, 'distil_ratio_type', 'type1')
            weighting_strategy = create_weighting_strategy(
                ratio_strategy,
                feature_dim=4096,
                num_tasks=3,  # 3个任务：main_loss, logits_distill, v_loss (attn_loss因Flash Attention为0所以不包含)
                hidden_dim=128
            ).to(device=teacher.device)
            # 注意：不要将 weighting_strategy 转换为 fp16，以避免梯度截断
            
            # 注册为模型子模块以支持DeepSpeed
            setattr(self.model, 'distill_attn_adapter', attn_adapter)
            setattr(self.model, 'distill_proj_adapter', proj_adapter)
            setattr(self.model, 'distill_weighting_strategy', weighting_strategy)
            
            self.attn_adapter = self.model.distill_attn_adapter
            self.proj_adapter = self.model.distill_proj_adapter
            self.weighting_strategy = self.model.distill_weighting_strategy
            
            # 初始化蒸馏损失函数
            self.ForwardKL = ForwardKL()
            self.ReverseKL = ReverseKL()
            self.TVD = TVD()
            self.JS = JS()
            self.AdaptiveKL = AdaptiveKL()
            self.SkewForwardKL = SkewForwardKL()
            self.SkewReverseKL = SkewReverseKL()
            self.CTKD = CTKD()
            
            # 获取模型词汇表大小用于CTKDMLP初始化
            vocab_size = getattr(model.config, 'vocab_size', 32000)
            self.CTKDMLP = CTKDMLP(vocab_size=vocab_size)
            
            self.DKD = DKD()
            self.TAID = TAID(
                t_start=getattr(args, 'taid_t_start', 0.4),
                t_end=getattr(args, 'taid_t_end', 1.0),
                alpha=getattr(args, 'taid_alpha', 5e-4),
                beta=getattr(args, 'taid_beta', 0.99),
                disable_adaptive=getattr(args, 'taid_disable_adaptive', False),
            )
            
            # MSE损失：默认在logit空间，mse-probs在概率空间
            mse_temperature = getattr(args, 'mse_temperature', 1.0)
            self.MSE_Logits = MSE_Logits(temperature=mse_temperature)  # 默认MSE
            self.MSE_Probs = MSE_Probs(temperature=mse_temperature)    # 概率空间MSE
            
            # Cosine损失：scale-free，只关注方向
            cosine_temperature = getattr(args, 'cosine_temperature', 1.0)
            self.Cosine = Cosine(temperature=cosine_temperature)
            self.CosineProbs = CosineProbs(temperature=cosine_temperature)  # 概率空间的Cosine distance
            
            # 初始化logits保存计数器（用于调试分析）
            self.save_logit = args.save_logit
            if self.save_logit:
                self.logits_save_counter = 0
                self.logits_save_dir = args.logits_save_dir
                self.logits_save_max_batches = args.logits_save_max_batches
                if not os.path.exists(self.logits_save_dir):
                    os.makedirs(self.logits_save_dir, exist_ok=True)
            
            # 移动蒸馏损失模块到模型设备并注册为子模块
            if hasattr(self, 'model') and self.model is not None:
                device = next(self.model.parameters()).device
                dtype = next(self.model.parameters()).dtype
                
                # 注册CTKD（包含可学习的global_temperature参数）
                self.CTKD = self.CTKD.to(device=device)
                setattr(self.model, 'distill_ctkd', self.CTKD)
                
                # 注册CTKDMLP（包含可学习的MLP参数）
                self.CTKDMLP = self.CTKDMLP.to(device=device, dtype=dtype)
                setattr(self.model, 'distill_ctkdmlp', self.CTKDMLP)
                
                # 注册TAID（包含可学习的temperature_mlp参数）
                self.TAID = self.TAID.to(device=device)
                setattr(self.model, 'distill_taid', self.TAID)
                
                # 注册DKD（包含可学习的alpha和beta参数）
                self.DKD = self.DKD.to(device=device)
                setattr(self.model, 'distill_dkd', self.DKD)
            
            # Mock trainer属性以兼容某些loss函数
            self.trainer = type('MockTrainer', (), {
                'current_epoch': 0,
                'global_step': 0,
                'estimated_stepping_batches': 1000
            })()
            
            self._log_uncertainty_weights = {'main_loss_weighted': 0.0}
            
            # 初始化权重分析器（如果启用）
            if getattr(args, 'enable_weight_analysis', False):
                from .weight_analysis import WeightAnalyzer
                self.weight_analyzer = WeightAnalyzer(
                    save_dir=os.path.join(args.output_dir, 'weight_analysis'),
                    max_samples=getattr(args, 'weight_analysis_max_samples', 1000),
                    save_interval=getattr(args, 'weight_analysis_interval', 10),
                )
                print(f"\n{'='*60}")
                print(f"📊 Weight Analysis Enabled")
                print(f"  Save directory: {os.path.join(args.output_dir, 'weight_analysis')}")
                print(f"  Max samples: {getattr(args, 'weight_analysis_max_samples', 1000)}")
                print(f"  Save interval: {getattr(args, 'weight_analysis_interval', 10)} steps")
                print(f"{'='*60}\n")
            else:
                self.weight_analyzer = None
        
    def _get_train_sampler(self) -> Optional[torch.utils.data.Sampler]:
        """获取训练数据采样器，支持按模态长度分组"""
        if self.train_dataset is None or not has_length(self.train_dataset):
            return None

        if self.args.group_by_modality_length:
            lengths = self.train_dataset.modality_lengths
            return LengthGroupedSampler(
                self.args.train_batch_size,
                world_size=self.args.world_size,
                lengths=lengths,
                group_by_modality=True,
            )
        else:
            return super()._get_train_sampler()

    def create_optimizer(self):
        """创建优化器，支持多模态投影器独立学习率"""
        if is_sagemaker_mp_enabled():
            return super().create_optimizer()
        if self.sharded_ddp == ShardedDDPOption.SIMPLE:
            return super().create_optimizer()

        opt_model = self.model

        if self.optimizer is None:
            decay_parameters = get_parameter_names(
                opt_model, ALL_LAYERNORM_LAYERS)
            decay_parameters = [
                name for name in decay_parameters if "bias" not in name]
            unused_parameters = [
                name for name, _ in opt_model.named_parameters() if "vision_tower" in name and "layers" not in name
            ]
            
            # 蒸馏相关的参数（adapter、weighting_strategy 和 各种distil_loss模块）
            distill_parameters = [
                name for name, _ in opt_model.named_parameters() 
                if "distill_attn_adapter" in name 
                or "distill_proj_adapter" in name 
                or "distill_weighting_strategy" in name
                or "distill_ctkd" in name  # CTKD的global_temperature
                or "distill_ctkdmlp" in name  # CTKDMLP的MLP网络
                or "distill_taid" in name  # TAID的temperature_mlp
                or "distill_dkd" in name  # DKD的alpha和beta
            ]
            
            if self.args.mm_projector_lr is not None and self.args.mm_projector_lr != 0:
                projector_parameters = [
                    name for name, _ in opt_model.named_parameters() if "mm_projector" in name]
                
                # 根据 distil_ratio_type 设置不同的学习率倍数
                ratio_strategy = getattr(self.args, 'distil_ratio_type', 'type1')
                if ratio_strategy == 'type2':
                    distill_weighting_lr = self.args.learning_rate * 1000.0
                elif ratio_strategy == 'type3':
                    distill_weighting_lr = self.args.learning_rate * 10.0
                else:
                    distill_weighting_lr = self.args.learning_rate * 1000.0
                
                optimizer_grouped_parameters = [
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() 
                            if (n in decay_parameters and n not in projector_parameters 
                                and n not in distill_parameters and n not in unused_parameters and p.requires_grad)
                        ],
                        "weight_decay": self.args.weight_decay,
                    },
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() 
                            if (n not in decay_parameters and n not in projector_parameters 
                                and n not in distill_parameters and n not in unused_parameters and p.requires_grad)
                        ],
                        "weight_decay": 0.0,
                    },
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() 
                            if (n in decay_parameters and n in projector_parameters 
                                and n not in unused_parameters and p.requires_grad)
                        ],
                        "weight_decay": self.args.weight_decay,
                        "lr": self.args.mm_projector_lr,
                    },
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() 
                            if (n not in decay_parameters and n in projector_parameters 
                                and n not in unused_parameters and p.requires_grad)
                        ],
                        "weight_decay": 0.0,
                        "lr": self.args.mm_projector_lr,
                    },
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() 
                            if (n in distill_parameters and p.requires_grad)
                        ],
                        "weight_decay": 0.0,
                        "lr": distill_weighting_lr,
                    },
                ]
            else:
                # 根据 distil_ratio_type 设置不同的学习率倍数
                ratio_strategy = getattr(self.args, 'distil_ratio_type', 'type1')
                if ratio_strategy == 'type2':
                    distill_weighting_lr = self.args.learning_rate * 1000.0
                elif ratio_strategy == 'type3':
                    distill_weighting_lr = self.args.learning_rate * 10.0
                else:
                    distill_weighting_lr = self.args.learning_rate * 1000.0
                
                optimizer_grouped_parameters = [
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() 
                            if (n in decay_parameters and n not in distill_parameters and p.requires_grad)
                        ],
                        "weight_decay": self.args.weight_decay,
                    },
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() 
                            if (n not in decay_parameters and n not in distill_parameters and p.requires_grad)
                        ],
                        "weight_decay": 0.0,
                    },
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() 
                            if (n in distill_parameters and p.requires_grad)
                        ],
                        "weight_decay": 0.0,
                        "lr": distill_weighting_lr,
                    },
                ]

            # 打印蒸馏参数信息（用于调试）
            if len(distill_parameters) > 0:
                print(f"\n{'='*60}")
                print(f"🔧 Distillation Parameters Detected:")
                for name in distill_parameters:
                    param = dict(opt_model.named_parameters())[name]
                    print(f"  - {name}: shape={param.shape}, dtype={param.dtype}, requires_grad={param.requires_grad}")
                # 显示实际使用的学习率
                ratio_strategy_for_print = getattr(self.args, 'distil_ratio_type', 'type1')
                if ratio_strategy_for_print == 'type2':
                    lr_multiplier = 1000.0
                elif ratio_strategy_for_print == 'type3':
                    lr_multiplier = 10.0
                else:
                    lr_multiplier = 1000.0
                print(f"  Learning Rate: {distill_weighting_lr:.2e} ({lr_multiplier:.0f}x base LR for {ratio_strategy_for_print})")
                print(f"{'='*60}\n")
            
            optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(
                self.args)
            
            if self.sharded_ddp == ShardedDDPOption.SIMPLE:
                self.optimizer = OSS(
                    params=optimizer_grouped_parameters,
                    optim=optimizer_cls,
                    **optimizer_kwargs,
                )
            else:
                self.optimizer = optimizer_cls(
                    optimizer_grouped_parameters, **optimizer_kwargs)
                if optimizer_cls.__name__ == "Adam8bit":
                    import bitsandbytes

                    manager = bitsandbytes.optim.GlobalOptimManager.get_instance()

                    skipped = 0
                    for module in opt_model.modules():
                        if isinstance(module, nn.Embedding):
                            skipped += sum({p.data_ptr(): p.numel()
                                           for p in module.parameters()}.values())
                            logger.info(
                                f"skipped {module}: {skipped/2**20}M params")
                            manager.register_module_override(
                                module, "weight", {"optim_bits": 32})
                            logger.debug(
                                f"bitsandbytes: will optimize {module} in fp32")
                    logger.info(f"skipped: {skipped/2**20}M params")

        return self.optimizer

    def _wrap_model(self, model):
        """包装模型，设置gradient_checkpointing和蒸馏模块"""
        wrapped_model = super()._wrap_model(model)
        
        if self.args.gradient_checkpointing:
            for module in wrapped_model.modules():
                if hasattr(module, 'gradient_checkpointing'):
                    module.gradient_checkpointing_kwargs = getattr(module, 'gradient_checkpointing_kwargs', {})
                    module.gradient_checkpointing_kwargs['use_reentrant'] = False
        
        # 注册所有蒸馏损失模块到wrapped_model（用于DeepSpeed等分布式训练）
        if self.args.distill == 1:
            try:
                device = next(wrapped_model.parameters()).device
                dtype = next(wrapped_model.parameters()).dtype
                
                # 注册CTKD
                if hasattr(self, 'CTKD'):
                    self.CTKD = self.CTKD.to(device=device)
                    if not hasattr(wrapped_model, 'distill_ctkd'):
                        setattr(wrapped_model, 'distill_ctkd', self.CTKD)
                
                # 注册CTKDMLP
                if hasattr(self, 'CTKDMLP'):
                    self.CTKDMLP = self.CTKDMLP.to(device=device, dtype=dtype)
                    if not hasattr(wrapped_model, 'distill_ctkdmlp'):
                        setattr(wrapped_model, 'distill_ctkdmlp', self.CTKDMLP)
                
                # 注册TAID
                if hasattr(self, 'TAID'):
                    self.TAID = self.TAID.to(device=device)
                    if not hasattr(wrapped_model, 'distill_taid'):
                        setattr(wrapped_model, 'distill_taid', self.TAID)
                
                # 注册DKD
                if hasattr(self, 'DKD'):
                    self.DKD = self.DKD.to(device=device)
                    if not hasattr(wrapped_model, 'distill_dkd'):
                        setattr(wrapped_model, 'distill_dkd', self.DKD)
                        
            except Exception as e:
                print(f"Error moving distillation modules: {e}")
        
        return wrapped_model

    def on_train_end(self):
        """训练结束时保存最终checkpoint和蒸馏模块"""
        import os
        
        if self.args.local_rank == 0 or self.args.local_rank == -1:
            save_checkpoint_final = getattr(self.args, 'save_checkpoint_final', True)
            
            if save_checkpoint_final:
                final_checkpoint_dir = os.path.join(self.args.output_dir, "checkpoint-final")
                try:
                    self.save_model(final_checkpoint_dir)
                    if getattr(self.args, 'save_latest', True):
                        latest_link = os.path.join(self.args.output_dir, "checkpoint-latest")
                        if os.path.islink(latest_link):
                            os.unlink(latest_link)
                        os.symlink(os.path.basename(final_checkpoint_dir), latest_link)
                except Exception as e:
                    print(f"Error saving checkpoint: {e}")
                
                if hasattr(self, 'proj_adapter') and hasattr(self, 'attn_adapter'):
                    try:
                        import torch
                        torch.save(self.proj_adapter.state_dict(), os.path.join(self.args.output_dir, 'proj_adapter_final.pt'))
                        torch.save(self.attn_adapter.state_dict(), os.path.join(self.args.output_dir, 'attn_adapter_final.pt'))
                        
                        if hasattr(self, 'weighting_strategy'):
                            torch.save(self.weighting_strategy.state_dict(), os.path.join(self.args.output_dir, 'weighting_strategy_final.pt'))
                    except Exception as e:
                        print(f"Error saving distillation adapters: {e}")
                
                # 保存权重分析数据
                if hasattr(self, 'weight_analyzer') and self.weight_analyzer is not None:
                    try:
                        print("\n" + "="*60)
                        print("💾 Saving weight analysis data...")
                        self.weight_analyzer.finalize()
                        print("="*60 + "\n")
                    except Exception as e:
                        print(f"Error finalizing weight analyzer: {e}")

    def train(self, resume_from_checkpoint: Optional[str] = None, trial=None, ignore_keys_for_eval=None, **kwargs):
        """训练方法，训练结束后自动保存checkpoint"""
        try:
            result = super().train(resume_from_checkpoint=resume_from_checkpoint, trial=trial, 
                                 ignore_keys_for_eval=ignore_keys_for_eval, **kwargs)
            self.on_train_end()
            return result
        except Exception as e:
            print(f"Training exception: {e}")
            try:
                self.on_train_end()
            except Exception as save_error:
                print(f"Error saving on exception: {save_error}")
            raise e

    def training_step(self, model, inputs):
        """执行单步训练"""
        torch.cuda.empty_cache()
        model.train()
        
        idx = inputs.pop('idx')
        inputs = self._prepare_inputs(inputs)
        inputs['idx'] = idx
        
        if is_sagemaker_mp_enabled():
            loss_mb = smp_forward_backward(model, inputs, self.args.gradient_accumulation_steps)
            return loss_mb.reduce_mean().detach().to(self.args.device)
        
        with self.compute_loss_context_manager():
            loss = self.compute_loss(model, inputs)

        if self.args.n_gpu > 1:
            loss = loss.mean()
        torch.cuda.empty_cache()
        if self.do_grad_scaling:
            self.scaler.scale(loss).backward()
        elif self.use_apex:
            with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            self.accelerator.backward(loss)
        torch.cuda.empty_cache()

        return loss.detach() / self.args.gradient_accumulation_steps
    
    def get_align_kd_loss(self, logits, teacher_logits):
        teacher_probs = F.softmax(teacher_logits, dim=-1, dtype=torch.float16) #
        inf_mask = torch.isinf(logits)
        logprobs = F.log_softmax(logits, dim=-1, dtype=torch.float16) 
        prod_probs = torch.masked_fill(teacher_probs * logprobs, inf_mask, 0)
        x = torch.sum(prod_probs, dim=-1).view(-1)
        distil_loss = -torch.mean(x, dim=0)
        
        return distil_loss

    def get_uncertainty_weights_info(self):
        """获取当前的不确定性权重信息"""
        if hasattr(self, '_log_uncertainty_weights'):
            return self._log_uncertainty_weights.copy()
        else:
            return {'main_loss_weight': 'N/A', 'main_loss_weighted': 'N/A'}

    def get_distil_loss(self, logits, teacher_logits, temperature=2.0):
        """标准知识蒸馏损失（Hinton et al., 2015），使用KL散度和温度缩放"""
        if torch.isnan(logits).any() or torch.isinf(logits).any():
            print("[KD] Warning: Student logits contain NaN or Inf values")
            return torch.tensor(0.0, device=logits.device, dtype=logits.dtype)
        if torch.isnan(teacher_logits).any() or torch.isinf(teacher_logits).any():
            print("[KD] Warning: Teacher logits contain NaN or Inf values")
            return torch.tensor(0.0, device=teacher_logits.device, dtype=teacher_logits.dtype)

        student_log_probs = F.log_softmax(logits / temperature, dim=-1)
        teacher_log_probs = F.log_softmax(teacher_logits / temperature, dim=-1)
        teacher_probs = torch.exp(teacher_log_probs)

        eps = 1e-8
        teacher_probs = torch.clamp(teacher_probs, min=eps)
        teacher_probs = teacher_probs / teacher_probs.sum(dim=-1, keepdim=True)

        distil_loss = F.kl_div(student_log_probs, teacher_probs, reduction='batchmean')

        if torch.isnan(distil_loss) or torch.isinf(distil_loss):
            print("[KD] Warning: Distillation loss is NaN or Inf, returning 0")
            return torch.tensor(0.0, device=logits.device, dtype=logits.dtype)

        distil_loss = distil_loss * (temperature ** 2)

        if torch.isnan(distil_loss) or torch.isinf(distil_loss):
            print("[KD] Warning: Final distillation loss is NaN or Inf, returning 0")
            return torch.tensor(0.0, device=logits.device, dtype=logits.dtype)

        return distil_loss

    def get_v_loss(self, teacher_front_attn, t_v_mask, v_feature, teacher_v_feature, k=16):
        """计算视觉特征蒸馏损失，基于注意力权重选择重要特征"""
        t_v_mask = t_v_mask[:, 0, :, :]
        teacher_front_attn = teacher_front_attn.mean(1)
        
        top_attn, top_idx = teacher_front_attn.sum(-2).topk(k, dim=-1)
        
        if t_v_mask.sum(-2)[0, 1] == True:
            top_idx = top_idx - 1
        else:
            top_idx = top_idx - 2
            
        batch_idx = torch.arange(top_idx.shape[0]).unsqueeze(-1).repeat(1, top_idx.shape[-1]).flatten()
        top_idx = top_idx.flatten()
        
        v_feature_pick = v_feature[batch_idx, top_idx]
        teacher_v_feature_pick = teacher_v_feature[batch_idx, top_idx]
        
        v_loss = ((self.proj_adapter(teacher_v_feature_pick.unsqueeze(1).permute(0, 2, 1)) - 
                  v_feature_pick.unsqueeze(1).permute(0, 2, 1))**2).mean()
        v_loss_all = ((self.proj_adapter(teacher_v_feature.permute(0, 2, 1)) - 
                      v_feature.permute(0, 2, 1))**2).mean()
        
        return v_loss + v_loss_all
    
    def compute_loss(self, model, inputs, return_outputs=False):
        """计算训练损失，支持知识蒸馏和多种蒸馏策略"""
        if self.label_smoother is not None and "labels" in inputs:
            labels = inputs.pop("labels")
        else:
            labels = None
        
        original_labels = labels
        
        idx = inputs.pop('idx')
        if len(idx) == 0:
            idx = None
        else:
            idx = idx[0]
            
        outputs, v_feature, front_attn, t_v_mask = model(**inputs)
        
        if self.args.distill == 1 and idx is not None:
            with torch.no_grad():
                teacher_outputs, teacher_v_feature, teacher_front_attn, _ = self.teacher(**inputs)
                teacher_logits = teacher_outputs['logits']
                teacher_front_attn = teacher_front_attn

        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        if labels is not None:
            if unwrap_model(model)._get_name() in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES.values():
                loss = self.label_smoother(outputs, labels, shift_labels=True)
            else:
                loss = self.label_smoother(outputs, labels)
        else:
            if isinstance(outputs, dict) and "loss" not in outputs:
                raise ValueError(
                    "The model did not return a loss from the inputs, only the following keys: "
                    f"{','.join(outputs.keys())}. For reference, the inputs it received are {','.join(inputs.keys())}."
                )
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]
            if t_v_mask is None:
                return (loss, outputs) if return_outputs else loss
            if idx is None:
                print('skip')
                return (loss*0.0001, outputs) if return_outputs else loss*0.0001
            
            if self.args.distill == 1:
                # 构建mask
                if original_labels is not None:
                    mask = (original_labels != -100).float()
                else:
                    mask = inputs.get('attention_mask', torch.ones_like(outputs['logits'][:, :, 0])).float()
                
                # 对齐mask和logits维度（处理图像token导致的长度差异）
                logits_len = outputs['logits'].shape[1]
                if mask.shape[1] != logits_len:
                    if mask.shape[1] < logits_len:
                        # padding（图像token位置设为0，不参与蒸馏）
                        pad = torch.zeros(mask.shape[0], logits_len - mask.shape[1], 
                                         device=mask.device, dtype=mask.dtype)
                        mask = torch.cat([pad, mask], dim=1)
                    else:
                        # 截断
                        mask = mask[:, :logits_len]
                print(f"Logits Shape: {outputs['logits'].shape}, Teacher Logits Shape: {teacher_logits.shape}, Mask Shape: {mask.shape}")
                
                # 保存logits和mask用于后续分析（可通过配置控制）
                if self.save_logit and hasattr(self, 'logits_save_counter') and self.logits_save_counter < self.logits_save_max_batches:
                    save_data = {
                        'student_logits': outputs['logits'].detach().cpu(),
                        'teacher_logits': teacher_logits.detach().cpu(),
                        'mask': mask.detach().cpu(),
                        'batch_idx': self.logits_save_counter,
                    }
                    save_path = os.path.join(self.logits_save_dir, f'batch_{self.logits_save_counter:05d}.pt')
                    torch.save(save_data, save_path)
                    self.logits_save_counter += 1
                    if self.logits_save_counter == self.logits_save_max_batches:
                        print(f"已保存{self.logits_save_max_batches}个batch的logits数据到 {self.logits_save_dir}")
                
                distil_type = getattr(self.args, 'distil_type', 'kl')
                
                if distil_type == 'align-kd':
                    v_logits_distill = self.get_align_kd_loss(outputs['logits'], teacher_logits)
                elif distil_type == 'mse':
                    v_logits_distill = self.MSE_Logits(outputs['logits'], teacher_logits, mask)
                elif distil_type == 'mse-probs':
                    # 使用MSE损失（在概率空间计算，更稳定）
                    v_logits_distill = self.MSE_Probs(outputs['logits'], teacher_logits, mask)
                elif distil_type == 'cosine':
                    # 使用Cosine distance损失（scale-free，只关注方向）
                    v_logits_distill = self.Cosine(outputs['logits'], teacher_logits, mask)
                elif distil_type == 'cosine-probs':
                    # 使用Cosine distance损失（在概率空间计算，更稳定）
                    v_logits_distill = self.CosineProbs(outputs['logits'], teacher_logits, mask)
                elif distil_type == 'fkl':
                    v_logits_distill = self.ForwardKL(outputs['logits'], teacher_logits, mask)
                elif distil_type == 'rkl':
                    v_logits_distill = self.ReverseKL(outputs['logits'], teacher_logits, mask)
                elif distil_type == 'tvd':
                    v_logits_distill = self.TVD(outputs['logits'], teacher_logits, mask)
                elif distil_type == 'js':
                    v_logits_distill = self.JS(outputs['logits'], teacher_logits, mask)
                elif distil_type == 'adaptive_kl':
                    v_logits_distill = self.AdaptiveKL(outputs['logits'], teacher_logits, mask)
                elif distil_type == 'sfkl':
                    v_logits_distill = self.SkewForwardKL(outputs['logits'], teacher_logits, mask)
                elif distil_type == 'srkl':
                    v_logits_distill = self.SkewReverseKL(outputs['logits'], teacher_logits, mask)
                elif distil_type == 'ctkd':
                    # 使用已注册到模型的CTKD模块，确保参数可被优化
                    ctkd_module = getattr(self.model, 'distill_ctkd', self.CTKD)
                    v_logits_distill = ctkd_module(
                        lightning_module=self,
                        logits=outputs['logits'], 
                        teacher_logits=teacher_logits, 
                        mask=mask,
                        batch={'model_inputs': inputs}
                    )
                elif distil_type == 'ctkd-mlp':
                    # 使用已注册到模型的CTKDMLP模块，确保参数可被优化
                    ctkdmlp_module = getattr(self.model, 'distill_ctkdmlp', self.CTKDMLP)
                    v_logits_distill = ctkdmlp_module(
                        lightning_module=self,
                        logits=outputs['logits'], 
                        teacher_logits=teacher_logits, 
                        mask=mask,
                        batch={'model_inputs': inputs}
                    )
                elif distil_type == 'dkd':
                    # 使用已注册到模型的DKD模块，确保参数可被优化
                    dkd_module = getattr(self.model, 'distill_dkd', self.DKD)
                    v_logits_distill = dkd_module(
                        lightning_module=self,
                        logits=outputs['logits'], 
                        teacher_logits=teacher_logits, 
                        mask=mask,
                        batch={'model_inputs': inputs}
                    )
                elif distil_type == 'taid':
                    # 使用已注册到模型的TAID模块，确保参数可被优化
                    taid_module = getattr(self.model, 'distill_taid', self.TAID)
                    taid_result = taid_module(
                        lightning_module=self,
                        logits=outputs['logits'], 
                        teacher_logits=teacher_logits, 
                        mask=mask,
                        batch={'model_inputs': inputs}
                    )
                    if isinstance(taid_result, dict):
                        v_logits_distill = taid_result['distil_loss']
                    else:
                        v_logits_distill = taid_result
                else:
                    v_logits_distill = self.get_distil_loss(outputs['logits'], teacher_logits)
                
                if teacher_front_attn is None or front_attn is None or t_v_mask is None:
                    v_loss = ((self.proj_adapter(teacher_v_feature.permute(0, 2, 1)) - 
                              v_feature.permute(0, 2, 1))**2).mean()
                    # 创建一个在计算图中的零值张量（与loss保持梯度连接）
                    attn_loss = loss * 0.0
                else:
                    if teacher_front_attn.shape[-1] < 144:
                        v_loss = ((self.proj_adapter(teacher_v_feature.permute(0, 2, 1)) - 
                                  v_feature.permute(0, 2, 1))**2).mean()
                    else:
                        v_loss = self.get_v_loss(teacher_front_attn, t_v_mask, v_feature, teacher_v_feature)
                    attn_loss = ((self.attn_adapter(teacher_front_attn) - front_attn)[t_v_mask != 0]**2).mean()
                
                # 检测并修复 v_loss 中的 NaN/Inf（防止训练崩溃）
                if torch.isnan(v_loss) or torch.isinf(v_loss):
                    print(f"[Warning] v_loss is NaN/Inf at step {self.state.global_step}, replacing with zero.")
                    v_loss = loss * 0.0  # 保持在计算图中

                # 使用加权策略
                if not hasattr(self, '_step_counter'):
                    self._step_counter = 0
                self._step_counter += 1
                
                logging_steps = getattr(self.args, 'logging_steps', 5)
                
                try:
                    # 调用加权策略
                    available_losses = [loss, v_logits_distill]
                    # available_losses = [loss, v_logits_distill, v_loss]
                    weighted_loss, loss_weights = self.weighting_strategy(
                        *available_losses,
                        teacher_features=teacher_v_feature
                    )
                    loss = weighted_loss
                    
                    # 记录权重信息
                    if hasattr(self, '_log_uncertainty_weights'):
                        self._log_uncertainty_weights = loss_weights
                    
                    # 收集权重分析数据
                    if hasattr(self, 'weight_analyzer') and self.weight_analyzer is not None:
                        try:
                            loss_values = {
                                'main_loss': loss.item() if hasattr(loss, 'item') else float(loss),
                                'logits_distill_loss': v_logits_distill.item() if hasattr(v_logits_distill, 'item') else float(v_logits_distill),
                                'v_loss': v_loss.item() if hasattr(v_loss, 'item') else float(v_loss),
                            }
                            
                            self.weight_analyzer.collect_sample_data(
                                step=self.state.global_step,
                                student_logits=outputs['logits'].detach(),
                                teacher_logits=teacher_logits.detach(),
                                mask=mask,
                                loss_values=loss_values,
                                weights=loss_weights,
                                sample_ids=[idx] if not isinstance(idx, list) else idx,
                            )
                        except Exception as e:
                            print(f"[Warning] Weight analyzer error: {e}")
                    
                    # 定期打印日志和记录到wandb
                    if self._step_counter % logging_steps == 0:
                        weight_info = []
                        loss_info = []
                        wandb_log_dict = {
                            "train/total_loss": weighted_loss.item(),
                            "train/learning_rate": self.optimizer.param_groups[0]['lr'],
                            "train/step": self._step_counter,
                        }
                        
                        for key, value in loss_weights.items():
                            if key.endswith('_weight') and not key.endswith('_weighted'):
                                task_name = key.replace('_weight', '').replace('_', ' ').title()
                                weight_info.append(f"{task_name}: {value:.4f}")
                                # 记录权重到wandb
                                wandb_key = f"weights/{key}"
                                wandb_log_dict[wandb_key] = value
                            elif key.endswith('_weighted'):
                                task_name = key.replace('_weighted', '').replace('_', ' ').title()
                                loss_info.append(f"{task_name}: {value:.4f}")
                                # 记录加权loss到wandb
                                wandb_key = f"weighted_losses/{key}"
                                wandb_log_dict[wandb_key] = value
                        
                        if weight_info:
                            print(f"[Step {self._step_counter}] Weights - " + ", ".join(weight_info))
                        if loss_info:
                            print(f"[Step {self._step_counter}] Weighted - " + ", ".join(loss_info))
                        
                        # 打印 log_vars 的值（仅对 type2 策略）
                        if hasattr(self.weighting_strategy, 'log_vars'):
                            log_vars_str = ", ".join([f"{v:.4f}" for v in self.weighting_strategy.log_vars.detach().cpu().tolist()])
                            print(f"[Step {self._step_counter}] Log_vars - [{log_vars_str}]")
                            # 记录log_vars到wandb
                            for i, log_var in enumerate(self.weighting_strategy.log_vars.detach().cpu().tolist()):
                                wandb_log_dict[f"log_vars/task_{i}"] = log_var
                        
                        # 上传到wandb（仅在rank 0进程中）
                        if self.args.local_rank == 0 or self.args.local_rank == -1:
                            if wandb.run is not None:
                                wandb.log(wandb_log_dict, step=self.state.global_step)
                            
                except Exception as e:
                    print(f"Warning: Weighting strategy failed, using simple sum: {e}")
                    loss = sum([loss, v_logits_distill])
            torch.cuda.empty_cache()

        return (loss, outputs) if return_outputs else loss
