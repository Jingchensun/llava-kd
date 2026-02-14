"""
知识蒸馏的损失加权策略实现

支持4种策略：
1. type1 (EqualWeighting): 等权重相加
2. type2 (HeteroscedasticUncertainty): 基于不确定性的任务权重学习 
3. type3 (InstanceConditionalWeighting): 基于实例特征的条件权重
4. type4 (TrueInstanceWiseUncertainty): 真正的实例级不确定性加权

向后兼容旧名称：
- "equal" -> type1
- "uncertainty" -> type3
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================================
# Type 1: Equal Weighting (基线)
# ============================================================================
class EqualWeighting(nn.Module):
    """简单相加，所有任务等权重"""
    
    def __init__(self):
        super().__init__()
    
    def forward(self, *losses, teacher_features=None):
        """
        Args:
            *losses: 可变数量的标量损失
            teacher_features: 未使用，仅为了接口统一
        Returns:
            total_loss: 标量
            weights: dict, 用于logging
        """
        total_loss = sum(losses)
        
        # 构建权重字典
        task_names = ['main_loss', 'logits_distill_loss', 'v_loss', 'attn_loss']
        weights = {}
        for i, loss in enumerate(losses):
            name = task_names[i] if i < len(task_names) else f'task_{i}'
            weights[f'{name}_weight'] = 1.0
            weights[f'{name}_weighted'] = loss.item() if hasattr(loss, 'item') else loss
        
        return total_loss, weights


# ============================================================================
# Type 2: Heteroscedastic Uncertainty (Kendall et al. 2018)
# ============================================================================
class HeteroscedasticUncertainty(nn.Module):
    """
    基于任务不确定性的损失加权 (Kendall et al. 2018)
    
    公式: L_total = Σ_i (1/(2σ_i²)) * L_i + log(σ_i)
    仅需学习 num_tasks 个参数
    """
    
    def __init__(self, num_tasks=2):
        super().__init__()
        self.num_tasks = num_tasks
        # 初始化为 0.0，这样 sigma = exp(0.0) = 1.0, precision = 1/sigma^2 = 1.0
        # 使用 float32 避免 fp16 梯度截断
        # 注意：根据 Kendall 2018，log(sigma) 可以从 0 开始，但为了更快收敛，
        # 可以考虑初始化为稍大的值（如 uniform(-1, 1)）
        self.log_vars = nn.Parameter(torch.zeros(num_tasks, dtype=torch.float32))
        # 如果收敛太慢，可以用: torch.randn(num_tasks, dtype=torch.float32) * 0.1
        
    def forward(self, *losses, teacher_features=None):
        """
        Args:
            *losses: 可变数量的标量损失
            teacher_features: 未使用，仅为了接口统一
        Returns:
            total_loss: 加权后的总损失
            weights: dict
        """
        # 使用 float32 计算 precision 以避免 fp16 梯度截断
        # 限制 log_vars 在合理范围内，避免数值溢出
        # log_vars ∈ [-5, 5] 意味着 sigma ∈ [e^-5, e^5] ≈ [0.0067, 148.4]
        # 注意：不使用.float()，因为self.log_vars已经是float32，避免断开梯度连接
        log_vars_clamped = torch.clamp(self.log_vars, min=-5.0, max=5.0)
        precision = torch.exp(-log_vars_clamped)
        
        total_loss = 0
        for i, loss in enumerate(losses):
            # 确保损失值转换为 float32 进行计算
            loss_fp32 = loss.float() if loss.dtype == torch.float16 else loss
            weighted = 0.5 * precision[i] * loss_fp32 + 0.5 * log_vars_clamped[i]
            # 转回原来的 dtype
            total_loss += weighted.to(loss.dtype) if loss.dtype == torch.float16 else weighted
        
        # 构建权重字典
        task_names = ['main_loss', 'logits_distill_loss', 'v_loss', 'attn_loss']
        weights = {}
        for i, loss in enumerate(losses):
            name = task_names[i] if i < len(task_names) else f'task_{i}'
            weights[f'{name}_weight'] = (0.5 * precision[i]).item()
            loss_fp32 = loss.float() if loss.dtype == torch.float16 else loss
            weights[f'{name}_weighted'] = (0.5 * precision[i] * loss_fp32).item()
        
        return total_loss, weights


# ============================================================================
# Type 3: Instance-Conditional Weighting
# ============================================================================
class InstanceConditionalWeighting(nn.Module):
    """
    基于实例特征的条件权重（原 uncertainty 方案）
    使用MLP根据teacher特征预测每个batch的任务权重
    """
    
    def __init__(self, feature_dim=4096, num_tasks=2, hidden_dim=128):
        super().__init__()
        self.num_tasks = num_tasks
        self.uncertainty_predictor = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_tasks)
        )
    
    def forward(self, *losses, teacher_features=None):
        """
        Args:
            *losses: 可变数量的标量损失
            teacher_features: (batch_size, feature_dim) 或 (batch_size, seq_len, feature_dim)
        Returns:
            total_loss: 加权后的总损失
            weights: dict
        """
        if teacher_features is None:
            # Fallback to equal weighting
            return EqualWeighting().forward(*losses)
        
        # 处理teacher_features维度
        if teacher_features.dim() == 3:
            teacher_features = teacher_features.mean(dim=1)
        
        batch_size = teacher_features.shape[0]
        
        # 预测log_vars并限制在合理范围内，避免数值溢出
        # 将teacher_features转换为float32以匹配MLP的dtype
        teacher_features_fp32 = teacher_features.detach().float()
        log_vars = self.uncertainty_predictor(teacher_features_fp32)
        log_vars = torch.clamp(log_vars, min=-5.0, max=5.0)
        
        # 将标量损失扩展为batch维度
        losses_tensor = []
        for loss in losses:
            loss_batch = loss.unsqueeze(0).repeat(batch_size) if loss.dim() == 0 else loss
            losses_tensor.append(loss_batch)
        losses_tensor = torch.stack(losses_tensor, dim=1)
        
        # 计算加权损失
        precision = torch.exp(-log_vars)
        weighted_losses = precision * losses_tensor + log_vars
        total_loss = torch.mean(torch.sum(weighted_losses, dim=1))
        
        # 构建权重字典
        task_names = ['main_loss', 'logits_distill_loss', 'v_loss', 'attn_loss']
        weights = {}
        individual_weights = torch.mean(precision, dim=0)
        individual_weighted_losses = torch.mean(weighted_losses, dim=0)
        
        for i in range(len(losses)):
            name = task_names[i] if i < len(task_names) else f'task_{i}'
            weights[f'{name}_weight'] = individual_weights[i].item()
            weights[f'{name}_weighted'] = individual_weighted_losses[i].item()
        
        return total_loss, weights


# ============================================================================
# Type 4: True Instance-wise Uncertainty (需要per-instance损失)
# ============================================================================
class TrueInstanceWiseUncertainty(nn.Module):
    """
    真正的实例级不确定性加权
    要求输入的损失必须是per-instance的（未在batch上平均）
    """
    
    def __init__(self, feature_dim=4096, num_tasks=2, hidden_dim=128):
        super().__init__()
        self.num_tasks = num_tasks
        self.uncertainty_predictor = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_tasks)
        )
    
    def forward(self, *losses, teacher_features=None):
        """
        Args:
            *losses: num_tasks个 (batch_size,) 的tensor（per-instance损失）
                    注意：如果传入标量损失，会自动扩展为batch维度
            teacher_features: (batch_size, feature_dim) 或 (batch_size, seq_len, feature_dim)
        Returns:
            total_loss: 标量
            weights: dict
        """
        if teacher_features is None:
            return EqualWeighting().forward(*losses)
        
        # 处理维度
        if teacher_features.dim() == 3:
            teacher_features = teacher_features.mean(dim=1)
        
        batch_size = teacher_features.shape[0]
        
        # 预测log_vars并限制在合理范围内，避免数值溢出
        # log_vars ∈ [-5, 5] 意味着 precision ∈ [e^-5, e^5] ≈ [0.0067, 148.4]
        # 将teacher_features转换为float32以匹配MLP的dtype
        teacher_features_fp32 = teacher_features.detach().float()
        log_vars = self.uncertainty_predictor(teacher_features_fp32)
        log_vars = torch.clamp(log_vars, min=-5.0, max=5.0)
        
        # 检查并处理损失维度
        # 如果损失是标量，则扩展为batch维度
        processed_losses = []
        for loss in losses:
            if loss.dim() == 0:
                # 标量损失，扩展为batch维度
                loss_batch = loss.unsqueeze(0).repeat(batch_size)
            else:
                # 已经是per-instance损失
                loss_batch = loss
            processed_losses.append(loss_batch)
        
        # Stack per-instance losses
        losses_tensor = torch.stack(processed_losses, dim=1)
        
        # 计算加权损失
        precision = torch.exp(-log_vars)
        weighted_losses = precision * losses_tensor + log_vars
        total_loss = torch.mean(torch.sum(weighted_losses, dim=1))
        
        # 构建权重字典
        task_names = ['main_loss', 'logits_distill_loss', 'v_loss', 'attn_loss']
        weights = {}
        individual_weights = torch.mean(precision, dim=0)
        individual_weighted_losses = torch.mean(weighted_losses, dim=0)
        
        for i in range(len(losses)):
            name = task_names[i] if i < len(task_names) else f'task_{i}'
            weights[f'{name}_weight'] = individual_weights[i].item()
            weights[f'{name}_weighted'] = individual_weighted_losses[i].item()
        
        return total_loss, weights


# ============================================================================
# 工厂函数：创建权重策略
# ============================================================================
def create_weighting_strategy(strategy_type, **kwargs):
    """
    创建损失加权策略
    
    Args:
        strategy_type: 策略类型
            - "type1" 或 "equal": 等权重
            - "type2": 异方差不确定性
            - "type3" 或 "uncertainty": 实例条件权重
            - "type4": 真实实例级不确定性
        **kwargs: 传递给策略构造函数的参数
    
    Returns:
        WeightingStrategy: 权重策略模块
    """
    # 向后兼容旧名称
    strategy_map = {
        'type1': EqualWeighting,
        'equal': EqualWeighting,
        'type2': HeteroscedasticUncertainty,
        'type3': InstanceConditionalWeighting,
        'uncertainty': InstanceConditionalWeighting,
        'type4': TrueInstanceWiseUncertainty,
    }
    
    strategy_type = strategy_type.lower()
    if strategy_type not in strategy_map:
        print(f"Unknown strategy type: {strategy_type}, falling back to type1 (equal)")
        strategy_type = 'type1'
    
    strategy_class = strategy_map[strategy_type]
    
    # 根据不同策略传递不同参数
    if strategy_class == EqualWeighting:
        return strategy_class()
    elif strategy_class == HeteroscedasticUncertainty:
        num_tasks = kwargs.get('num_tasks', 2)
        return strategy_class(num_tasks=num_tasks)
    else:  # Type 3 or Type 4
        feature_dim = kwargs.get('feature_dim', 4096)
        num_tasks = kwargs.get('num_tasks', 2)
        hidden_dim = kwargs.get('hidden_dim', 128)
        return strategy_class(feature_dim=feature_dim, num_tasks=num_tasks, hidden_dim=hidden_dim)