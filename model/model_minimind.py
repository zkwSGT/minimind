# 📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘
#                                             MiniMind Config
# 📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘

from transformers import PretrainedConfig  # transformers 中的配置基类


class MiniMindConfig(PretrainedConfig):  # 模型配置定义
    """MiniMind 模型的配置类，定义了网络结构的超参数。"""

    model_type = "minimind"  # 用于加载权重时的标识

    def __init__(
            self,
            dropout: float = 0.0,  # dropout概率
            bos_token_id: int = 1,  # 开始符号ID
            eos_token_id: int = 2,  # 结束符号ID
            hidden_act: str = 'silu',  # 激活函数
            hidden_size: int = 512,  # 隐藏层维度
            intermediate_size: int = None,  # 前馈层维度
            max_position_embeddings: int = 32768,  # 最大位置
            num_attention_heads: int = 8,  # 注意力头数
            num_hidden_layers: int = 8,  # 层数
            num_key_value_heads: int = 2,  # kv头数
            vocab_size: int = 6400,  # 词表大小
            rms_norm_eps: float = 1e-05,  # RMSNorm eps
            rope_theta: int = 1000000.0,  # 旋转位置编码theta
            flash_attn: bool = True,  # 是否使用FlashAttention
            ####################################################
            # Here are the specific configurations of MOE
            # When use_moe is false, the following is invalid
            ####################################################
            use_moe: bool = False,  # 是否启用MoE
            num_experts_per_tok: int = 2,  # 每个token选择的专家数
            n_routed_experts: int = 4,  # 路由专家数量
            n_shared_experts: int = 1,  # 共享专家数量
            scoring_func: str = 'softmax',  # 打分函数
            aux_loss_alpha: float = 0.1,  # 辅助损失系数
            seq_aux: bool = True,  # 是否对序列级别求aux loss
            norm_topk_prob: bool = True,  # 是否归一化top-k概率
            **kwargs
    ):
        super().__init__(**kwargs)  # 调用父类初始化
        self.dropout = dropout  # dropout 概率
        self.bos_token_id = bos_token_id  # 开始 token
        self.eos_token_id = eos_token_id  # 结束 token
        self.hidden_act = hidden_act  # 激活函数名称
        self.hidden_size = hidden_size  # 隐藏层维度
        self.intermediate_size = intermediate_size  # 前馈层维度
        self.max_position_embeddings = max_position_embeddings  # 最大位置
        self.num_attention_heads = num_attention_heads  # 注意力头数
        self.num_hidden_layers = num_hidden_layers  # Transformer 层数
        self.num_key_value_heads = num_key_value_heads  # KV 头数
        self.vocab_size = vocab_size  # 词表大小
        self.rms_norm_eps = rms_norm_eps  # RMSNorm eps
        self.rope_theta = rope_theta  # 旋转位置编码 theta
        self.flash_attn = flash_attn  # 是否使用 FlashAttention
        ####################################################
        # Here are the specific configurations of MOE
        # When use_moe is false, the following is invalid
        ####################################################
        self.use_moe = use_moe  # 是否使用 MoE
        self.num_experts_per_tok = num_experts_per_tok  # 每个token选择的专家数量
        self.n_routed_experts = n_routed_experts  # 总的专家数量
        self.n_shared_experts = n_shared_experts  # 共享专家
        self.scoring_func = scoring_func  # 评分函数，默认为'softmax'
        self.aux_loss_alpha = aux_loss_alpha  # 辅助损失的alpha参数
        self.seq_aux = seq_aux  # 是否在序列级别上计算辅助损失
        self.norm_topk_prob = norm_topk_prob  # 是否标准化top-k概率


# 📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘
#                                             MiniMind Model
# 📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘

import math  # 数学运算库
import torch  # PyTorch 主库
from torch import nn  # 神经网络模块
from transformers.activations import ACT2FN  # 激活函数映射表
from typing import Optional, Tuple, List, Union  # 类型注解
import torch.nn.functional as F  # 常用函数集合
from transformers import PreTrainedModel, GenerationMixin, PretrainedConfig  # 预训练模型基类等
from transformers.modeling_outputs import CausalLMOutputWithPast  # 输出数据结构


class RMSNorm(torch.nn.Module):  # RMSNorm 实现
    """RMSNorm 层，效果类似 LayerNorm，但计算更简洁。"""
    def __init__(self, dim: int, eps: float = 1e-5):  # dim 表示特征维度
        super().__init__()  # 调用父类初始化
        self.eps = eps  # 防止除零
        self.weight = nn.Parameter(torch.ones(dim))  # 可训练缩放参数

    def _norm(self, x):  # 计算均方根归一化
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)  # 归一化公式

    def forward(self, x):  # 前向传播
        return self.weight * self._norm(x.float()).type_as(x)  # 应用权重


def precompute_freqs_cis(dim: int, end: int = int(32 * 1024), theta: float = 1e6):  # 预计算旋转位置编码参数
    """预先计算旋转位置编码所需的余弦和正弦值。"""
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))  # 角频率
    t = torch.arange(end, device=freqs.device)  # 位置索引
    freqs = torch.outer(t, freqs).float()  # 外积得到角度矩阵
    freqs_cos = torch.cat([torch.cos(freqs), torch.cos(freqs)], dim=-1)  # cos 部分
    freqs_sin = torch.cat([torch.sin(freqs), torch.sin(freqs)], dim=-1)  # sin 部分
    return freqs_cos, freqs_sin  # 返回两者


def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):  # 应用旋转位置编码
    """将旋转位置编码应用到 q、k 上。"""
    def rotate_half(x):  # 交换向量前后两半
        return torch.cat((-x[..., x.shape[-1] // 2:], x[..., : x.shape[-1] // 2]), dim=-1)

    q_embed = (q * cos.unsqueeze(unsqueeze_dim)) + (rotate_half(q) * sin.unsqueeze(unsqueeze_dim))
    k_embed = (k * cos.unsqueeze(unsqueeze_dim)) + (rotate_half(k) * sin.unsqueeze(unsqueeze_dim))
    return q_embed, k_embed  # 返回旋转后的q,k


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:  # 扩展 KV 头数
    """类似 torch.repeat_interleave 的实现"""
    bs, slen, num_key_value_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    return (
        x[:, :, :, None, :]
        .expand(bs, slen, num_key_value_heads, n_rep, head_dim)
        .reshape(bs, slen, num_key_value_heads * n_rep, head_dim)
    )  # 在head维度复制


class Attention(nn.Module):  # 自注意力层
    """多头自注意力模块，支持 FlashAttention 与 KV 缓存。"""

    def __init__(self, args: MiniMindConfig):  # 初始化注意力层
        super().__init__()  # 调用父类初始化
        self.num_key_value_heads = args.num_attention_heads if args.num_key_value_heads is None else args.num_key_value_heads  # kv头数
        assert args.num_attention_heads % self.num_key_value_heads == 0  # 保证整除
        self.n_local_heads = args.num_attention_heads  # 本地头数
        self.n_local_kv_heads = self.num_key_value_heads
        self.n_rep = self.n_local_heads // self.n_local_kv_heads  # 每个kv对应多少q
        self.head_dim = args.hidden_size // args.num_attention_heads
        self.q_proj = nn.Linear(args.hidden_size, args.num_attention_heads * self.head_dim, bias=False)  # q线性层
        self.k_proj = nn.Linear(args.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)  # k线性层
        self.v_proj = nn.Linear(args.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)  # v线性层
        self.o_proj = nn.Linear(args.num_attention_heads * self.head_dim, args.hidden_size, bias=False)  # 输出线性层
        self.attn_dropout = nn.Dropout(args.dropout)  # 注意力层dropout
        self.resid_dropout = nn.Dropout(args.dropout)  # 残差连接dropout
        self.dropout = args.dropout  # dropout概率
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention') and args.flash_attn  # 是否使用FlashAttention
        # print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")

    def forward(self,
                x: torch.Tensor,
                position_embeddings: Tuple[torch.Tensor, torch.Tensor],  # 修改为接收cos和sin
                past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
                use_cache=False,
                attention_mask: Optional[torch.Tensor] = None):
        bsz, seq_len, _ = x.shape  # 批大小与序列长度
        xq, xk, xv = self.q_proj(x), self.k_proj(x), self.v_proj(x)  # 线性映射得到qkv
        xq = xq.view(bsz, seq_len, self.n_local_heads, self.head_dim)  # 重塑q形状
        xk = xk.view(bsz, seq_len, self.n_local_kv_heads, self.head_dim)  # 重塑k形状
        xv = xv.view(bsz, seq_len, self.n_local_kv_heads, self.head_dim)  # 重塑v形状

        cos, sin = position_embeddings
        xq, xk = apply_rotary_pos_emb(xq, xk, cos[:seq_len], sin[:seq_len])  # 加入旋转位置编码

        # 如果提供了past_key_value，则拼接历史kv实现kv cache
        if past_key_value is not None:
            xk = torch.cat([past_key_value[0], xk], dim=1)
            xv = torch.cat([past_key_value[1], xv], dim=1)
        past_kv = (xk, xv) if use_cache else None  # 根据需要缓存kv

        xq, xk, xv = (
            xq.transpose(1, 2),  # 将q调换维度
            repeat_kv(xk, self.n_rep).transpose(1, 2),  # 扩展并调换k
            repeat_kv(xv, self.n_rep).transpose(1, 2)  # 扩展并调换v
        )  # 变换形状以便计算

        if self.flash and seq_len != 1:  # 使用FlashAttention加速
            dropout_p = self.dropout if self.training else 0.0  # 训练时启用dropout
            attn_mask = None  # 默认无mask
            if attention_mask is not None:
                attn_mask = attention_mask.view(bsz, 1, 1, -1).expand(bsz, self.n_local_heads, seq_len, -1)
                attn_mask = attn_mask.bool() if attention_mask is not None else None

            output = F.scaled_dot_product_attention(xq, xk, xv, attn_mask=attn_mask, dropout_p=dropout_p, is_causal=True)  # 调用FlashAttention
        else:  # 普通注意力实现
            scores = (xq @ xk.transpose(-2, -1)) / math.sqrt(self.head_dim)  # 点积并缩放
            scores = scores + torch.triu(
                torch.full((seq_len, seq_len), float("-inf"), device=scores.device),
                diagonal=1
            ).unsqueeze(0).unsqueeze(0)  # 下三角mask

            if attention_mask is not None:
                extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)  # 扩展维度
                extended_attention_mask = (1.0 - extended_attention_mask) * -1e9  # mask掉无效位置
                scores = scores + extended_attention_mask  # 添加到分数中

            scores = F.softmax(scores.float(), dim=-1).type_as(xq)  # softmax 得到权重
            scores = self.attn_dropout(scores)  # dropout
            output = scores @ xv  # 加权求和得到输出

        output = output.transpose(1, 2).reshape(bsz, seq_len, -1)  # 恢复形状
        output = self.resid_dropout(self.o_proj(output))  # 输出投影
        return output, past_kv  # 返回结果及缓存


class FeedForward(nn.Module):  # 前馈层模块
    """前馈网络，由两层线性层和激活函数组成。"""

    def __init__(self, config: MiniMindConfig):  # 构建前馈层
        super().__init__()  # 调用父类初始化
        if config.intermediate_size is None:
            intermediate_size = int(config.hidden_size * 8 / 3)
            config.intermediate_size = 64 * ((intermediate_size + 64 - 1) // 64)
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)  # 第一线性层
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)  # 输出层
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)  # 第二线性层
        self.dropout = nn.Dropout(config.dropout)  # dropout层
        self.act_fn = ACT2FN[config.hidden_act]  # 激活函数

    def forward(self, x):  # 前向
        return self.dropout(self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x)))


class MoEGate(nn.Module):  # MoE 门控模块
    """MoE门控，负责选择每个token要路由到哪些专家。"""
    def __init__(self, config: MiniMindConfig):  # 初始化门控
        super().__init__()  # 调用父类初始化
        self.config = config  # 保存配置
        self.top_k = config.num_experts_per_tok  # 每个 token 选取的专家数
        self.n_routed_experts = config.n_routed_experts  # 路由专家数

        self.scoring_func = config.scoring_func  # 评分函数
        self.alpha = config.aux_loss_alpha  # 辅助损失系数
        self.seq_aux = config.seq_aux  # 是否计算序列级辅助损失

        self.norm_topk_prob = config.norm_topk_prob  # 是否归一化概率
        self.gating_dim = config.hidden_size  # 输入维度
        self.weight = nn.Parameter(torch.empty((self.n_routed_experts, self.gating_dim)))  # 专家权重
        self.reset_parameters()

    def reset_parameters(self) -> None:  # 初始化权重
        import torch.nn.init as init
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, hidden_states):
        bsz, seq_len, h = hidden_states.shape  # 批大小、序列长度和隐藏维度
        hidden_states = hidden_states.view(-1, h)  # 展平成二维
        logits = F.linear(hidden_states, self.weight, None)  # 计算得分
        if self.scoring_func == 'softmax':  # 使用 softmax 计算概率
            scores = logits.softmax(dim=-1)
        else:
            raise NotImplementedError(f'insupportable scoring function for MoE gating: {self.scoring_func}')

        topk_weight, topk_idx = torch.topk(scores, k=self.top_k, dim=-1, sorted=False)  # 选择概率最高的专家

        if self.top_k > 1 and self.norm_topk_prob:  # 归一化 top-k 概率
            denominator = topk_weight.sum(dim=-1, keepdim=True) + 1e-20
            topk_weight = topk_weight / denominator

        if self.training and self.alpha > 0.0:  # 训练阶段计算辅助loss
            scores_for_aux = scores  # 用于计算辅助损失的分数
            aux_topk = self.top_k  # top-k 数量
            topk_idx_for_aux_loss = topk_idx.view(bsz, -1)  # 展平索引
            if self.seq_aux:
                scores_for_seq_aux = scores_for_aux.view(bsz, seq_len, -1)
                ce = torch.zeros(bsz, self.n_routed_experts, device=hidden_states.device)
                ce.scatter_add_(1, topk_idx_for_aux_loss,
                                torch.ones(bsz, seq_len * aux_topk, device=hidden_states.device)).div_(
                    seq_len * aux_topk / self.n_routed_experts)
                aux_loss = (ce * scores_for_seq_aux.mean(dim=1)).sum(dim=1).mean() * self.alpha
            else:
                mask_ce = F.one_hot(topk_idx_for_aux_loss.view(-1), num_classes=self.n_routed_experts)  # one-hot
                ce = mask_ce.float().mean(0)  # 平均负载
                Pi = scores_for_aux.mean(0)  # 平均概率
                fi = ce * self.n_routed_experts  # token占比
                aux_loss = (Pi * fi).sum() * self.alpha  # 计算损失
        else:
            aux_loss = 0
        return topk_idx, topk_weight, aux_loss  # 返回路由结果与辅助损失


class MOEFeedForward(nn.Module):  # 含专家的前馈层
    """门控专家前馈模块，训练时路由到多个专家。"""
    def __init__(self, config: MiniMindConfig):  # 构建包含多个专家的前馈层
        super().__init__()  # 调用父类初始化
        self.config = config  # 模型配置
        self.experts = nn.ModuleList([
            FeedForward(config)
            for _ in range(config.n_routed_experts)
        ])  # 专家列表
        self.gate = MoEGate(config)  # 门控
        if config.n_shared_experts > 0:  # 是否存在共享专家
            self.shared_experts = nn.ModuleList([
                FeedForward(config)
                for _ in range(config.n_shared_experts)
            ])

    def forward(self, x):  # 前向执行MoE
        identity = x  # 残差分支
        orig_shape = x.shape  # 记录原始形状
        bsz, seq_len, _ = x.shape  # 获取维度
        topk_idx, topk_weight, aux_loss = self.gate(x)  # 选择专家
        x = x.view(-1, x.shape[-1])  # 展平成二维
        flat_topk_idx = topk_idx.view(-1)  # 展平特征索引
        if self.training:
            x = x.repeat_interleave(self.config.num_experts_per_tok, dim=0)  # 重复token
            y = torch.empty_like(x, dtype=torch.float16)  # 缓存输出
            for i, expert in enumerate(self.experts):  # 遍历每个专家
                y[flat_topk_idx == i] = expert(x[flat_topk_idx == i]).to(y.dtype)  # 确保类型一致
            y = (y.view(*topk_weight.shape, -1) * topk_weight.unsqueeze(-1)).sum(dim=1)  # 加权合并
            y = y.view(*orig_shape)  # 恢复形状
        else:
            y = self.moe_infer(x, flat_topk_idx, topk_weight.view(-1, 1)).view(*orig_shape)  # 推理阶段合并
        if self.config.n_shared_experts > 0:  # 计算共享专家输出
            for expert in self.shared_experts:
                y = y + expert(identity)
        self.aux_loss = aux_loss  # 保存辅助损失
        return y

    @torch.no_grad()
    def moe_infer(self, x, flat_expert_indices, flat_expert_weights):  # 推理时的专家选择
        """推理阶段的MoE路由实现，避免循环带来的开销。"""
        expert_cache = torch.zeros_like(x)  # 存放各专家计算结果
        idxs = flat_expert_indices.argsort()  # 排序以便按专家分组
        tokens_per_expert = flat_expert_indices.bincount().cpu().numpy().cumsum(0)  # 每个专家处理的token数量
        token_idxs = idxs // self.config.num_experts_per_tok  # 获得token索引
        # 当tokens_per_expert = [6, 15, 20, 26]，tokens_per_expert.shape[0]即为专家数量（此时为4）
        # 且token_idxs = [3, 7, 19, 21, 24, 25,  4,  5,  6, 10, 11, 12...] 时
        # 意味token_idxs[:6] -> [3, 7, 19, 21, 24, 25]这6个位置属于专家0处理的token（每个token有可能被多个专家处理，这取决于num_experts_per_tok）
        # 接下来9个位置token_idxs[6:15] -> [4,  5,  6, 10, 11, 12...]属于专家1处理的token...依此类推
        for i, end_idx in enumerate(tokens_per_expert):  # 遍历专家
            start_idx = 0 if i == 0 else tokens_per_expert[i - 1]
            if start_idx == end_idx:
                continue  # 没有分配token
            expert = self.experts[i]
            exp_token_idx = token_idxs[start_idx:end_idx]  # 当前专家处理的token索引
            expert_tokens = x[exp_token_idx]
            expert_out = expert(expert_tokens).to(expert_cache.dtype)
            expert_out.mul_(flat_expert_weights[idxs[start_idx:end_idx]])  # 乘以权重
            expert_cache.scatter_add_(0, exp_token_idx.view(-1, 1).repeat(1, x.shape[-1]), expert_out)

        return expert_cache  # 返回聚合后的结果


class MiniMindBlock(nn.Module):  # Transformer 基础块
    """Transformer 的基础 Block，包含自注意力和前馈。"""

    def __init__(self, layer_id: int, config: MiniMindConfig):  # 初始化一个Transformer Block
        super().__init__()  # 调用父类初始化
        self.num_attention_heads = config.num_attention_heads  # 头数
        self.hidden_size = config.hidden_size  # 隐藏维度
        self.head_dim = config.hidden_size // config.num_attention_heads  # 单头维度
        self.self_attn = Attention(config)  # 自注意力层

        self.layer_id = layer_id  # 层序号
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)  # 输入层归一化
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)  # 注意力后归一化
        self.mlp = FeedForward(config) if not config.use_moe else MOEFeedForward(config)  # 前馈或MoE

    def forward(self, hidden_states, position_embeddings, past_key_value=None, use_cache=False, attention_mask=None):  # 前向传播
        residual = hidden_states  # 残差
        hidden_states, present_key_value = self.self_attn(
            self.input_layernorm(hidden_states), position_embeddings,
            past_key_value, use_cache, attention_mask
        )  # 自注意力
        hidden_states += residual  # 加回残差
        hidden_states = hidden_states + self.mlp(self.post_attention_layernorm(hidden_states))  # 前馈网络
        return hidden_states, present_key_value


class MiniMindModel(nn.Module):  # 解码器模型
    """MiniMind 主模型，堆叠多个 Block 形成解码器结构。"""

    def __init__(self, config: MiniMindConfig):  # 初始化主模型
        super().__init__()  # 调用父类初始化
        self.config = config  # 保存配置
        self.vocab_size, self.num_hidden_layers = config.vocab_size, config.num_hidden_layers  # 基本参数
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)  # 词向量层
        self.dropout = nn.Dropout(config.dropout)  # 输入dropout
        self.layers = nn.ModuleList([MiniMindBlock(l, config) for l in range(self.num_hidden_layers)])  # 堆叠多个块
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)  # 最终归一化

        freqs_cos, freqs_sin = precompute_freqs_cis(dim=config.hidden_size // config.num_attention_heads,
                                                    end=config.max_position_embeddings, theta=config.rope_theta)
        self.register_buffer("freqs_cos", freqs_cos, persistent=False)  # 位置编码 cos
        self.register_buffer("freqs_sin", freqs_sin, persistent=False)  # 位置编码 sin

    def forward(self,
                input_ids: Optional[torch.Tensor] = None,
                attention_mask: Optional[torch.Tensor] = None,
                past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
                use_cache: bool = False,
                **kwargs):
        batch_size, seq_length = input_ids.shape  # 获取批量和长度
        past_key_values = past_key_values or [None] * len(self.layers)  # 处理缓存
        start_pos = past_key_values[0][0].shape[1] if past_key_values[0] is not None else 0  # 已生成长度

        hidden_states = self.dropout(self.embed_tokens(input_ids))  # 词嵌入并dropout

        position_embeddings = (
            self.freqs_cos[start_pos:start_pos + seq_length],
            self.freqs_sin[start_pos:start_pos + seq_length]
        )  # 取出对应位置编码

        presents = []  # 存储各层的kv缓存
        for layer_idx, (layer, past_key_value) in enumerate(zip(self.layers, past_key_values)):  # 遍历所有层
            hidden_states, present = layer(
                hidden_states,
                position_embeddings,
                past_key_value=past_key_value,
                use_cache=use_cache,
                attention_mask=attention_mask
            )
            presents.append(present)  # 记录缓存

        hidden_states = self.norm(hidden_states)  # 最终层归一化

        aux_loss = sum(
            layer.mlp.aux_loss
            for layer in self.layers
            if isinstance(layer.mlp, MOEFeedForward)
        )  # 累加所有块的MoE辅助损失

        return hidden_states, presents, aux_loss  # 返回输出与缓存


class MiniMindForCausalLM(PreTrainedModel, GenerationMixin):  # 带LM头的模型
    """带语言模型头的 MiniMind，用于文本生成任务。"""
    config_class = MiniMindConfig

    def __init__(self, config: MiniMindConfig = None):  # 初始化带语言模型头的模型
        self.config = config or MiniMindConfig()  # 模型配置
        super().__init__(self.config)
        self.model = MiniMindModel(self.config)  # 主模型
        self.lm_head = nn.Linear(self.config.hidden_size, self.config.vocab_size, bias=False)  # 语言模型头
        self.model.embed_tokens.weight = self.lm_head.weight  # 权重共享
        self.OUT = CausalLMOutputWithPast()  # 输出结构

    def forward(self,
                input_ids: Optional[torch.Tensor] = None,
                attention_mask: Optional[torch.Tensor] = None,
                past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
                use_cache: bool = False,
                logits_to_keep: Union[int, torch.Tensor] = 0,
                **args):
        h, past_kvs, aux_loss = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            **args
        )  # 调用底层模型
        # 只取最后几步的隐藏状态用于生成，提高效率
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep  # 取最后部分
        logits = self.lm_head(h[:, slice_indices, :])  # 计算logits
        self.OUT.__setitem__('last_hidden_state', h)
        self.OUT.__setitem__('logits', logits)
        self.OUT.__setitem__('aux_loss', aux_loss)
        self.OUT.__setitem__('past_key_values', past_kvs)
        return self.OUT  # 返回结构化输出
