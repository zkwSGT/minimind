import os  # 操作系统接口
import sys  # 系统接口
__package__ = "trainer"  # 指定包名
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))  # 添加上级目录到路径

import argparse  # 解析命令行参数
import time  # 计时
import math  # 数学运算
import warnings  # 警告过滤
import torch  # PyTorch
import torch.distributed as dist  # 分布式训练
from torch import optim, nn  # 优化器和神经网络模块
from torch.nn.parallel import DistributedDataParallel  # 分布式封装
from torch.utils.data import DataLoader, DistributedSampler  # 数据加载相关
from contextlib import nullcontext  # 上下文管理器
from transformers import AutoTokenizer  # Tokenizer 加载
from model.model_minimind import MiniMindConfig, MiniMindForCausalLM  # 模型定义
from dataset.lm_dataset import PretrainDataset  # 预训练数据集

warnings.filterwarnings('ignore')  # 关闭警告信息


def Logger(content):  # 简易日志函数
    """在分布式环境下，仅主进程输出日志。"""
    if not ddp or dist.get_rank() == 0:  # 仅主进程打印
        print(content)


def get_lr(current_step, total_steps, lr):  # 计算学习率
    """使用余弦退火策略计算学习率。"""
    return lr / 10 + 0.5 * lr * (1 + math.cos(math.pi * current_step / total_steps))  # 余弦退火


def train_epoch(epoch, wandb):
    """执行单个训练周期。"""
    loss_fct = nn.CrossEntropyLoss(reduction='none')  # 损失函数
    start_time = time.time()  # 记录时间
    for step, (X, Y, loss_mask) in enumerate(train_loader):  # 遍历数据
        # 将数据移动到目标设备
        X = X.to(args.device)
        Y = Y.to(args.device)
        loss_mask = loss_mask.to(args.device)

        lr = get_lr(epoch * iter_per_epoch + step, args.epochs * iter_per_epoch, args.learning_rate)  # 当前学习率
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr  # 更新学习率

        with ctx:  # 混合精度上下文
            res = model(X)
            loss = loss_fct(
                res.logits.view(-1, res.logits.size(-1)),
                Y.view(-1)
            ).view(Y.size())
            loss = (loss * loss_mask).sum() / loss_mask.sum()  # 加权平均损失
            loss += res.aux_loss  # 加上辅助损失
            loss = loss / args.accumulation_steps  # 梯度累积平均

        # 反向传播，梯度累积
        scaler.scale(loss).backward()  # 反向传播

        if (step + 1) % args.accumulation_steps == 0:
            scaler.unscale_(optimizer)  # 取消缩放
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)  # 梯度裁剪

            scaler.step(optimizer)  # 更新参数
            scaler.update()  # 更新比例因子

            optimizer.zero_grad(set_to_none=True)  # 清空梯度

        # 定期输出日志并记录到wandb
        if step % args.log_interval == 0:  # 日志输出
            spend_time = time.time() - start_time
            Logger(
                'Epoch:[{}/{}]({}/{}) loss:{:.3f} lr:{:.12f} epoch_Time:{}min:'.format(
                    epoch + 1,
                    args.epochs,
                    step,
                    iter_per_epoch,
                    loss.item() * args.accumulation_steps,
                    optimizer.param_groups[-1]['lr'],
                    spend_time / (step + 1) * iter_per_epoch // 60 - spend_time // 60))

            if (wandb is not None) and (not ddp or dist.get_rank() == 0):
                wandb.log({"loss": loss.item() * args.accumulation_steps,
                           "lr": optimizer.param_groups[-1]['lr'],
                           "epoch_Time": spend_time / (step + 1) * iter_per_epoch // 60 - spend_time // 60})

        if (step + 1) % args.save_interval == 0 and (not ddp or dist.get_rank() == 0):  # 定期保存
            model.eval()
            moe_path = '_moe' if lm_config.use_moe else ''
            ckp = f'{args.save_dir}/pretrain_{lm_config.hidden_size}{moe_path}.pth'

            if isinstance(model, torch.nn.parallel.DistributedDataParallel):
                state_dict = model.module.state_dict()  # 获取实际模型
            else:
                state_dict = model.state_dict()

            state_dict = {k: v.half() for k, v in state_dict.items()}  # 半精度保存
            torch.save(state_dict, ckp)  # 保存检查点
            model.train()  # 恢复训练模式


def init_model(lm_config):
    """初始化模型与分词器，并打印参数规模。"""
    tokenizer = AutoTokenizer.from_pretrained('../model/')
    model = MiniMindForCausalLM(lm_config).to(args.device)
    Logger(f'LLM可训练总参数量：{sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.3f} 百万')
    return model, tokenizer


def init_distributed_mode():
    if not ddp: return
    """分布式训练的初始化，设置通信环境与device。"""
    global ddp_local_rank, DEVICE

    dist.init_process_group(backend="nccl")
    ddp_rank = int(os.environ["RANK"])
    ddp_local_rank = int(os.environ["LOCAL_RANK"])
    ddp_world_size = int(os.environ["WORLD_SIZE"])
    DEVICE = f"cuda:{ddp_local_rank}"
    torch.cuda.set_device(DEVICE)


# torchrun --nproc_per_node 2 1-pretrain.py
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MiniMind Pretraining")
    parser.add_argument("--out_dir", type=str, default="../out")
    # 若要以最快速度实现zero则epochs设置为1轮；否则应当利用有限的数据训练2~6个epochs。
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=5e-4)
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="MiniMind-Pretrain")
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--ddp", action="store_true")
    parser.add_argument("--accumulation_steps", type=int, default=8)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--warmup_iters", type=int, default=0)
    parser.add_argument("--log_interval", type=int, default=100)
    parser.add_argument("--save_interval", type=int, default=100)
    parser.add_argument('--local_rank', type=int, default=-1)
    parser.add_argument('--hidden_size', default=512, type=int)
    parser.add_argument('--num_hidden_layers', default=8, type=int)
    parser.add_argument('--max_seq_len', default=512, type=int)
    parser.add_argument('--use_moe', default=False, type=bool)
    parser.add_argument("--data_path", type=str, default="../dataset/pretrain_hq.jsonl")
    args = parser.parse_args()

    lm_config = MiniMindConfig(hidden_size=args.hidden_size, num_hidden_layers=args.num_hidden_layers, use_moe=args.use_moe)  # 构建配置
    args.save_dir = os.path.join(args.out_dir)
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.out_dir, exist_ok=True)
    tokens_per_iter = args.batch_size * args.max_seq_len  # 每次迭代的token数
    device_type = "cuda" if "cuda" in args.device else "cpu"  # 判断设备类型

    args.wandb_run_name = f"MiniMind-Pretrain-Epoch-{args.epochs}-BatchSize-{args.batch_size}-LearningRate-{args.learning_rate}"

    ctx = nullcontext() if device_type == "cpu" else torch.cuda.amp.autocast()  # 自动混合精度

    ddp = int(os.environ.get("RANK", -1)) != -1  # 是否为分布式运行
    ddp_local_rank, DEVICE = 0, "cuda:0"

    base_seed = 1337  # 随机种子
    torch.manual_seed(base_seed)
    torch.cuda.manual_seed(base_seed)

    if ddp:  # 分布式训练下的设置
        init_distributed_mode()
        args.device = torch.device(DEVICE)
        rank = dist.get_rank()
        torch.manual_seed(base_seed + rank)
        # 同时设置 CUDA 的随机种子
        torch.cuda.manual_seed(base_seed + rank)

    if args.use_wandb and (not ddp or ddp_local_rank == 0):  # 初始化 wandb
        import wandb

        wandb.init(project=args.wandb_project, name=args.wandb_run_name)
    else:
        wandb = None

    model, tokenizer = init_model(lm_config)
    train_ds = PretrainDataset(args.data_path, tokenizer, max_length=args.max_seq_len)  # 构建数据集
    train_sampler = DistributedSampler(train_ds) if ddp else None  # 分布式采样器
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        pin_memory=True,
        drop_last=False,
        shuffle=False,
        num_workers=args.num_workers,
        sampler=train_sampler
    )  # 数据加载器

    scaler = torch.cuda.amp.GradScaler(enabled=(args.dtype in ['float16', 'bfloat16']))  # AMP 缩放器
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)  # 优化器

    if ddp:
        model._ddp_params_and_buffers_to_ignore = {"pos_cis"}
        model = DistributedDataParallel(model, device_ids=[ddp_local_rank])  # 包装 DDP

    iter_per_epoch = len(train_loader)  # 每个epoch的迭代次数
    for epoch in range(args.epochs):  # 训练多个epoch
        train_epoch(epoch, wandb)
