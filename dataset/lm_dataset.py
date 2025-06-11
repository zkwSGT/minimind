import json  # 处理 JSON 数据
import random  # 随机相关操作
import re  # 正则表达式处理

import pandas as pd  # 数据分析库
import numpy as np  # 科学计算库
from torch.utils.data import Dataset, DataLoader  # 数据集与数据加载器
import torch  # PyTorch 框架
from sklearn.model_selection import train_test_split  # 数据集划分工具
import os  # 与操作系统交互
import ast  # 抽象语法树处理

os.environ["TOKENIZERS_PARALLELISM"] = "false"  # 禁用tokenizer多进程并行


class PretrainDataset(Dataset):
    """用于语言模型预训练的数据集，读取 jsonl 文件中每行的文本。"""

    def __init__(self, data_path, tokenizer, max_length=512):  # 初始化函数
        super().__init__()  # 调用父类构造
        self.tokenizer = tokenizer  # 保存分词器
        self.max_length = max_length  # 最大序列长度
        self.samples = self.load_data(data_path)  # 读取文件中的所有样本

    def load_data(self, path):  # 从文件加载数据
        samples = []  # 保存所有样本
        with open(path, 'r', encoding='utf-8') as f:  # 逐行读取
            for line_num, line in enumerate(f, 1):  # 枚举每行
                data = json.loads(line.strip())  # 解析 json
                samples.append(data)  # 收集样本
        return samples  # 返回结果

    def __len__(self):  # 数据集大小
        return len(self.samples)  # 样本数

    def __getitem__(self, index):  # 按索引获取样本
        sample = self.samples[index]  # 取出样本

        # 使用tokenizer将文本转为ID，并统一长度
        encoding = self.tokenizer(  # 调用分词器
            str(sample['text']),  # 待编码文本
            max_length=self.max_length,  # 最大长度
            padding='max_length',  # 不足补pad
            truncation=True,  # 过长截断
            return_tensors='pt'  # 返回tensor
        )
        input_ids = encoding.input_ids.squeeze()  # 获取id序列
        loss_mask = (input_ids != self.tokenizer.pad_token_id)  # padding部分不计入损失

        # 右移一位得到标签，左移一位作为输入
        X = torch.tensor(input_ids[:-1], dtype=torch.long)  # 模型输入
        Y = torch.tensor(input_ids[1:], dtype=torch.long)  # 预测目标
        loss_mask = torch.tensor(loss_mask[1:], dtype=torch.long)  # 对齐mask
        return X, Y, loss_mask


class SFTDataset(Dataset):
    """SFT监督微调阶段的数据集，输入为对话形式。"""

    def __init__(self, jsonl_path, tokenizer, max_length=1024):  # 初始化
        super().__init__()  # 调用父类构造
        self.tokenizer = tokenizer  # 保存分词器
        self.max_length = max_length  # 最大长度
        self.samples = self.load_data(jsonl_path)  # 加载数据集
        self.bos_id = tokenizer('<|im_start|>assistant', add_special_tokens=False).input_ids  # assistant起始标记
        self.eos_id = tokenizer('<|im_end|>', add_special_tokens=False).input_ids  # assistant结束标记

    def __len__(self):  # 返回样本数量
        return len(self.samples)

    def load_data(self, path):  # 加载jsonl数据
        samples = []  # 用于存储样本
        with open(path, 'r', encoding='utf-8') as f:  # 打开文件
            for line_num, line in enumerate(f, 1):  # 遍历每行
                data = json.loads(line.strip())  # 解析json
                samples.append(data)  # 收集样本
        return samples  # 返回结果

    def _create_chat_prompt(self, conversations):  # 构建对话提示
        """构建符合ChatML格式的对话"""
        messages = []  # 临时存储对话
        for i, turn in enumerate(conversations):  # 遍历轮次
            role = 'user' if i % 2 == 0 else 'assistant'  # 交替角色
            messages.append({"role": role, "content": turn['content']})  # 加入列表
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False
        )  # 返回渲染后的字符串

    def _generate_loss_mask(self, input_ids):  # 生成损失掩码
        loss_mask = [0] * len(input_ids)  # 默认全0
        i = 0  # 遍历位置
        while i < len(input_ids):
            if input_ids[i:i + len(self.bos_id)] == self.bos_id:  # 找到assistant开头
                start = i + len(self.bos_id)
                end = start
            while end < len(input_ids):  # 向后搜索结束标记
                if input_ids[end:end + len(self.eos_id)] == self.eos_id:
                    break
                end += 1
                for j in range(start + 1, min(end + len(self.eos_id) + 1, self.max_length)):
                    loss_mask[j] = 1  # 标记需要预测的位置
                i = end + len(self.eos_id) if end < len(input_ids) else len(input_ids)
            else:
                i += 1
        return loss_mask  # 返回掩码

    def __getitem__(self, index):  # 获取单条样本
        sample = self.samples[index]
        prompt = self._create_chat_prompt(sample['conversations'])  # 构建对话提示
        input_ids = self.tokenizer(prompt).input_ids[:self.max_length]  # 转为id
        input_ids += [self.tokenizer.pad_token_id] * (self.max_length - len(input_ids))  # 补pad

        loss_mask = self._generate_loss_mask(input_ids)  # 生成动态损失掩码

        X = torch.tensor(input_ids[:-1], dtype=torch.long)  # 输入序列
        Y = torch.tensor(input_ids[1:], dtype=torch.long)  # 标签序列
        loss_mask = torch.tensor(loss_mask[1:], dtype=torch.long)  # 对齐预测位置

        return X, Y, loss_mask  # 返回训练所需数据


class DPODataset(Dataset):
    """用于直接偏好优化（DPO）的数据集，包含正、负两种回答。"""
    def __init__(self, file_path, tokenizer, max_length=4096):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.padding = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
        self.bos_id = tokenizer('<|im_start|>assistant', add_special_tokens=False).input_ids
        self.eos_id = tokenizer('<|im_end|>', add_special_tokens=False).input_ids
        with open(file_path, 'r', encoding='utf-8') as f:
            self.data = []
            for line in f:
                line = line.strip()
                obj = json.loads(line)
                self.data.append(obj)

    def __len__(self):  # 数据集大小
        return len(self.data)

    def __getitem__(self, index):  # 获取对比样本
        item = self.data[index]
        chosen = item['chosen']  # 正样本对话
        rejected = item['rejected']  # 负样本对话
        chosen_prompt = self.tokenizer.apply_chat_template(
            chosen, tokenize=False, add_generation_prompt=False
        )  # 渲染正样本

        rejected_prompt = self.tokenizer.apply_chat_template(
            rejected, tokenize=False, add_generation_prompt=False
        )  # 渲染负样本
        chosen_encoding = self.tokenizer(
            chosen_prompt, truncation=True, max_length=self.max_length, padding='max_length'
        )  # 编码正样本
        rejected_encoding = self.tokenizer(
            rejected_prompt, truncation=True, max_length=self.max_length, padding='max_length'
        )  # 编码负样本

        chosen_input_ids = chosen_encoding['input_ids']  # 正样本ID序列
        chosen_loss_mask = self._generate_loss_mask(chosen_input_ids)  # 正样本mask

        rejected_input_ids = rejected_encoding['input_ids']  # 负样本ID序列
        rejected_loss_mask = self._generate_loss_mask(rejected_input_ids)  # 负样本mask
        x_chosen = torch.tensor(chosen_input_ids[:-1], dtype=torch.long)  # 输入
        y_chosen = torch.tensor(chosen_input_ids[1:], dtype=torch.long)  # 标签
        mask_chosen = torch.tensor(chosen_loss_mask[1:], dtype=torch.long)  # mask
        x_rejected = torch.tensor(rejected_input_ids[:-1], dtype=torch.long)
        y_rejected = torch.tensor(rejected_input_ids[1:], dtype=torch.long)
        mask_rejected = torch.tensor(rejected_loss_mask[1:], dtype=torch.long)

        return {
            'x_chosen': x_chosen,
            'y_chosen': y_chosen,
            'mask_chosen': mask_chosen,
            'x_rejected': x_rejected,
            'y_rejected': y_rejected,
            'mask_rejected': mask_rejected
        }  # 返回打包后的样本

    def _generate_loss_mask(self, input_ids):  # 计算正负样本的loss mask
        loss_mask = [0] * len(input_ids)
        i = 0
        while i < len(input_ids):
            if input_ids[i:i + len(self.bos_id)] == self.bos_id:
                start = i + len(self.bos_id)
                end = start
                while end < len(input_ids):
                    if input_ids[end:end + len(self.eos_id)] == self.eos_id:
                        break
                    end += 1
                for j in range(start + 1, min(end + len(self.eos_id) + 1, self.max_length)):
                    loss_mask[j] = 1
                i = end + len(self.eos_id) if end < len(input_ids) else len(input_ids)
            else:
                i += 1
        return loss_mask


class RLAIFDataset(Dataset):
    """用于强化学习与偏好数据构建的辅助数据集。"""
    def __init__(self, jsonl_path, tokenizer, max_length=1024):  # 初始化
        super().__init__()
        self.tokenizer = tokenizer  # 分词器
        self.max_length = max_length  # 最大长度
        self.samples = self.load_data(jsonl_path)  # 载入样本
        self.bos_id = tokenizer('<|im_start|>assistant', add_special_tokens=False).input_ids  # 开始标记
        self.eos_id = tokenizer('<|im_end|>', add_special_tokens=False).input_ids  # 结束标记

    def __len__(self):  # 样本数
        return len(self.samples)

    def load_data(self, path):  # 加载文件
        samples = []
        with open(path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                data = json.loads(line.strip())
                samples.append(data)
        return samples

    def _create_chat_prompt(self, conversations):  # 构建对话并返回答案
        """构建符合ChatML格式的对话"""
        messages = []
        answer = ''
        for i, turn in enumerate(conversations):
            role = 'user' if i % 2 == 0 else 'assistant'
            messages.append({"role": role, "content": turn['content']})
            answer = turn['content']
        return self.tokenizer.apply_chat_template(
            messages[:-1],
            tokenize=False,
            add_generation_prompt=True
        ), answer

    def __getitem__(self, index):  # 获取样本
        sample = self.samples[index]
        prompt, answer = self._create_chat_prompt(sample['conversations'])  # 构建对话提示

        return {
            'prompt': prompt,
            'answer': answer
        }  # 返回提示和答案


if __name__ == "__main__":
    pass
