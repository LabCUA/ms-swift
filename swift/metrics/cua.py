#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) ModelScope Contributors. All rights reserved.
"""
CUA（Computer Use Agent）专用评估指标。

支持：
1. 每一步 Action 类型准确率（按 Action Space 汇总 + 总体准确率）。
2. 在 Action 类型正确的前提下：
   - click/left_double/right_single/scroll：评估坐标误差；
   - drag：评估起点/终点坐标误差；
   - hotkey/type：评估字符串匹配准确率。

使用方式：
  在训练 / 评估命令中增加：
    --eval_metric cua
"""

from __future__ import annotations

import math
import re
from collections import defaultdict
from typing import Dict, Tuple, Optional

import json
import os
import numpy as np
import torch
from transformers.trainer_utils import EvalPrediction

from .base import EvalMetrics
from swift.utils import Serializer
from swift.utils import Serializer


_ACTION_NAME_RE = re.compile(r"^\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*\(")
# 坐标部分只要求能在整条 action 字符串中找到数字即可：
# - click/scroll 等：至少 2 个数字 -> (x, y)
# - drag：至少 4 个数字 -> (sx, sy, ex, ey)
# 不再强依赖特定的 point='<point>x y</point>' 或 start_box/end_box 格式，
# 以兼容 v1/v2 不同版本的 CUA 数据。
_NUM_RE = re.compile(r"-?\d+(?:\.\d+)?")
_TYPE_CONTENT_RE = re.compile(r"type\(\s*content='(.*?)'\s*\)")
_HOTKEY_RE = re.compile(r"hotkey\(\s*key='(.*?)'\s*\)")


def _extract_last_action_line(text: str) -> Optional[str]:
    """从完整输出中抽取最后一行 Action: ... 的内容。

    约定数据中格式为：
      Thought: ...
      Action: xxx
    """
    if not text:
        return None
    # 允许中间有多轮 Thought/Action，这里始终取最后一个 Action 段
    matches = list(re.finditer(r"Action:\s*(.+)", text))
    if not matches:
        return None
    line = matches[-1].group(1).strip()
    # 只取第一行，避免后面跟解释性文字
    return line.splitlines()[0].strip()


def _parse_action_name(action_str: str) -> Optional[str]:
    """解析函数名，例如 click(point='...') -> click。"""
    if not action_str:
        return None
    m = _ACTION_NAME_RE.match(action_str)
    if not m:
        return None
    return m.group(1)


def _parse_point(action_str: str) -> Optional[Tuple[int, int]]:
    """解析单点坐标（click/scroll 等）。

    规则：在整条 action 字符串中抽取所有数字（支持整数 / 小数 / 负数），
    至少找到 2 个时，将前两个作为 (x, y)。
    """
    nums = _NUM_RE.findall(action_str or "")
    if len(nums) < 2:
        return None
    x = float(nums[0])
    y = float(nums[1])
    return x, y


def _parse_drag_points(action_str: str) -> Tuple[Optional[Tuple[int, int]], Optional[Tuple[int, int]]]:
    """解析 drag 的起止点坐标。

    规则：在整条 action 字符串中抽取所有数字（支持整数 / 小数 / 负数），
    至少找到 4 个时，将前两个作为起点 (sx, sy)，后两个作为终点 (ex, ey)。
    """
    nums = _NUM_RE.findall(action_str or "")
    if len(nums) < 4:
        return None, None
    sx, sy, ex, ey = map(float, nums[:4])
    return (sx, sy), (ex, ey)


def _parse_type_content(action_str: str) -> Optional[str]:
    """解析 type(content='...') 中的 content 字符串（反转义由 tokenizer 负责，这里只做原始比较）。"""
    m = _TYPE_CONTENT_RE.search(action_str)
    if not m:
        return None
    return m.group(1)


def _parse_hotkey(action_str: str) -> Optional[str]:
    """解析 hotkey(key='...') 中的 key 字符串。"""
    m = _HOTKEY_RE.search(action_str)
    if not m:
        return None
    return m.group(1)


class CuaMetrics(EvalMetrics):
    """CUA 专用评估指标实现。"""

    def compute_metrics(self, eval_prediction: EvalPrediction) -> Dict[str, float]:
        """
        只在 predict_with_generate=True 场景下评估 CUA 指标。
        为了避免 DDP 下 eval_prediction.predictions/label_ids 聚合形态不一致的问题，
        这里仅用它来确定「本轮 eval 的样本数」，真实的文本来源是当前 run 下
        `output_dir/predict.jsonl` 的最后 N 行（N = len(eval_prediction.label_ids)）。
        """
        # 仅在自由生成（predict_with_generate）场景下才评估 CUA 指标
        if not getattr(self.args, "predict_with_generate", False):
            return {}

        _, labels = eval_prediction.predictions, eval_prediction.label_ids
        # 当前这轮 eval 的样本数（batch 维度）
        try:
            num_samples = labels.shape[0]
        except Exception:
            num_samples = len(labels)

        # 从当前 run 的 predict.jsonl 中读取最后 num_samples 行
        pred_path = os.path.join(self.args.output_dir, "predict.jsonl")
        if not os.path.isfile(pred_path):
            return {}
        with open(pred_path, "r", encoding="utf-8") as f:
            all_lines = [ln.strip() for ln in f if ln.strip()]

        if not all_lines:
            return {}

        if num_samples <= 0 or num_samples > len(all_lines):
            # 防御性处理：样本数异常时退化为使用全部
            effective_lines = all_lines
        else:
            effective_lines = all_lines[-num_samples:]

        # 统计量
        type_total = defaultdict(int)      # 每种 GT action type 出现次数
        type_correct = defaultdict(int)    # 每种 action type 预测正确次数

        coord_l2_sum = defaultdict(float)  # 每种 action type，在 type 正确前提下的坐标 L2 误差和
        coord_l1_sum = defaultdict(float)  # 同上，L1 误差和
        coord_count = defaultdict(int)     # 同上，计数

        str_total = defaultdict(int)       # hotkey/type 字符串的 GT 个数
        str_correct = defaultdict(int)     # hotkey/type 字符串完全匹配次数

        for line in effective_lines:
            try:
                obj = json.loads(line)
            except Exception:
                continue

            pred_text = str(obj.get("response", "") or "")
            label_text = str(obj.get("labels", "") or "")

            pred_action_str = _extract_last_action_line(pred_text)
            label_action_str = _extract_last_action_line(label_text)

            gt_type = _parse_action_name(label_action_str or "")
            pred_type = _parse_action_name(pred_action_str or "")

            if gt_type is None:
                # 数据异常，跳过该样本
                continue

            # 1) Action 类型准确率
            type_total[gt_type] += 1
            if pred_type == gt_type:
                type_correct[gt_type] += 1

            # 仅在类型正确时再做更细粒度的评估
            if pred_type != gt_type:
                continue

            # 2) 细粒度评估：坐标 / 文本
            if gt_type in {"click", "left_double", "right_single", "scroll"}:
                gt_pt = _parse_point(label_action_str or "")
                pr_pt = _parse_point(pred_action_str or "")
                if gt_pt is not None and pr_pt is not None:
                    dx = pr_pt[0] - gt_pt[0]
                    dy = pr_pt[1] - gt_pt[1]
                    l2 = math.sqrt(dx * dx + dy * dy)
                    l1 = abs(dx) + abs(dy)
                    coord_l2_sum[gt_type] += l2
                    coord_l1_sum[gt_type] += l1
                    coord_count[gt_type] += 1

            elif gt_type == "drag":
                gt_s, gt_e = _parse_drag_points(label_action_str or "")
                pr_s, pr_e = _parse_drag_points(pred_action_str or "")
                # 只在起点和终点都解析成功时统计
                if gt_s and gt_e and pr_s and pr_e:
                    def _dist(a, b):
                        dx = b[0] - a[0]
                        dy = b[1] - a[1]
                        l2_ = math.sqrt(dx * dx + dy * dy)
                        l1_ = abs(dx) + abs(dy)
                        return l2_, l1_

                    l2_s, l1_s = _dist(gt_s, pr_s)
                    l2_e, l1_e = _dist(gt_e, pr_e)
                    # 平均起点 & 终点误差
                    l2 = 0.5 * (l2_s + l2_e)
                    l1 = 0.5 * (l1_s + l1_e)
                    coord_l2_sum[gt_type] += l2
                    coord_l1_sum[gt_type] += l1
                    coord_count[gt_type] += 1

            elif gt_type == "type":
                gt_content = _parse_type_content(label_action_str or "") or ""
                pr_content = _parse_type_content(pred_action_str or "") or ""
                str_total["type"] += 1
                if pr_content == gt_content:
                    str_correct["type"] += 1

            elif gt_type == "hotkey":
                gt_key = _parse_hotkey(label_action_str or "") or ""
                pr_key = _parse_hotkey(pred_action_str or "") or ""
                str_total["hotkey"] += 1
                if pr_key == gt_key:
                    str_correct["hotkey"] += 1

            # 其他 action（如 wait()/finished()）目前仅做 type 正确率统计

        # 汇总指标
        metrics: Dict[str, float] = {}

        # 1) 每种 action 类型准确率 + 总体准确率
        total_all = sum(type_total.values())
        correct_all = sum(type_correct.values())
        if total_all > 0:
            metrics["cua_action_acc"] = correct_all / total_all

        for t in sorted(type_total.keys()):
            if type_total[t] > 0:
                metrics[f"cua_action_acc/{t}"] = type_correct[t] / type_total[t]

        # 2) 坐标误差（仅在 type 正确的样本上计算）
        for t in sorted(coord_count.keys()):
            if coord_count[t] > 0:
                metrics[f"cua_coord_l2_mean/{t}"] = coord_l2_sum[t] / coord_count[t]
                metrics[f"cua_coord_l1_mean/{t}"] = coord_l1_sum[t] / coord_count[t]

        # 3) 文本准确率（type / hotkey）
        for t in ["type", "hotkey"]:
            if str_total[t] > 0:
                metrics[f"cua_str_acc/{t}"] = str_correct[t] / str_total[t]

        return metrics

    def preprocess_logits_for_metrics(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """将 logits 转换为 token 序列，供 compute_metrics 解码使用。"""
        if isinstance(logits, (list, tuple)):
            logits = logits[0]
        preds = logits.argmax(dim=-1)
        return preds

