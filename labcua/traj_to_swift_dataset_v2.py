#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
轨迹数据 → ms-swift 多模态 SFT 格式转换脚本

用途：
  将 /data/250010149/hf_cache/traj 下的 CUA 轨迹数据转换为 ms-swift 框架
  微调 UI-TARS 1.5（CUA）模型所需的数据集格式，输出到 traj_v2。

参考：
  - ms-swift 自定义数据集：https://swift.readthedocs.io/zh-cn/latest/Customization/Custom-dataset.html
  - UI-TARS 1.5 agent 实现：/data/250010149/OSWorld/mm_agents/uitars15_v2.py

使用示例：
  python traj_to_swift_dataset.py --traj_dir /data/250010149/hf_cache/traj --out_dir /data/250010149/hf_cache/traj_v2
  python traj_to_swift_dataset.py --max_image_history_length 5 --no_use_thinking  # 与 agent 的 history_n / use_thinking 一致

微调时（ms-swift）：
  swift sft --dataset /data/250010149/hf_cache/traj_v2/train.jsonl ...  # 或 --custom_dataset_info traj_v2/dataset_info.json --dataset traj_cua_sft
"""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import Any

# -----------------------------------------------------------------------------
# 与 UI-TARS 1.5 agent 一致的 prompt 与参数（见 uitars15_v2.py）
# 保持与推理时一致，避免 train/serve 输入分布不一致。
# -----------------------------------------------------------------------------

COMPUTER_USE_DOUBAO = """You are a GUI agent. You are given a task and your action history, with screenshots. You need to perform the next action to complete the task.

## Output Format
You should first think about the reasoning process in the mind and then provide the user with the answer. 
The reasoning process is enclosed within <think> </think> tags
After the <think> tags, you should place final answer, which concludes your summarized thought and your action.

For example,
```
<think>detailed reasoning content here</think>
Thought: a small plan and finally summarize your next action (with its target element) in one sentence
Action: ...
```

## Action Space

click(point='<point>x1 y1</point>')
left_double(point='<point>x1 y1</point>')
right_single(point='<point>x1 y1</point>')
drag(start_point='<point>x1 y1</point>', end_point='<point>x2 y2</point>')
hotkey(key='ctrl c') # Split keys with a space and use lowercase. Also, do not use more than 3 keys in one hotkey action.
type(content='xxx') # Use escape characters \\', \\\", and \\n in content part to ensure we can parse the content in normal python string format. If you want to submit your input, use \\n at the end of content. 
scroll(point='<point>x1 y1</point>', direction='down or up or right or left') # Show more information on the `direction` side.
wait() #Sleep for 5s and take a screenshot to check for any changes.
finished(content='xxx') # Use escape characters \\', \\", and \\n in content part to ensure we can parse the content in normal python string format.

## Output Example
<think>Now that...</think>
Thought: Let's click ...
Action: click(point='<point>100 200</point>')

## Note
- Use {language} in `Thought` part.
- Write a small plan and finally summarize your next action (with its target element) in one sentence in `Thought` part.
- If you have executed several same actions (like repeatedly clicking the same point) but the screen keeps no change, please try to execute a modified action when necessary.

## User Instruction
{instruction}
"""

COMPUTER_USE_NO_THINKING = """You are a GUI agent. You are given a task and your action history, with screenshots. You need to perform the next action to complete the task.

## Output Format
```
Thought: ...
Action: ...
```

## Action Space

click(point='<point>x1 y1</point>')
left_double(point='<point>x1 y1</point>')
right_single(point='<point>x1 y1</point>')
drag(start_point='<point>x1 y1</point>', end_point='<point>x2 y2</point>')
hotkey(key='ctrl c') # Split keys with a space and use lowercase. Also, do not use more than 3 keys in one hotkey action.
type(content='xxx') # Use escape characters \\', \\\", and \\n in content part to ensure we can parse the content in normal python string format. If you want to submit your input, use \\n at the end of content. 
scroll(point='<point>x1 y1</point>', direction='down or up or right or left') # Show more information on the `direction` side.
wait() #Sleep for 5s and take a screenshot to check for any changes.
finished(content='xxx') # Use escape characters \\', \\", and \\n in content part to ensure we can parse the content in normal python string format.


## Note
- Use Chinese in `Thought` part.
- Write a small plan and finally summarize your next action (with its target element) in one sentence in `Thought` part.

## User Instruction
{instruction}
"""

# -----------------------------------------------------------------------------
# 默认 User Instruction：SEM 设备拍摄样品流程（用于所有轨迹的 system prompt）
# 可通过 --instruction 或 --instruction_file 覆盖。
# -----------------------------------------------------------------------------
DEFAULT_INSTRUCTION = """1.开始进行sem设备拍摄某样品，其形貌应该是什么什么样子的
2.首先在导航相机上选择（双击）选择合适的样品聚焦的区域，选择完毕后如果屏幕上没有样品则重新进行切换其他区域进行选择
3.在选择好合适区域后进行自动对焦操作，可以观察到屏幕上样品，然后双击屏幕中样品的位置将他们移动到视野的正中央
4.点击（单击）放大倍数，选择放大倍数为10000倍后再次进行自动对焦操作
5.观察在bse模式下的图像是否清晰，如果清晰，则进行手动微调焦距，将鼠标移动到聚焦前-号前，使用滚轮进行左右调节至图像为最清晰的状态，然后进行拍照和保存工作。
6.如果在bse模式下不清楚，则需要调整为se模式，首先点击se模式，然后点击se高增益，之后拖拽se高增益后进度条，使得图像保持一个合适的亮度
7.完成亮度的调节后，再次进行自动对焦操作。
8.进行手动微调焦距，将鼠标移动到聚焦前-号前，使用滚轮进行左右调节至图像为最清晰的状态
9.点击拍照进行拍照，再进行保存
10.将设备重新回到初始状态，关闭se高增益模式，拖动进度条到最初的状态，切换回bse模式，调整放大倍数为500倍"""


def _norm_path(path: str, traj_dir: Path) -> str:
    """将 trajectory.json 中的相对路径（可能含 \\）转为基于 traj_dir 的绝对路径，便于 ms-swift 加载图片。"""
    path = path.replace("\\", os.sep)
    if os.path.isabs(path):
        return path
    return str((traj_dir / path).resolve())


def _action_to_uitars(
    action_item: dict[str, Any],
    width: int,
    height: int,
) -> str:
    """
    将轨迹中的单条 action（name + parameters）转成 UI-TARS 模型输出格式：
    Thought: ... Action: <action_string>
    与 uitars15_v2.py 中 Action Space 及 parse_action_to_structure_output 的输入格式一致。
    坐标：轨迹里为归一化 (0~1)，这里转为像素整数，与 prompt 中 <point>x y</point> 一致。
    """
    name = action_item.get("name", "")
    params = action_item.get("parameters") or {}

    def px(x: float) -> int:
        return int(round(x * width))

    def py(y: float) -> int:
        return int(round(y * height))

    def point(x: float, y: float) -> str:
        return f"<point>{px(x)} {py(y)}</point>"

    # 转义 content 中的特殊字符，与 agent 侧 type(content='...') 解析一致
    def escape_content(s: str) -> str:
        s = s.replace("\\", "\\\\").replace("'", "\\'").replace('"', '\\"')
        s = s.replace("\n", "\\n")
        return s

    action_str: str
    if name == "click":
        x, y = params.get("x", 0), params.get("y", 0)
        action_str = f"click(point='{point(x, y)}')"
    elif name == "doubleClick":
        x, y = params.get("x", 0), params.get("y", 0)
        action_str = f"left_double(point='{point(x, y)}')"
    elif name == "rightClick":
        x, y = params.get("x", 0), params.get("y", 0)
        action_str = f"right_single(point='{point(x, y)}')"
    elif name == "dragTo":
        sx = params.get("start_x", 0)
        sy = params.get("start_y", 0)
        ex = params.get("end_x", 0)
        ey = params.get("end_y", 0)
        action_str = (
            f"drag(start_point='{point(sx, sy)}', end_point='{point(ex, ey)}')"
        )
    elif name == "wait":
        action_str = "wait()"
    elif name == "scroll":
        x, y = params.get("x", 0), params.get("y", 0)
        direction = params.get("direction", "down")
        action_str = f"scroll(point='{point(x, y)}', direction='{direction}')"
    elif name == "type":
        content = params.get("content", "")
        action_str = f"type(content='{escape_content(content)}')"
    elif name == "hotkey" or name == "keyboard":
        key = params.get("key", params.get("hotkey", ""))
        action_str = f"hotkey(key='{key}')"
    elif name == "finished":
        content = params.get("content", "")
        action_str = f"finished(content='{escape_content(content)}')"
    else:
        # 未知动作保留原名与参数，便于后续扩展
        action_str = f"{name}({json.dumps(params, ensure_ascii=False)})"

    return action_str


def _step_to_thought_action(
    step: dict[str, Any],
    width: int,
    height: int,
) -> str:
    """
    将轨迹中一步转为模型单轮输出：Thought: ... Action: ...
    优先使用 step_instruction / rationale 作为 Thought，否则用简短默认描述。
    """
    thought = (
        (step.get("step_instruction") or step.get("rationale") or "").strip()
        or "执行当前步骤操作。"
    )
    actions = step.get("action") or []
    if not actions:
        return f"Thought: {thought}\nAction: wait()"
    parts = [_action_to_uitars(a, width, height) for a in actions]
    action_line = " ".join(parts) if len(parts) == 1 else parts[0]
    return f"Thought: {thought}\nAction: {action_line}"


def _collect_trajectory_dirs(traj_root: Path):
    """收集所有包含 trajectory.json 的轨迹目录（与原始 traj 目录结构一致）。"""
    dirs = []
    for root, _, files in os.walk(traj_root):
        if "trajectory.json" in files:
            dirs.append(Path(root))
    return dirs


def convert_trajectory_to_samples(
    traj_path: Path,
    system_prompt_template: str,
    instruction: str,
    language: str = "Chinese",
    max_image_history_length: int = 5,
    use_thinking: bool = True,
) -> list[dict[str, Any]]:
    """
    将单条轨迹转为多条 ms-swift 多模态 SFT 样本。

    策略（与 UI-TARS 1.5 推理一致）：
    - 每个 step 作为一条训练样本的「当前步」；
    - 每条样本只保留「当前步」之前的 max_image_history_length 步作为历史
      （即 [t - max_image_history_length + 1, t] 的滑动窗口），这样模型输入
      的图片数与推理时 history_n 一致，避免 train/serve 不一致。
    - 每条样本格式：system + (user<image>, assistant response) × N，N ≤ max_image_history_length。
    """
    p = traj_path / "trajectory.json"
    if not p.is_file():
        return []

    with open(p, "r", encoding="utf-8") as f:
        data = json.load(f)

    trajectory = data.get("trajectory") or []
    if not trajectory:
        return []

    # 使用传入的 instruction（默认 SEM 拍摄流程），不再使用 trajectory 内的 query
    # query = data.get("query") or "完成当前任务。"
    system_content = system_prompt_template.format(
        # instruction=query,
        instruction=instruction,
        language=language,
    )
    samples = []

    for t in range(len(trajectory)):
        # 与 uitars15_v2 的 history_n 一致：只保留当前步及前 (max_image_history_length - 1) 步
        start_idx = max(0, t - max_image_history_length + 1)
        steps_slice = trajectory[start_idx : t + 1]

        images = []
        messages = [{"role": "system", "content": system_content}]

        for i, step in enumerate(steps_slice):
            obs = step.get("observation") or f"images/step_{start_idx + i}.png"
            img_path = _norm_path(obs, traj_path)
            if not os.path.isfile(img_path):
                # 跳过缺失图片的步，避免训练时报错
                continue
            images.append(img_path)
            # ms-swift 多模态：content 中一个 <image> 对应 images 列表中的一张图，顺序一致
            messages.append({"role": "user", "content": "<image>"})
            response = _step_to_thought_action(
                step,
                step.get("width", 1920),
                step.get("height", 1080),
            )
            messages.append({"role": "assistant", "content": response})

        if not images:
            continue
        # 标准格式：messages + images，符合 swift 文档中的多模态 SFT 示例
        samples.append({"messages": messages, "images": images})

    return samples


def main():
    parser = argparse.ArgumentParser(
        description="将 traj 轨迹数据转换为 ms-swift 多模态 SFT 格式（traj_v2）"
    )
    parser.add_argument(
        "--traj_dir",
        type=str,
        default="/data/250010149/hf_cache/traj",
        help="原始轨迹根目录",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="/data/250010149/hf_cache/traj_v2",
        help="输出目录，将写入 train.jsonl 等",
    )
    parser.add_argument(
        "--max_image_history_length",
        type=int,
        default=5,
        help="每条样本中保留的最大历史图片数，与 UI-TARS 1.5 的 history_n 一致，影响模型输入长度",
    )
    parser.add_argument(
        "--use_thinking",
        action="store_true",
        default=True,
        help="使用带 <think> 的 prompt 模板（与 uitars15_v2 use_thinking 一致）",
    )
    parser.add_argument(
        "--no_use_thinking",
        action="store_false",
        dest="use_thinking",
        help="不使用 <think>，改用 COMPUTER_USE_NO_THINKING",
    )
    parser.add_argument(
        "--language",
        type=str,
        default="Chinese",
        help="Prompt 中的语言占位 {language}",
    )
    parser.add_argument(
        "--split_ratio",
        type=float,
        default=0.90,
        help="训练集占比，其余为验证集；0 表示不划分验证集",
    )
    parser.add_argument(
        "--instruction",
        type=str,
        default=None,
        help="User Instruction 文本，替换默认的 SEM 拍摄流程；与 --instruction_file 二选一",
    )
    parser.add_argument(
        "--instruction_file",
        type=str,
        default=None,
        help="从文件读取 User Instruction（UTF-8），一行或整段均可",
    )
    args = parser.parse_args()

    traj_root = Path(args.traj_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    system_prompt = (
        COMPUTER_USE_DOUBAO if args.use_thinking else COMPUTER_USE_NO_THINKING
    )

    # 确定 User Instruction：文件优先，其次命令行，最后用默认 SEM 流程
    if args.instruction_file:
        with open(args.instruction_file, "r", encoding="utf-8") as f:
            instruction = f.read().strip()
    elif args.instruction is not None:
        instruction = args.instruction
    else:
        instruction = DEFAULT_INSTRUCTION

    all_samples = []
    traj_dirs = _collect_trajectory_dirs(traj_root)
    for d in traj_dirs:
        samples = convert_trajectory_to_samples(
            d,
            system_prompt_template=system_prompt,
            instruction=instruction,
            language=args.language,
            max_image_history_length=args.max_image_history_length,
            use_thinking=args.use_thinking,
        )
        all_samples.extend(samples)

    if not all_samples:
        print("No samples generated. Check traj_dir and trajectory.json structure.")
        return

    # 写入 jsonl（ms-swift 支持 --dataset xxx.jsonl）
    train_path = out_dir / "train.jsonl"
    val_path = out_dir / "val.jsonl"
    n = len(all_samples)
    split = int(n * args.split_ratio) if args.split_ratio > 0 else n
    train_samples = all_samples[:split]
    val_samples = all_samples[split:] if split < n else []

    with open(train_path, "w", encoding="utf-8") as f:
        for s in train_samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    print(f"Wrote {len(train_samples)} samples to {train_path}")

    if val_samples:
        with open(val_path, "w", encoding="utf-8") as f:
            for s in val_samples:
                f.write(json.dumps(s, ensure_ascii=False) + "\n")
        print(f"Wrote {len(val_samples)} samples to {val_path}")

    # 可选：写一份 dataset_info 示例，便于用 --custom_dataset_info 指定
    info = {
        "dataset_path": str(out_dir.resolve()),
        "dataset_name": "traj_cua_sft",
        "columns": None,
    }
    info_path = out_dir / "dataset_info.json"
    with open(info_path, "w", encoding="utf-8") as f:
        json.dump([info], f, ensure_ascii=False, indent=2)
    print(f"Wrote {info_path} (optional, for --custom_dataset_info)")


if __name__ == "__main__":
    main()
