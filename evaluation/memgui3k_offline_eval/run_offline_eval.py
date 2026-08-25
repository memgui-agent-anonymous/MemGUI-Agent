#!/usr/bin/env python3
"""
MemGUI-3K 测试集离线推理脚本 (OpenAI-compatible API)

对 test_sft.jsonl 中的每个 step，喂入 gold prompt (system + user + screenshot)，
调用 OpenAI 格式 API 生成 response，保存预测结果。

支持:
  - 断点续跑 (--resume)
  - 高并发 (--workers, 默认 32)
  - tqdm 进度条 + 时间预估
  - 无限重试 (retryable errors)
  - 环境变量 / 命令行参数灵活配置

Usage:
    # MemGUI-8B-SFT (本地 vLLM serve)
    python run_offline_eval.py \
        --input data/MemGUI-3K/training_data/test_sft.jsonl \
        --output results/memgui8b_sft_predictions.jsonl \
        --api-base http://localhost:8000/v1 \
        --api-model MemGUI-8B-SFT

    # 远程 API
    OPENAI_API_KEY=your_key python run_offline_eval.py \
        --input data/MemGUI-3K/training_data/test_sft.jsonl \
        --output results/teacher_predictions.jsonl \
        --api-base https://dashscope.aliyuncs.com/compatible-mode/v1 \
        --api-model qwen3-vl-235b-a22b-thinking

    # 断点续跑
    python run_offline_eval.py --input ... --output results/predictions.jsonl --resume
"""

import argparse
import json
import os
import random
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Set, Tuple

from openai import OpenAI
from tqdm import tqdm


# ─── Retry Logic (infinite retry for retryable errors) ───────────────────────

def is_retryable_error(error: Exception) -> bool:
    error_str = str(error).lower()
    retryable_patterns = [
        "429", "rate limit", "too many requests",
        "500", "502", "503", "504",
        "internal server error", "service unavailable",
        "timeout", "timed out", "connection",
        "请求频率过高", "toomanyrequest",
        "overloaded", "capacity", "reset by peer",
        "broken pipe", "eof", "incomplete",
    ]
    return any(p in error_str for p in retryable_patterns)


def call_with_retry(
    func,
    base_delay: float = 1.0,
    max_delay: float = 30.0,
    max_total_time: float = 300.0,
    verbose: bool = True,
    label: str = "",
) -> Any:
    attempt = 0
    t0 = time.time()
    while True:
        try:
            return func()
        except Exception as e:
            elapsed = time.time() - t0
            if not is_retryable_error(e):
                raise
            if elapsed > max_total_time:
                raise TimeoutError(f"Exceeded {max_total_time}s total retry time after {attempt} attempts: {e}")
            attempt += 1
            delay = min(base_delay * (2 ** min(attempt, 6)), max_delay)
            delay *= 0.5 + random.random()
            if verbose and (attempt <= 3 or attempt % 5 == 0):
                tqdm.write(f"  [Retry #{attempt}, {elapsed:.0f}s] {label} {type(e).__name__}: {str(e)[:100]}... wait {delay:.1f}s")
            time.sleep(delay)


# ─── API Inference ───────────────────────────────────────────────────────────

def create_client(api_base: str, api_key: str) -> OpenAI:
    return OpenAI(base_url=api_base, api_key=api_key, timeout=60.0)


def infer_single_step(
    client: OpenAI,
    model: str,
    entry: Dict,
    max_tokens: int,
    temperature: float,
) -> Dict:
    messages = entry["messages"][:2]
    gold = entry["messages"][2]["content"] if len(entry["messages"]) > 2 else ""
    task_id = entry.get("task_id", "")
    step = entry.get("step", 0)
    label = f"{task_id}/s{step}"

    def _call():
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        content = response.choices[0].message.content
        if not content:
            raise RuntimeError("Empty response from API")

        usage = {}
        if response.usage:
            usage = {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
            }
        return content, usage

    try:
        prediction, usage = call_with_retry(_call, label=label)
    except Exception as e:
        tqdm.write(f"  [FAIL] {label}: {e}")
        prediction = ""
        usage = {}

    return {
        "task_id": task_id,
        "step": step,
        "prediction": prediction,
        "gold": gold,
        "usage": usage,
    }


# ─── Resume Support ──────────────────────────────────────────────────────────

def load_completed_keys(output_path: str) -> Set[Tuple[str, int]]:
    completed = set()
    if not os.path.exists(output_path):
        return completed
    with open(output_path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                entry = json.loads(line.strip())
                if entry.get("prediction"):
                    completed.add((entry["task_id"], entry["step"]))
            except json.JSONDecodeError:
                continue
    return completed


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="MemGUI-3K Test Set Offline Inference (OpenAI API)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--input", required=True, help="Path to test_sft.jsonl")
    parser.add_argument("--output", required=True, help="Output predictions jsonl path")
    parser.add_argument(
        "--api-base",
        default=os.environ.get("QWEN_BASE_URL") or os.environ.get("BASE_URL", "http://localhost:8000/v1"),
    )
    parser.add_argument(
        "--api-model",
        default=os.environ.get("QWEN_MODEL", "MemGUI-8B-SFT"),
    )
    parser.add_argument(
        "--api-key",
        default=os.environ.get("QWEN_API_KEY") or os.environ.get("OPENAI_API_KEY", "EMPTY"),
    )
    parser.add_argument("--max-tokens", type=int, default=16384)
    parser.add_argument("--temperature", type=float, default=0.01)
    parser.add_argument("--workers", type=int, default=32, help="Concurrent API workers")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of samples")
    parser.add_argument("--resume", action="store_true", help="Skip already completed steps")
    args = parser.parse_args()

    print("=" * 60)
    print("MemGUI-3K Test Set Offline Inference")
    print("=" * 60)
    print(f"  Input:       {args.input}")
    print(f"  Output:      {args.output}")
    print(f"  API Base:    {args.api_base}")
    print(f"  Model:       {args.api_model}")
    print(f"  Workers:     {args.workers}")
    print(f"  Max Tokens:  {args.max_tokens}")
    print(f"  Temperature: {args.temperature}")
    print(f"  Resume:      {args.resume}")
    print()

    print("Loading test data...")
    data = []
    with open(args.input, "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line.strip()))
    print(f"  Total samples: {len(data)}")

    if args.limit:
        data = data[:args.limit]
        print(f"  Limited to: {len(data)}")

    if args.resume:
        completed = load_completed_keys(args.output)
        before = len(data)
        data = [d for d in data if (d.get("task_id", ""), d.get("step", 0)) not in completed]
        print(f"  Resumed: {before - len(data)} done, {len(data)} remaining")

    if not data:
        print("Nothing to do.")
        return

    client = create_client(args.api_base, args.api_key)

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    write_lock = threading.Lock()

    mode = "a" if args.resume else "w"
    fout = open(args.output, mode, encoding="utf-8")

    pbar = tqdm(total=len(data), desc="Inference", unit="step", dynamic_ncols=True)

    def process_entry(entry):
        result = infer_single_step(
            client, args.api_model, entry,
            args.max_tokens, args.temperature,
        )
        with write_lock:
            fout.write(json.dumps(result, ensure_ascii=False) + "\n")
            fout.flush()
        pbar.update(1)
        return result

    t0 = time.time()

    if args.workers <= 1:
        for entry in data:
            process_entry(entry)
    else:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {executor.submit(process_entry, entry): entry for entry in data}
            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    entry = futures[future]
                    tqdm.write(f"  [ERROR] {entry.get('task_id')}/s{entry.get('step')}: {e}")
                    pbar.update(1)

    pbar.close()
    fout.close()
    elapsed = time.time() - t0

    print(f"\nDone! {len(data)} steps in {elapsed:.1f}s ({elapsed/max(len(data),1):.2f}s/step)")
    print(f"Predictions saved to {args.output}")


if __name__ == "__main__":
    main()
