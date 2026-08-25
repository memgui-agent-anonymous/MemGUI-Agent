#!/usr/bin/env python3
"""
MemGUI-3K 测试集自动评估脚本

参考 ClawGUI/clawgui-eval 的评估方案:
  - UI 动作: type_match (动作类型匹配) + match_success (完整操作匹配)
  - 坐标匹配: 欧氏距离 ≤ 14% 屏幕对角线
  - 文本匹配: exact match 或 token-level F1 > 0.5

在此基础上扩展:
  - Memory 动作: type_match + match_success (content F1)
  - Folding: presence / range / depth_type 三级匹配
  - Format Compliance: 5-part XML tag 完整性

Usage:
    python evaluate_predictions.py \
        --predictions results/memgui8b_sft_predictions.jsonl \
        --gold data/MemGUI-3K/training_data/test_sft.jsonl \
        --output results/evaluation_report.json
"""

import argparse
import json
import math
import os
import re
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional, Tuple


# ─── Screen & Thresholds (following ClawGUI/AndroidControl) ──────────────────

SCREEN_WIDTH = 1000
SCREEN_HEIGHT = 1000
CLICK_THRESHOLD = math.sqrt((SCREEN_WIDTH * 0.14) ** 2 + (SCREEN_HEIGHT * 0.14) ** 2)
SWIPE_THRESHOLD = CLICK_THRESHOLD * 1.5
TEXT_F1_THRESHOLD = 0.5
MEMORY_CONTENT_F1_THRESHOLD = 0.5
FOLD_RANGE_TOLERANCE = 2


# ─── Text Matching (from ClawGUI) ────────────────────────────────────────────

def calculate_f1_score(predicted_str: str, ground_truth_str: str) -> float:
    if not predicted_str or not ground_truth_str:
        return 0.0
    predicted_tokens = set(predicted_str.lower().split())
    ground_truth_tokens = set(ground_truth_str.lower().split())
    common = predicted_tokens & ground_truth_tokens
    if not predicted_tokens:
        precision = 0
    else:
        precision = len(common) / len(predicted_tokens)
    if not ground_truth_tokens:
        recall = 0
    else:
        recall = len(common) / len(ground_truth_tokens)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def text_matching(gt_text: str, pred_text: str) -> bool:
    if gt_text.strip() == pred_text.strip():
        return True
    return calculate_f1_score(pred_text, gt_text) > TEXT_F1_THRESHOLD


# ─── Coordinate Matching (from ClawGUI) ──────────────────────────────────────

def coord_distance(c1: List, c2: List) -> float:
    if not c1 or not c2 or len(c1) < 2 or len(c2) < 2:
        return float("inf")
    try:
        return math.sqrt((float(c1[0]) - float(c2[0])) ** 2 + (float(c1[1]) - float(c2[1])) ** 2)
    except (TypeError, ValueError):
        return float("inf")


# ─── Response Parsing ─────────────────────────────────────────────────────────

def parse_tool_call(response: str) -> Optional[Dict]:
    m = re.search(r"<tool_call>(.*?)</tool_call>", response, re.DOTALL)
    if not m:
        return None
    try:
        return json.loads(m.group(1).strip())
    except json.JSONDecodeError:
        return None


def parse_folding(response: str) -> Optional[Dict]:
    m = re.search(r"<folding>(.*?)</folding>", response, re.DOTALL)
    if not m:
        return None
    try:
        return json.loads(m.group(1).strip())
    except json.JSONDecodeError:
        return None


def parse_tag_content(response: str, tag: str) -> Optional[str]:
    m = re.search(rf"<{tag}>(.*?)</{tag}>", response, re.DOTALL)
    return m.group(1).strip() if m else None


def extract_action_info(tool_call: Optional[Dict]) -> Tuple[str, Dict]:
    if not tool_call:
        return "", {}
    args = tool_call.get("arguments", {})
    action_type = args.get("action", "")
    return action_type, args


# ─── Action Categories ────────────────────────────────────────────────────────

UI_ACTIONS = {"click", "long_press", "swipe", "type", "answer", "system_button", "wait", "terminate"}
MEMORY_ACTIONS = {"memory_add", "memory_update", "memory_delete"}
ALL_ACTIONS = UI_ACTIONS | MEMORY_ACTIONS


# ─── UI Action Matching (following ClawGUI action_matching pattern) ───────────

def match_ui_action(pred_type: str, pred_args: Dict, gold_type: str, gold_args: Dict) -> Dict[str, Any]:
    """
    Two-level evaluation:
      type_match:    action type 是否一致
      match_success: 完整操作是否匹配 (坐标/文本/参数)
    """
    type_match = pred_type == gold_type
    if not type_match:
        return {
            "type_match": False,
            "match_success": False,
            "info": f"type_mismatch:{pred_type}_vs_{gold_type}",
        }

    if gold_type in ("click", "long_press"):
        dist = coord_distance(pred_args.get("coordinate"), gold_args.get("coordinate"))
        success = dist <= CLICK_THRESHOLD
        return {
            "type_match": True,
            "match_success": success,
            "info": f"coord_dist={dist:.1f},thr={CLICK_THRESHOLD:.1f}",
        }

    elif gold_type == "swipe":
        d1 = coord_distance(pred_args.get("coordinate"), gold_args.get("coordinate"))
        d2 = coord_distance(pred_args.get("coordinate2"), gold_args.get("coordinate2"))
        success = d1 <= SWIPE_THRESHOLD and d2 <= SWIPE_THRESHOLD
        return {
            "type_match": True,
            "match_success": success,
            "info": f"swipe_d1={d1:.1f},d2={d2:.1f},thr={SWIPE_THRESHOLD:.1f}",
        }

    elif gold_type in ("type", "answer"):
        gt_text = str(gold_args.get("text", "")).strip()
        pred_text = str(pred_args.get("text", "")).strip()
        success = text_matching(gt_text, pred_text)
        f1 = calculate_f1_score(pred_text, gt_text)
        return {
            "type_match": True,
            "match_success": success,
            "info": f"text_f1={f1:.3f},exact={gt_text == pred_text}",
        }

    elif gold_type == "system_button":
        success = pred_args.get("button") == gold_args.get("button")
        return {
            "type_match": True,
            "match_success": success,
            "info": f"button:{pred_args.get('button')}_vs_{gold_args.get('button')}",
        }

    elif gold_type == "terminate":
        success = pred_args.get("status") == gold_args.get("status")
        return {
            "type_match": True,
            "match_success": success,
            "info": f"status:{pred_args.get('status')}_vs_{gold_args.get('status')}",
        }

    elif gold_type == "wait":
        return {"type_match": True, "match_success": True, "info": "wait_ok"}

    else:
        return {
            "type_match": True,
            "match_success": pred_args == gold_args,
            "info": f"generic_eq={pred_args == gold_args}",
        }


# ─── Memory Action Matching ──────────────────────────────────────────────────

def match_memory_action(pred_type: str, pred_args: Dict, gold_type: str, gold_args: Dict) -> Dict[str, Any]:
    """
    Two-level evaluation for memory actions:
      type_match:    memory 动作类型是否一致 (add/update/delete)
      match_success: 模型知道在这一步该做 memory 操作且语义意图匹配
                     (description F1 > 0.3, 或 content F1 > 0.3)
    
    注意: 离线评估中 content 的精确值（如价格数字）可能因前序步骤不同而不同，
    因此 match_success 不依赖 content exact match，而是看 description 语义
    是否表达了相同的记忆意图。content F1 单独作为附加指标报告。
    """
    type_match = pred_type == gold_type
    if not type_match:
        return {
            "type_match": False,
            "match_success": False,
            "info": f"mem_type_mismatch:{pred_type}_vs_{gold_type}",
        }

    if gold_type == "memory_delete":
        id_match = pred_args.get("memory_id", "").strip() == gold_args.get("memory_id", "").strip()
        return {
            "type_match": True,
            "match_success": id_match,
            "info": f"mem_delete_id_match={id_match}",
        }

    pred_content = str(pred_args.get("content", "")).strip()
    gold_content = str(gold_args.get("content", "")).strip()
    content_f1 = calculate_f1_score(pred_content, gold_content)

    pred_desc = str(pred_args.get("description", "")).strip()
    gold_desc = str(gold_args.get("description", "")).strip()
    desc_f1 = calculate_f1_score(pred_desc, gold_desc) if gold_desc else 1.0

    success = desc_f1 > MEMORY_CONTENT_F1_THRESHOLD or content_f1 > MEMORY_CONTENT_F1_THRESHOLD
    return {
        "type_match": True,
        "match_success": success,
        "info": f"mem_content_f1={content_f1:.3f},desc_f1={desc_f1:.3f}",
        "content_f1": content_f1,
        "desc_f1": desc_f1,
    }


# ─── Unified Action Matching ─────────────────────────────────────────────────

def match_action(pred_tc: Optional[Dict], gold_tc: Optional[Dict]) -> Dict[str, Any]:
    if pred_tc is None and gold_tc is None:
        return {"type_match": True, "match_success": True, "info": "both_none", "category": "none"}
    if pred_tc is None:
        gold_type, _ = extract_action_info(gold_tc)
        cat = "memory" if gold_type in MEMORY_ACTIONS else "ui"
        return {"type_match": False, "match_success": False, "info": "pred_none", "category": cat, "gold_action": gold_type, "pred_action": ""}
    if gold_tc is None:
        pred_type, _ = extract_action_info(pred_tc)
        cat = "memory" if pred_type in MEMORY_ACTIONS else "ui"
        return {"type_match": False, "match_success": False, "info": "gold_none", "category": cat, "gold_action": "", "pred_action": pred_type}

    pred_type, pred_args = extract_action_info(pred_tc)
    gold_type, gold_args = extract_action_info(gold_tc)

    is_gold_memory = gold_type in MEMORY_ACTIONS
    is_pred_memory = pred_type in MEMORY_ACTIONS

    if is_gold_memory or is_pred_memory:
        result = match_memory_action(pred_type, pred_args, gold_type, gold_args)
        result["category"] = "memory"
    else:
        result = match_ui_action(pred_type, pred_args, gold_type, gold_args)
        result["category"] = "ui"

    result["gold_action"] = gold_type
    result["pred_action"] = pred_type
    return result


# ─── Folding Matching ─────────────────────────────────────────────────────────

def match_folding(pred_fold: Optional[Dict], gold_fold: Optional[Dict]) -> Dict[str, Any]:
    pred_has = pred_fold is not None
    gold_has = gold_fold is not None

    if not gold_has and not pred_has:
        return {"presence_match": True, "range_match": True, "depth_type_match": True, "info": "both_absent"}
    if gold_has and not pred_has:
        return {"presence_match": False, "range_match": False, "depth_type_match": False, "info": "pred_missing"}
    if not gold_has and pred_has:
        return {"presence_match": False, "range_match": False, "depth_type_match": False, "info": "pred_extra"}

    pred_range = pred_fold.get("range", [])
    gold_range = gold_fold.get("range", [])

    range_match = False
    if len(pred_range) == 2 and len(gold_range) == 2:
        range_match = (
            abs(pred_range[0] - gold_range[0]) <= FOLD_RANGE_TOLERANCE
            and abs(pred_range[1] - gold_range[1]) <= FOLD_RANGE_TOLERANCE
        )

    pred_is_deep = len(pred_range) == 2 and pred_range[1] > pred_range[0]
    gold_is_deep = len(gold_range) == 2 and gold_range[1] > gold_range[0]
    depth_type_match = pred_is_deep == gold_is_deep

    return {
        "presence_match": True,
        "range_match": range_match,
        "depth_type_match": depth_type_match,
        "gold_is_deep": gold_is_deep,
        "pred_is_deep": pred_is_deep,
        "info": f"range:pred={pred_range},gold={gold_range},deep:pred={pred_is_deep},gold={gold_is_deep}",
    }


# ─── Format Compliance ────────────────────────────────────────────────────────

REQUIRED_TAGS = ["thinking", "folding", "tool_call", "ui_observation", "action_intent"]


def check_format_compliance(response: str, step: int) -> Dict[str, bool]:
    results = {}
    for tag in REQUIRED_TAGS:
        if tag == "folding" and step <= 1:
            results[tag] = True
            continue
        results[tag] = f"<{tag}>" in response and f"</{tag}>" in response
    return results


# ─── Main Evaluation ─────────────────────────────────────────────────────────

def evaluate(predictions_path: str, gold_path: str) -> Tuple[Dict, List[Dict]]:
    preds = {}
    with open(predictions_path, "r", encoding="utf-8") as f:
        for line in f:
            entry = json.loads(line.strip())
            key = (entry["task_id"], entry["step"])
            preds[key] = entry

    golds = {}
    with open(gold_path, "r", encoding="utf-8") as f:
        for line in f:
            entry = json.loads(line.strip())
            key = (entry["task_id"], entry["step"])
            gold_response = entry["messages"][2]["content"] if len(entry["messages"]) > 2 else ""
            golds[key] = gold_response

    # ── Per-step evaluation ──
    step_results = []
    # UI action stats
    ui_type_match = 0
    ui_match_success = 0
    ui_total = 0
    ui_action_stats = defaultdict(lambda: {"total": 0, "type_match": 0, "match_success": 0})
    # Memory action stats
    mem_type_match = 0
    mem_match_success = 0
    mem_total = 0
    mem_action_stats = defaultdict(lambda: {"total": 0, "type_match": 0, "match_success": 0})
    mem_content_f1s = []
    # Memory trigger: gold_is_memory vs pred_is_memory (for precision/recall)
    mem_trigger_gold = 0
    mem_trigger_pred = 0
    mem_trigger_tp = 0
    # Overall action stats
    all_type_match = 0
    all_match_success = 0
    total = 0
    # Folding stats
    fold_presence = []
    fold_range = []
    fold_depth_type = []
    fold_range_shallow = []
    fold_range_deep = []
    fold_pred_deep_count = 0
    fold_pred_shallow_count = 0
    # Format stats
    format_tag_scores = defaultdict(list)
    format_all_correct_count = 0

    for key in sorted(golds.keys()):
        if key not in preds:
            continue

        total += 1
        task_id, step = key
        pred_response = preds[key].get("prediction", "")
        gold_response = golds[key]

        # ── Action ──
        pred_tc = parse_tool_call(pred_response)
        gold_tc = parse_tool_call(gold_response)
        action_result = match_action(pred_tc, gold_tc)

        if action_result["type_match"]:
            all_type_match += 1
        if action_result["match_success"]:
            all_match_success += 1

        gold_action = action_result.get("gold_action", "")
        pred_action = action_result.get("pred_action", "")
        cat = action_result.get("category", "ui")

        if cat == "ui" and gold_action:
            ui_total += 1
            if action_result["type_match"]:
                ui_type_match += 1
            if action_result["match_success"]:
                ui_match_success += 1
            ui_action_stats[gold_action]["total"] += 1
            if action_result["type_match"]:
                ui_action_stats[gold_action]["type_match"] += 1
            if action_result["match_success"]:
                ui_action_stats[gold_action]["match_success"] += 1

        if cat == "memory" and (gold_action in MEMORY_ACTIONS or pred_action in MEMORY_ACTIONS):
            if gold_action in MEMORY_ACTIONS:
                mem_total += 1
                mem_trigger_gold += 1
                if action_result["type_match"]:
                    mem_type_match += 1
                if action_result["match_success"]:
                    mem_match_success += 1
                mem_action_stats[gold_action]["total"] += 1
                if action_result["type_match"]:
                    mem_action_stats[gold_action]["type_match"] += 1
                if action_result["match_success"]:
                    mem_action_stats[gold_action]["match_success"] += 1
                if "content_f1" in action_result:
                    mem_content_f1s.append(action_result["content_f1"])
            if pred_action in MEMORY_ACTIONS:
                mem_trigger_pred += 1
            if gold_action in MEMORY_ACTIONS and pred_action in MEMORY_ACTIONS:
                mem_trigger_tp += 1

        # ── Folding ──
        pred_fold = parse_folding(pred_response)
        gold_fold = parse_folding(gold_response)
        fold_result = match_folding(pred_fold, gold_fold)
        fold_presence.append(fold_result["presence_match"])
        fold_range.append(fold_result["range_match"])
        fold_depth_type.append(fold_result["depth_type_match"])
        if fold_result.get("gold_is_deep") is not None:
            if fold_result["gold_is_deep"]:
                fold_range_deep.append(fold_result["range_match"])
            else:
                fold_range_shallow.append(fold_result["range_match"])
        if fold_result.get("pred_is_deep") is not None:
            if fold_result["pred_is_deep"]:
                fold_pred_deep_count += 1
            else:
                fold_pred_shallow_count += 1

        # ── Format ──
        fmt = check_format_compliance(pred_response, step)
        all_ok = True
        for tag, ok in fmt.items():
            format_tag_scores[tag].append(ok)
            if not ok:
                all_ok = False
        if all_ok:
            format_all_correct_count += 1

        step_results.append({
            "task_id": task_id,
            "step": step,
            "category": cat,
            "gold_action": gold_action,
            "pred_action": pred_action,
            "type_match": action_result["type_match"],
            "match_success": action_result["match_success"],
            "action_info": action_result["info"],
            "fold_presence": fold_result["presence_match"],
            "fold_range": fold_result["range_match"],
            "fold_depth_type": fold_result["depth_type_match"],
            "format_all_ok": all_ok,
        })

    # ── Memory trigger precision / recall / F1 ──
    mem_trigger_precision = mem_trigger_tp / mem_trigger_pred if mem_trigger_pred else 0
    mem_trigger_recall = mem_trigger_tp / mem_trigger_gold if mem_trigger_gold else 0
    mem_trigger_f1 = (
        2 * mem_trigger_precision * mem_trigger_recall / (mem_trigger_precision + mem_trigger_recall)
        if (mem_trigger_precision + mem_trigger_recall) else 0
    )

    # ── Trajectory-level ──
    traj_groups = defaultdict(list)
    for sr in step_results:
        traj_groups[sr["task_id"]].append(sr)
    traj_full_match = sum(1 for steps in traj_groups.values() if all(s["match_success"] for s in steps))

    # ── Build report ──
    def _safe_pct(n, d):
        return round(n / d * 100, 2) if d else 0

    def _action_breakdown(stats_dict):
        out = {}
        for action, s in sorted(stats_dict.items()):
            out[action] = {
                "total": s["total"],
                "type_match": s["type_match"],
                "match_success": s["match_success"],
                "type_acc": _safe_pct(s["type_match"], s["total"]),
                "match_acc": _safe_pct(s["match_success"], s["total"]),
            }
        return out

    import numpy as np

    report = {
        "total_steps": total,
        "total_trajectories": len(traj_groups),
        "overall": {
            "type_match": all_type_match,
            "match_success": all_match_success,
            "type_acc": _safe_pct(all_type_match, total),
            "match_acc": _safe_pct(all_match_success, total),
        },
        "ui_actions": {
            "total": ui_total,
            "type_match": ui_type_match,
            "match_success": ui_match_success,
            "type_acc": _safe_pct(ui_type_match, ui_total),
            "match_acc": _safe_pct(ui_match_success, ui_total),
            "by_action": _action_breakdown(ui_action_stats),
        },
        "memory_actions": {
            "total": mem_total,
            "type_match": mem_type_match,
            "match_success": mem_match_success,
            "type_acc": _safe_pct(mem_type_match, mem_total),
            "match_acc": _safe_pct(mem_match_success, mem_total),
            "by_action": _action_breakdown(mem_action_stats),
            "content_f1_mean": round(float(np.mean(mem_content_f1s)), 4) if mem_content_f1s else 0,
            "trigger": {
                "gold_count": mem_trigger_gold,
                "pred_count": mem_trigger_pred,
                "tp": mem_trigger_tp,
                "precision": round(mem_trigger_precision * 100, 2),
                "recall": round(mem_trigger_recall * 100, 2),
                "f1": round(mem_trigger_f1 * 100, 2),
            },
        },
        "folding": {
            "presence_acc": _safe_pct(sum(fold_presence), len(fold_presence)) if fold_presence else 0,
            "range_acc": _safe_pct(sum(fold_range), len(fold_range)) if fold_range else 0,
            "range_acc_shallow": _safe_pct(sum(fold_range_shallow), len(fold_range_shallow)) if fold_range_shallow else 0,
            "range_acc_deep": _safe_pct(sum(fold_range_deep), len(fold_range_deep)) if fold_range_deep else 0,
            "depth_type_acc": _safe_pct(sum(fold_depth_type), len(fold_depth_type)) if fold_depth_type else 0,
            "pred_deep_count": fold_pred_deep_count,
            "pred_shallow_count": fold_pred_shallow_count,
            "pred_deep_ratio": _safe_pct(fold_pred_deep_count, fold_pred_deep_count + fold_pred_shallow_count),
            "gold_shallow_count": len(fold_range_shallow),
            "gold_deep_count": len(fold_range_deep),
        },
        "format_compliance": {
            "all_correct": _safe_pct(format_all_correct_count, total),
            "by_tag": {
                tag: _safe_pct(sum(scores), len(scores))
                for tag, scores in format_tag_scores.items()
            },
        },
        "trajectory_level": {
            "total": len(traj_groups),
            "full_match": traj_full_match,
            "full_match_rate": _safe_pct(traj_full_match, len(traj_groups)),
        },
    }

    return report, step_results


# ─── CLI ──────────────────────────────────────────────────────────────────────

def print_report(report: Dict):
    print(f"\n{'='*70}")
    print(f"  MemGUI-3K Offline Evaluation Report")
    print(f"{'='*70}")
    print(f"  Steps: {report['total_steps']}  |  Trajectories: {report['total_trajectories']}")

    o = report["overall"]
    print(f"\n--- Overall Action Matching ---")
    print(f"  Type Accuracy:  {o['type_acc']:.2f}%  ({o['type_match']}/{report['total_steps']})")
    print(f"  Match Accuracy: {o['match_acc']:.2f}%  ({o['match_success']}/{report['total_steps']})")

    u = report["ui_actions"]
    print(f"\n--- UI Actions ({u['total']} steps) ---")
    print(f"  Type Accuracy:  {u['type_acc']:.2f}%")
    print(f"  Match Accuracy: {u['match_acc']:.2f}%")
    print(f"  {'Action':<15} {'Total':>6} {'Type%':>8} {'Match%':>8}")
    print(f"  {'-'*40}")
    for action, s in u["by_action"].items():
        print(f"  {action:<15} {s['total']:>6} {s['type_acc']:>8.2f} {s['match_acc']:>8.2f}")

    m = report["memory_actions"]
    print(f"\n--- Memory Actions ({m['total']} steps) ---")
    print(f"  Type Accuracy:  {m['type_acc']:.2f}%")
    print(f"  Match Accuracy: {m['match_acc']:.2f}%")
    print(f"  Content F1:     {m['content_f1_mean']:.4f}")
    t = m["trigger"]
    print(f"  Trigger:  P={t['precision']:.1f}%  R={t['recall']:.1f}%  F1={t['f1']:.1f}%  (TP={t['tp']}, Pred={t['pred_count']}, Gold={t['gold_count']})")
    for action, s in m["by_action"].items():
        print(f"    {action:<15} {s['total']:>5} type={s['type_acc']:.1f}%  match={s['match_acc']:.1f}%")

    fl = report["folding"]
    print(f"\n--- Folding ---")
    print(f"  Presence Accuracy:   {fl['presence_acc']:.2f}%")
    print(f"  Range Accuracy:      {fl['range_acc']:.2f}%  (shallow={fl['range_acc_shallow']:.1f}%, deep={fl['range_acc_deep']:.1f}%)")
    print(f"  Depth Type Accuracy: {fl['depth_type_acc']:.2f}%")
    print(f"  Pred deep ratio:     {fl['pred_deep_ratio']:.1f}%  ({fl['pred_deep_count']} deep / {fl['pred_shallow_count']} shallow)")
    print(f"  Gold distribution:   {fl['gold_shallow_count']} shallow, {fl['gold_deep_count']} deep")

    fc = report["format_compliance"]
    print(f"\n--- Format Compliance ---")
    print(f"  All Tags Correct:  {fc['all_correct']:.2f}%")
    for tag, pct in fc["by_tag"].items():
        print(f"    <{tag}>: {pct:.1f}%")

    tl = report["trajectory_level"]
    print(f"\n--- Trajectory-Level ---")
    print(f"  Full Match Rate: {tl['full_match_rate']:.2f}%  ({tl['full_match']}/{tl['total']})")
    print(f"{'='*70}\n")


def main():
    parser = argparse.ArgumentParser(description="MemGUI-3K Offline Evaluation")
    parser.add_argument("--predictions", required=True, help="Predictions jsonl")
    parser.add_argument("--gold", required=True, help="Gold test_sft.jsonl")
    parser.add_argument("--output", required=True, help="Output report json")
    args = parser.parse_args()

    print("Running evaluation...")
    report, step_results = evaluate(args.predictions, args.gold)

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    step_output = args.output.replace(".json", "_step_details.jsonl")
    with open(step_output, "w", encoding="utf-8") as f:
        for sr in step_results:
            f.write(json.dumps(sr, ensure_ascii=False) + "\n")

    print_report(report)
    print(f"Report:  {args.output}")
    print(f"Details: {step_output}")


if __name__ == "__main__":
    main()
