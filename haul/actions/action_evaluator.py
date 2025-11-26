"""Lightweight helpers to score action detection outputs."""

from typing import Dict, List, Optional


def evaluate_action(video_stem: str, detected_actions: List, action_type: str) -> Dict:

    ground_truth = _parse_ground_truth(video_stem, action_type)
    if ground_truth is None:
        return {
            "skipped": True,
            "detected_actions": len(detected_actions),
            "gt_actions": 0,
            "success": False,
        }

    return {
        "skipped": False,
        "detected_actions": len(detected_actions),
        "gt_actions": ground_truth,
        "success": len(detected_actions) == ground_truth,
    }


def _parse_ground_truth(video_stem: str, action_type: str) -> Optional[int]:
    parts = video_stem.split("_")
    suffix = parts[-1] if parts else ""
    if not suffix.isdigit():
        return None

    if action_type in {"pumping", "setting"}:
        return len(suffix) // 2 if len(suffix) % 2 == 0 else None
    if action_type in {"haul", "catch"}:
        return len(suffix) if len(suffix) % 2 == 0 else None
    return None


def calculate_accuracy(results: List[Dict]) -> Dict[str, float]:
    total = 0
    success = 0

    for result in results:
        evaluation = result.get("evaluation", {})
        if evaluation.get("skipped"):
            continue
        total += 1
        if evaluation.get("success"):
            success += 1

    accuracy = (success / total) * 100 if total else 0.0
    return {"total_videos": total, "success_count": success, "accuracy": accuracy}
