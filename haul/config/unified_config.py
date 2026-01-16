"""Peak detection parameters and scaling."""

from typing import Dict, Optional

PEAK_DETECTION_PARAMS: Dict[str, float] = {
    "peak_prominence_factor": 0.05,
    "fraction_of_max": 0.3,
    "coalesce_time_thr": 1200.0,
}


def scale_peak_params(frame_skip: int, overrides: Optional[Dict[str, float]] = None) -> Dict[str, float]:
    params = dict(PEAK_DETECTION_PARAMS)
    if overrides:
        params.update({k: v for k, v in overrides.items() if k in params})

    skip = max(int(frame_skip or 1), 1)
    params["coalesce_time_thr"] = params["coalesce_time_thr"] / skip
    return params


__all__ = ["PEAK_DETECTION_PARAMS", "scale_peak_params"]
