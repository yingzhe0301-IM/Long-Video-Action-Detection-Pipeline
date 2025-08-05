# config.py

class ActionDetectionConfig:
    def __init__(self) -> None:
        # --- Parameters for Peak-based detection ---
        self.peak_prominence_factor: float = 0.05
        self.fraction_of_max: float = 0.20
        self.adaptive_factor: float = 2.0
        self.valley_ratio: float = 0.7
        # Note: This default value is for frame_skip=1
        self.coalesce_time_thr: float = 400

        # --- Parameters for Interval-based detection ('setting' action) ---

        # MODIFIED: Renamed for clarity and added new thresholds for ending an action.

        # The minimum 'average detections' value required to START an action interval.
        self.setting_action_start_threshold: float = 0.5

        # The 'average detections' value must fall BELOW this threshold to be considered for ending an action.
        self.setting_action_end_threshold: float = 0.0

        # The minimum duration (in frames, assuming frame_skip=1) that the signal must remain
        # below 'setting_action_end_threshold' to officially mark the end of an action.
        self.setting_action_end_duration_threshold: int = 400

        # The minimum duration (in frames, assuming frame_skip=1) an action must last to be considered valid.
        # This filters out brief, spurious detection blips. This value is scaled automatically with frame_skip.
        self.setting_action_min_duration: int = 1000

        # The minimum peak value that the 'average detections' must reach within an interval
        # for that interval to be considered a valid, high-intensity action.
        self.setting_action_max_val_threshold: float = 3


# Global singleton instance
action_config: ActionDetectionConfig = ActionDetectionConfig()