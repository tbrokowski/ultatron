"""vlm/rewards/  ·  Reward functions for Ultatron VLM GRPO."""
from vlm.rewards.base        import RewardFunction, RewardOutput
from vlm.rewards.format_reward import FormatReward
from vlm.rewards.seg_reward  import SegmentationReward
from vlm.rewards.medgemini   import MedGeminiReward
from vlm.rewards.composite   import CompositeReward
from vlm.rewards.teacher_rubric import TeacherRubricReward, DEFAULT_WEIGHTS

__all__ = [
    "RewardFunction", "RewardOutput",
    "FormatReward",
    "SegmentationReward",
    "MedGeminiReward",
    "CompositeReward",
    "TeacherRubricReward",
    "DEFAULT_WEIGHTS",
]
