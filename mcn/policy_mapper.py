"""
Policy Mapper
=============
Deterministic mapping from classified Scenario/Intent ID to a
ROS2-compatible behavioral policy JSON structure.

This is NOT a learned component — it's a lookup table that converts
the MCN's intent classification into actionable robot commands.
"""

import json
from typing import Dict, Optional

from .config import INTENT_CATEGORIES, INTENT_IDX2STR


# ──────────────────────────────────────────────────────────────────────
# BEHAVIORAL POLICY DEFINITIONS
# ──────────────────────────────────────────────────────────────────────

BEHAVIORAL_POLICIES: Dict[str, Dict] = {
    "HELP_REQUEST": {
        "proxemic_action": "APPROACH_AND_ENGAGE",
        "target_linear_velocity": 0.20,
        "social_buffer_zone_radius": 1.50,
        "vocal_affect_tone": "GENTLE_SUPPORTIVE",
        "priority": 3,
        "description": "User needs assistance; approach gently and offer help",
    },
    "NEUTRAL_PASS": {
        "proxemic_action": "MAINTAIN_COURSE",
        "target_linear_velocity": 0.30,
        "social_buffer_zone_radius": 1.00,
        "vocal_affect_tone": "IDLE",
        "priority": 1,
        "description": "User is passing by; maintain current trajectory",
    },
    "GIVE_WAY": {
        "proxemic_action": "MOVE_ASIDE",
        "target_linear_velocity": 0.15,
        "social_buffer_zone_radius": 2.00,
        "vocal_affect_tone": "POLITE_NEUTRAL",
        "priority": 2,
        "description": "User needs space; move aside and yield path",
    },
    "EMERGENCY": {
        "proxemic_action": "YIELD_AND_ALERT",
        "target_linear_velocity": 0.00,
        "social_buffer_zone_radius": 3.00,
        "vocal_affect_tone": "URGENT_ALERT",
        "priority": 5,
        "description": "Emergency detected; stop, clear path, alert authorities",
    },
    "GREETING": {
        "proxemic_action": "APPROACH_AND_STOP",
        "target_linear_velocity": 0.25,
        "social_buffer_zone_radius": 1.20,
        "vocal_affect_tone": "WARM_FRIENDLY",
        "priority": 2,
        "description": "User is greeting; approach and reciprocate",
    },
    "TASK_ASSIST": {
        "proxemic_action": "APPROACH_TARGET_OBJECT",
        "target_linear_velocity": 0.20,
        "social_buffer_zone_radius": 1.50,
        "vocal_affect_tone": "INFORMATIVE",
        "priority": 3,
        "description": "User needs task help; navigate to indicated object",
    },
    "HOSTILE_CONFRONTATION": {
        "proxemic_action": "RETREAT_AND_DEESCALATE",
        "target_linear_velocity": -0.10,
        "social_buffer_zone_radius": 3.50,
        "vocal_affect_tone": "CALM_DEESCALATION",
        "priority": 4,
        "description": "Hostile user detected; retreat and de-escalate",
    },
    "DISTRESSED_STUDENT_QUERY": {
        "proxemic_action": "APPROACH_AND_ENGAGE",
        "target_linear_velocity": 0.15,
        "social_buffer_zone_radius": 1.50,
        "vocal_affect_tone": "GENTLE_SUPPORTIVE",
        "priority": 3,
        "description": "Distressed student; approach carefully and offer support",
    },
    "UNKNOWN": {
        "proxemic_action": "HOLD_POSITION",
        "target_linear_velocity": 0.00,
        "social_buffer_zone_radius": 2.00,
        "vocal_affect_tone": "IDLE",
        "priority": 0,
        "description": "Intent unclear; hold position and continue observing",
    },
}


# ──────────────────────────────────────────────────────────────────────
# SCENARIO ID MAPPING (from HRI_Scenarios.pdf)
# ──────────────────────────────────────────────────────────────────────

SCENARIO_ID_MAP: Dict[str, str] = {
    "HELP_REQUEST": "SCENARIO_ID_74",
    "NEUTRAL_PASS": "SCENARIO_ID_02",
    "GIVE_WAY": "SCENARIO_ID_03",
    "EMERGENCY": "SCENARIO_ID_88",
    "GREETING": "SCENARIO_ID_05",
    "TASK_ASSIST": "SCENARIO_ID_06",
    "HOSTILE_CONFRONTATION": "SCENARIO_ID_99",
    "DISTRESSED_STUDENT_QUERY": "SCENARIO_ID_74",
    "UNKNOWN": "SCENARIO_ID_00",
}


class PolicyMapper:
    """
    Maps MCN intent classification outputs to ROS2 behavioral policy JSON.

    Usage:
        mapper = PolicyMapper()
        policy_json = mapper.map(
            frame_id=1024,
            intent_label="HELP_REQUEST",
            intent_probability=0.94,
        )
    """

    def __init__(self):
        self.policies = BEHAVIORAL_POLICIES
        self.scenario_map = SCENARIO_ID_MAP

    def map(
        self,
        frame_id: int,
        intent_label: str,
        intent_probability: float,
        context_label: Optional[str] = None,
    ) -> Dict:
        """
        Generate the full ROS2 behavioral policy JSON response.

        Args:
            frame_id: Current frame counter.
            intent_label: Predicted intent string (e.g., "HELP_REQUEST").
            intent_probability: Confidence of the prediction [0, 1].
            context_label: Optional environment context for policy adaptation.

        Returns:
            Dict matching the ROS2 Behavioral Policy output schema.
        """
        # Lookup policy (default to UNKNOWN if not found)
        policy = self.policies.get(intent_label, self.policies["UNKNOWN"]).copy()
        scenario_id = self.scenario_map.get(intent_label, "SCENARIO_ID_00")

        # Context-aware policy adaptation
        if context_label:
            policy = self._adapt_to_context(policy, intent_label, context_label)

        # Remove internal fields
        policy.pop("priority", None)
        policy.pop("description", None)

        return {
            "frame_id": frame_id,
            "predicted_intent": intent_label,
            "intent_probability": round(intent_probability, 4),
            "scenario_id": scenario_id,
            "behavioral_policy": policy,
        }

    def _adapt_to_context(
        self, policy: Dict, intent: str, context: str
    ) -> Dict:
        """
        Adapt behavioral policy based on environmental context.

        Example adaptations:
        - Emergency in Narrow Hallway → yield to wall
        - Emergency in Open Lobby → increase buffer zone
        - GIVE_WAY in Narrow path → slow down more
        """
        if intent == "EMERGENCY":
            if context in ("Narrow Hallway", "Narrow path"):
                policy["proxemic_action"] = "YIELD_TO_WALL"
                policy["target_linear_velocity"] = 0.0
                policy["social_buffer_zone_radius"] = 1.50
            elif context in ("Open Lobby", "Open area"):
                policy["social_buffer_zone_radius"] = 4.00

        elif intent == "GIVE_WAY":
            if context in ("Narrow Hallway", "Narrow path", "Doorway"):
                policy["target_linear_velocity"] = 0.10
                policy["social_buffer_zone_radius"] = 1.00

        elif intent in ("HELP_REQUEST", "DISTRESSED_STUDENT_QUERY"):
            if context in ("Hospital", "Clinic"):
                policy["vocal_affect_tone"] = "CALM_MEDICAL"
                policy["social_buffer_zone_radius"] = 1.80

        return policy

    def map_from_index(
        self,
        frame_id: int,
        intent_index: int,
        intent_probability: float,
        context_label: Optional[str] = None,
    ) -> Dict:
        """
        Convenience method that takes an integer intent index
        (from model output argmax) instead of a string label.
        """
        intent_label = INTENT_IDX2STR.get(intent_index, "UNKNOWN")
        return self.map(frame_id, intent_label, intent_probability, context_label)

    def to_json(self, policy_dict: Dict, indent: int = 2) -> str:
        """Serialize the policy dict to a JSON string."""
        return json.dumps(policy_dict, indent=indent)
