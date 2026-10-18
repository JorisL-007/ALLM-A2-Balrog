"""E3 follow-up: one-call ReflAct-inspired goal-state line, without a critic."""

from exercises.exercise2_planning.agents.grounded_planning_v1 import GroundedPlanningAgent


class GoalStateReflectionAgent(GroundedPlanningAgent):
    state_line = "STATE_VS_GOAL: <current observed state; unmet mission requirement; whether the previous action advanced it>\n"
    line_count = "five"
