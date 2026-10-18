"""Explicit protocols in chronological order. History length is a case setting."""
from pathlib import Path
import json

ROOT = Path(__file__).resolve().parents[2]
MODEL = "Qwen/Qwen3-4B-Instruct-2507-FP8"
MODELS = {"qwen4-instruct": (MODEL, 16384),
          "qwen14-awq": ("Qwen/Qwen3-14B-AWQ", 8192),
          "phi4-instruct": ("microsoft/Phi-4-mini-instruct", 8192)}
TASKS = ["goto", "pickup", "open", "putnext", "pick_up_seq_go_to"]
CONDITIONS = {
    "naive": ("naive", "planning", 16, 0, 0, 0, 256),
    "plan": ("custom", "planning", 16, 0, 0, 0, 256),
    "e3": ("custom", "reflect", 16, 0, 0, 0, 256),
    "guide_h4": ("custom", "planning", 4, 1, 0, 0, 256),
    "guide": ("custom", "planning", 16, 1, 0, 0, 256),
    "guide_h32": ("custom", "planning", 32, 1, 0, 0, 256),
    "guide_gate": ("custom", "planning", 16, 1, 1, 0, 256),
    "guide_goal": ("custom", "planning", 16, 1, 0, 1, 256),
}
def panel_cases(panel):
    if panel == "pilot":
        for seed in range(20, 25):
            for task in ("goto", "pickup", "open"):
                for condition in ("plan", "guide", "guide_gate", "guide_goal", "guide_h32"):
                    yield "babyai", task, seed, condition, 32, 0
    if panel == "primary":
        for seed in range(30, 90):
            for task in ("goto", "pickup", "open"):
                for condition in CONDITIONS:
                    yield "babyai", task, seed, condition, 32, 0
    if panel == "broad":
        for seed in range(100, 112):
            for task in TASKS:
                for condition in ("plan", "guide", "guide_gate", "guide_goal"):
                    yield "babyai", task, seed, condition, 128, 0
    if panel == "crafter":
        for seed in range(120, 130):
            for condition in ("plan", "guide_goal"):
                yield "crafter", "default", seed, condition, 50, 0
    if panel == "repeat":
        for seed in range(30, 35):
            for task in ("goto", "pickup"):
                for condition in ("plan", "guide_gate"):
                    for repeat in range(1, 4):
                        yield "babyai", task, seed, condition, 32, repeat

VARIANTS = {
    "plan": "compact_planning_frozen_v1", "e3": "action_critique_v2",
    "guide": "grounded_planning_v1", "guide_h4": "grounded_planning_v1",
    "guide_h32": "grounded_planning_v1", "guide_gate": "grounded_planning_v1",
    "guide_goal": "goal_state_reflection_v1",
}
STUDIES = ("e1_naive", "e1_cot", "e2_development", "e2_evaluation", "e3_development",
           "e3_evaluation", "followup_pilot", "followup_evaluation", "smoke", "model_transfer_v1")

TRANSFER_MODELS = ("qwen4-instruct", "gpt41mini-cloud", "olmo1b-local")
OLMO_MODEL = "allenai/OLMo-2-0425-1B-Instruct"
OLMO_REVISION = "48d788eca847d4d7548f375ad03d3c9312f6139e"
OLMO_GGUF_REVISION = "62f8c199538474c3e33ed5d7e0580abd66686a27"


def transfer_case(label, environment, task, seed, condition, budget, pilot=False):
    item = case("preflight" if pilot else "primary", environment, task, seed, condition, budget)
    if condition == "cot":
        item.update(agent="cot", variant="")
    item.update(transfer=True, model_label=label, backend="openai" if label == "gpt41mini-cloud" else "vllm",
                episode_timeout=1800, pilot=pilot, reuse_from="")
    if label == "gpt41mini-cloud":
        item.update(model="gpt-4.1-mini-2025-04-14", context=8192)
    elif label == "olmo1b-local":
        item.update(model=OLMO_MODEL, context=4096)
    elif not pilot and condition != "cot" and (environment == "babyai" or condition == "plan"):
        item["reuse_from"] = (f"experiment_results/followup_v1/episodes/{environment}/{task}/"
                              f"{condition}/seed_{seed:03d}/rep_0")
    return item


def transfer_pilots(label):
    # Excluded development seeds; all actors and both environments are exercised.
    values = [("babyai", "goto", "naive", 20), ("babyai", "pickup", "cot", 20),
              ("babyai", "open", "plan", 20), ("babyai", "open", "e3", 21),
              ("crafter", "default", "cot", 20), ("crafter", "default", "e3", 21)]
    return [transfer_case(label, env, task, seed, condition, 8, pilot=True)
            for env, task, condition, seed in values]


def transfer_cases():
    cases = []
    for label in TRANSFER_MODELS:
        for seed in range(30, 90):
            for task in ("goto", "pickup", "open"):
                for condition in ("naive", "cot", "plan", "e3"):
                    if condition != "cot" or seed % 2 == 0:
                        cases.append(transfer_case(label, "babyai", task, seed, condition, 32))
        for seed in range(120, 130):
            for condition in ("naive", "cot", "plan", "e3"):
                if condition != "cot" or seed % 2 == 0:
                    cases.append(transfer_case(label, "crafter", "default", seed, condition, 50))
    return cases


def case(panel, environment, task, seed, condition, budget, repeat=0):
    agent, mode, history, guide, gate, goal, tokens = CONDITIONS.get(condition, CONDITIONS["plan"])
    return dict(panel=panel, environment=environment, task=task, seed=seed,
                condition=condition, budget=budget, repeat=repeat, agent=agent,
                variant=VARIANTS.get(condition, "compact_planning_frozen_v1"),
                history=history, guide=guide, gate=gate, goal=goal, tokens=tokens,
                model=MODEL, model_label="qwen4-instruct", context=8192, temperature=0, workers=1, gpu_memory_utilization=0.88)


def study_cases(study):
    if study == "model_transfer_v1":
        return transfer_cases()
    if study.startswith("followup"):
        panels = ("pilot",) if study == "followup_pilot" else ("primary", "broad", "crafter", "repeat")
        return [case(panel, *values) for panel in panels for values in panel_cases(panel)]
    if study == "smoke":
        # Pilot seed 24 is a documented pickup substitution; keep the original cap.
        values = [("plan", "babyai", "pickup", 20, 32), ("e3", "babyai", "goto", 30, 32),
                  ("guide", "babyai", "pickup", 20, 32), ("guide_gate", "babyai", "pickup", 24, 32),
                  ("guide_goal", "babyai", "pickup", 20, 32), ("guide_h4", "babyai", "open", 20, 32),
                  ("guide_h32", "babyai", "pickup", 20, 32), ("guide_goal", "crafter", "default", 120, 50)]
        return [case("smoke", env, task, seed, condition, budget) for condition, env, task, seed, budget in values]
    if study.startswith("e1_"):
        agent = "naive" if study == "e1_naive" else "cot"
        labels = MODELS if agent == "naive" else ["qwen4-instruct"]
        cases = []
        for label in labels:
            model, context = MODELS[label]
            for repeat in (1, 2, 3):
                for env, count in (("babyai", 10), ("crafter", 10), ("babaisai", 3)):
                    for seed in range(1, count + 1):
                        item = case(study, env, "all", seed, agent, None, repeat)
                        item.update(agent=agent, variant="", model=model, model_label=label,
                                    context=context, temperature=None, workers=4, gpu_memory_utilization=0.90,
                                    tokens=128 if agent == "naive" else 512)
                        cases.append(item)
        return cases
    # Historical panels retain their original tags and numerical settings.
    tags = {
        "e2_development": ["e2_supplied", "e2_compact_v1", "cot_reference"],
        "e3_development": ["e3_off_dev2", "e3_on_dev2",
                           "e3_on_dev3", "e3_off_open_dev", "e3_on_open_dev", "e3_on_open_dev2"],
        "e2_evaluation": ["heldout_qwen_naive", "heldout_qwen_plan", "transfer_qwen_naive", "transfer_qwen_plan",
                          "cross_qwen14_naive", "cross_qwen14_plan", "cross_phi4_naive", "cross_phi4_plan"],
        "e3_evaluation": ["heldout_qwen_plan", "heldout_qwen_reflect", "transfer_qwen_plan", "transfer_qwen_reflect",
                          "cross_qwen14_plan", "cross_qwen14_reflect", "cross_phi4_plan", "cross_phi4_reflect"],
    }[study]
    profiles = json.loads((Path(__file__).with_name("historical_profiles.json")).read_text())
    cases = []
    for tag in tags:
        settings = profiles[tag]
        for seed in settings["seeds"]:
            for task in settings["tasks"]:
                item = case(study, settings["environment"], task, seed, tag, settings["budget"])
                item.update({key: value for key, value in settings.items() if key not in ("seeds", "tasks", "environment", "budget")})
                cases.append(item)
    return cases
