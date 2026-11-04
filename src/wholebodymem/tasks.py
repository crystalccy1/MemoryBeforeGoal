"""The tasks and the memory settings each WholeBodyMem checkpoint was trained with.

``memory``: the six simulated HuMeGo memory tasks. ``simple``: the 14 SIMPLE tasks (T1-T14).

The values below must match the training run of a checkpoint: the SAM3 prompt(s), the object-id namespace of the training
tracks, the declared video length and the keyframe selector. They are not tuning knobs.
"""
from __future__ import annotations

from dataclasses import dataclass

# Selector settings shared by all tasks (the values the training object memory was built with).
SELECTOR_KWARGS = {"max_frames": 7, "event_fraction": 0.75, "event_min_gap": 5, "score_threshold": 0.35}
# SAM3 is told the episode has horizon + N_DECL_MARGIN frames.
N_DECL_MARGIN = 64


@dataclass(frozen=True)
class TaskSpec:
    name: str
    env_id: str                  # SimFactory environment of the closed-loop evaluation
    dataset: str                 # training data on the Hugging Face Hub
    sam3_prompt: str             # SAM3 text prompt; "|" separates several prompts (one SAM3 session each)
    track_namespace: str         # object-id namespace of the training tracks ("<namespace>/<episode:06d>/<prompt_idx>:<id>")
    horizon: int                 # maximum episode length in control steps (50 Hz)
    selector: str                # module in wholebodymem.selectors
    # None: object slots by first appearance. A tuple (possibly empty): the selector assigns the object slots
    # (container binding / hidden object) and receives these container prompts.
    container_prompts: tuple[str, ...] | None = None
    suite: str = "memory"        # folder of the checkpoint in the model repository: wholebodymem/<suite>/<name>

    @property
    def n_decl(self) -> int:
        return self.horizon + N_DECL_MARGIN

    @property
    def selector_slots(self) -> bool:
        return self.container_prompts is not None

    @property
    def selector_kwargs(self) -> dict:
        kwargs = dict(SELECTOR_KWARGS)
        if self.container_prompts is not None:
            kwargs["container_prompts"] = list(self.container_prompts)
        return kwargs


TASKS = {
    "BlinkPad": TaskSpec("BlinkPad", "SimFactoryBlinkPad2T-v0", "ChenyuanC/HuMeGo-Sim-BlinkPad",
                         "bright yellow rectangle", "blinkpad2t", 1500, "flex"),
    "TouchPad": TaskSpec("TouchPad", "SimFactoryBlinkPad2TSingle-v0", "ChenyuanC/HuMeGo-Sim-TouchPad",
                         "yellow square", "blinkpad", 1700, "flex"),
    "FlashPad": TaskSpec("FlashPad", "SimFactoryFlashPad2T-v0", "ChenyuanC/HuMeGo-Sim-FlashPad",
                         "yellow cylinder", "flashpad2tx", 1700, "flex_flash"),
    "LitLamp": TaskSpec("LitLamp", "SimFactoryLitLamp2T-v0", "ChenyuanC/HuMeGo-Sim-LitLamp",
                        "yellow cylinder", "litlamp2tx", 1500, "flex"),
    "HideMug": TaskSpec("HideMug", "SimFactoryHideMug2T-v0", "ChenyuanC/HuMeGo-Sim-HideMug",
                        "red mug", "hidemug2txr", 1700, "flex_hidden", container_prompts=()),
    "ShellGame": TaskSpec("ShellGame", "SimFactoryShellGame2T-v0", "ChenyuanC/HuMeGo-Sim-ShellGame",
                          "blue cup|red ball", "shellgame2tcfb", 1700, "flex_binding", container_prompts=("blue cup",)),
}

# SIMPLE: training data = USC-PSI-Lab/psi-data simple/<env_id>.zip (LeRobot layout inside) at this revision.
SIMPLE_DATA = "USC-PSI-Lab/psi-data"
SIMPLE_DATA_REVISION = "92da00fe58b90ecfaaa30a096ff6b9b9090feb26"
for _name, _env, _prompt, _namespace, _horizon in (
        ("T1_XMovePick", "G1WholebodyXMovePickTeleop-v0", "cracker box", "xmove", 800),
        ("T2_BendPickMP", "G1WholebodyBendPickMP-v0", "red box", "bpmp", 400),
        ("T3_Handover", "G1WholebodyHandoverTeleop-v0", "cracker box", "hand", 800),
        ("T4_TabletopGraspMP", "G1WholebodyTabletopGraspMP-v0", "red can", "ttg", 800),
        ("T5_XMoveBendPick", "G1WholebodyXMoveBendPickTeleop-v0", "cracker box", "bendpick", 800),
        ("T6_LocomotionPickBetweenTables", "G1WholebodyLocomotionPickBetweenTablesTeleop-v0", "cracker box", "locomotion", 1200),
        ("T7_PickAndPlaceAndHugContainer", "G1WholebodyPickAndPlaceAndHugContainerTeleop-v0", "basket", "hug", 1000),
        ("T8_OpenFaucet", "G1WholebodyOpenFaucetTeleop-v0", "faucet", "faucet", 1000),
        ("T9_CloseDoor", "G1WholebodyCloseDoorTeleop-v0", "door", "door", 1000),
        ("T10_OpenOven", "G1WholebodyOpenOvenTeleop-v0", "oven door", "oven", 1000),
        ("T11_OpenTrashCan", "G1WholebodyOpenTrashCanTeleop-v0", "green lid", "trash", 800),
        ("T12_PushOfficeChair", "G1WholebodyPushOfficeChairTeleop-v0", "chair backrest", "chair", 800),
        ("T13_BendHandover", "G1WholebodyBendHandoverTeleop-v0", "cracker box", "bht", 800),
        ("T14_BendPick", "G1WholebodyBendPickTeleop-v0", "cracker box", "bpt", 800)):
    TASKS[_name] = TaskSpec(_name, _env, SIMPLE_DATA, _prompt, _namespace, _horizon, "causal", suite="simple")


def get_task(name: str) -> TaskSpec:
    for key, spec in TASKS.items():
        if key.lower() == str(name).lower():
            return spec
    raise KeyError(f"unknown task {name!r}; choose one of {sorted(TASKS)}")
