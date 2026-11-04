"""Causal keyframe selectors, one module per selector variant.

Each module is self-contained and is the selector the task's training memory was built with; ``SELECTOR_CLASS`` (the last
assignment in the module) is the class used online. ``causal`` is ``CausalSelector`` itself (evidence intake, event
ranking); the other variants extend it with task-generic rules:

    causal        object-event keyframes (the SIMPLE tasks)
    flex          adaptive keyframes (BlinkPad, TouchPad, LitLamp)
    flex_flash    + a lower mask-change threshold, so each flash of a lamp becomes an event (FlashPad)
    flex_hidden   + hidden-object memory: the last clear view of an occluded object is kept in memory and in slot 0 (HideMug)
    flex_binding  + container binding: follows the container that holds an object while containers swap (ShellGame)
"""
from __future__ import annotations

import importlib

VARIANTS = ("causal", "flex", "flex_flash", "flex_hidden", "flex_binding")


def load_selector_class(name: str) -> type:
    if name not in VARIANTS:
        raise KeyError(f"unknown selector {name!r}; choose one of {VARIANTS}")
    return importlib.import_module(f"{__name__}.{name}").SELECTOR_CLASS
