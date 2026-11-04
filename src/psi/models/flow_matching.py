"""Stateless flow-matching sampling and fixed-point inversion utilities.

The functions in this module intentionally depend only on PyTorch.  A
``FlowSchedule`` can be snapshotted from a Diffusers scheduler, after which
sampling and inversion do not depend on the scheduler's mutable ``step_index``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch


VelocityFunction = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


@dataclass(frozen=True)
class FlowSchedule:
    """Immutable descending-sigma schedule used by Psi0 flow inference."""

    timesteps: torch.Tensor
    sigmas: torch.Tensor

    def __post_init__(self) -> None:
        if self.timesteps.ndim != 1 or self.sigmas.ndim != 1:
            raise ValueError("timesteps and sigmas must be one-dimensional")
        if self.timesteps.numel() == 0:
            raise ValueError("flow schedule must contain at least one step")
        if self.sigmas.numel() != self.timesteps.numel() + 1:
            raise ValueError(
                "flow schedule requires exactly one more sigma than timestep"
            )
        if not torch.isfinite(self.timesteps).all():
            raise ValueError("flow timesteps must be finite")
        if not torch.isfinite(self.sigmas).all():
            raise ValueError("flow sigmas must be finite")
        if not torch.all(self.sigmas[:-1] > self.sigmas[1:]):
            raise ValueError("Psi0 flow sigmas must be strictly decreasing")
        if not torch.isclose(
            self.sigmas[-1],
            torch.zeros_like(self.sigmas[-1]),
            rtol=0,
            atol=1e-8,
        ):
            raise ValueError("Psi0 flow schedule must terminate at sigma=0")

    @property
    def num_steps(self) -> int:
        return int(self.timesteps.numel())

    @classmethod
    def from_scheduler(
        cls,
        scheduler: object,
        num_inference_steps: int,
        *,
        device: torch.device | str | None = None,
    ) -> "FlowSchedule":
        """Snapshot a Diffusers-compatible scheduler without using its step state."""
        if num_inference_steps <= 0:
            raise ValueError("num_inference_steps must be positive")

        scheduler_config = getattr(scheduler, "config", None)
        if bool(
            getattr(scheduler_config, "stochastic_sampling", False)
        ):
            raise ValueError(
                "FlowDAgger requires deterministic flow sampling; "
                "stochastic_sampling must be False"
            )

        # Diffusers schedulers are mutable. Reconstructing from config prevents
        # concurrent sampling/inversion calls from sharing `_step_index`.
        scheduler_type = type(scheduler)
        if hasattr(scheduler_type, "from_config") and hasattr(scheduler, "config"):
            local_scheduler = scheduler_type.from_config(scheduler.config)
        else:
            local_scheduler = scheduler
        local_scheduler.set_timesteps(num_inference_steps)

        timesteps = local_scheduler.timesteps.detach().clone()
        sigmas = local_scheduler.sigmas.detach().clone()
        if device is not None:
            timesteps = timesteps.to(device)
            sigmas = sigmas.to(device)
        return cls(
            timesteps=timesteps.to(dtype=torch.float32),
            sigmas=sigmas.to(dtype=torch.float32),
        )


@dataclass(frozen=True)
class FlowInversionResult:
    """Result and convergence diagnostics from fixed-point inversion."""

    noise: torch.Tensor
    residuals: torch.Tensor
    per_sample_residuals: torch.Tensor


def _validate_flow_tensor(name: str, value: torch.Tensor) -> None:
    if not value.is_floating_point():
        raise ValueError(f"{name} must be a floating-point tensor")
    if not torch.isfinite(value).all():
        raise ValueError(f"{name} must contain only finite values")


def flow_euler_step(
    sample: torch.Tensor,
    velocity: torch.Tensor,
    sigma: torch.Tensor,
    sigma_next: torch.Tensor,
    *,
    validate_finite: bool = True,
) -> torch.Tensor:
    """Apply one Euler step with Diffusers 0.37 FlowMatch cast semantics."""
    if sample.shape != velocity.shape:
        raise ValueError(
            f"sample and velocity shapes differ: {sample.shape} vs {velocity.shape}"
        )
    if sample.device != velocity.device:
        raise ValueError(
            f"sample and velocity devices differ: {sample.device} vs {velocity.device}"
        )
    if validate_finite:
        _validate_flow_tensor("sample", sample)
        _validate_flow_tensor("velocity", velocity)
    elif not sample.is_floating_point() or not velocity.is_floating_point():
        raise ValueError("sample and velocity must be floating-point tensors")

    # Preserve scalar placement from the schedule. Diffusers keeps its sigma
    # table on CPU by default, and CPU scalar × GPU BF16 has measurably
    # different rounding from moving the scalar to ROCm first.
    sigma = torch.as_tensor(sigma, dtype=torch.float32)
    sigma_next = torch.as_tensor(sigma_next, dtype=torch.float32)
    if sigma.numel() != 1 or sigma_next.numel() != 1:
        raise ValueError("sigma and sigma_next must be scalar")
    if not torch.isfinite(sigma) or not torch.isfinite(sigma_next):
        raise ValueError("sigma and sigma_next must be finite")
    if not sigma_next < sigma:
        raise ValueError("flow Euler step requires sigma_next < sigma")

    # Keep `velocity` in its model dtype here. Diffusers multiplies the scalar
    # dt by model_output before adding it to the FP32 sample; explicitly
    # upcasting velocity first changes BF16 rounding on CPU/ROCm.
    updated = sample.to(dtype=torch.float32) + (sigma_next - sigma) * velocity
    return updated.to(dtype=velocity.dtype)


@torch.inference_mode()
def decode_flow(
    initial_noise: torch.Tensor,
    schedule: FlowSchedule,
    velocity_fn: VelocityFunction,
) -> torch.Tensor:
    """Decode initial noise with an immutable, deterministic Euler schedule."""
    _validate_flow_tensor("initial_noise", initial_noise)
    sample = initial_noise.clone()
    for index in range(schedule.num_steps):
        timestep = schedule.timesteps[index].to(sample.device)
        velocity = velocity_fn(sample, timestep)
        if velocity.shape != sample.shape:
            raise ValueError(
                f"velocity shape {velocity.shape} does not match sample {sample.shape}"
            )
        sample = flow_euler_step(
            sample,
            velocity,
            schedule.sigmas[index],
            schedule.sigmas[index + 1],
            validate_finite=False,
        )
    _validate_flow_tensor("decoded sample", sample)
    return sample


@torch.inference_mode()
def invert_flow_fixed_point(
    clean_action: torch.Tensor,
    schedule: FlowSchedule,
    velocity_fn: VelocityFunction,
    *,
    num_fixed_point_steps: int = 5,
) -> FlowInversionResult:
    """Invert a clean action to initial noise using per-step fixed points."""
    if num_fixed_point_steps <= 0:
        raise ValueError("num_fixed_point_steps must be positive")
    _validate_flow_tensor("clean_action", clean_action)
    if clean_action.ndim == 0:
        raise ValueError("clean_action must include a batch dimension")

    x_next = clean_action.to(dtype=torch.float32).clone()
    residuals = torch.empty(
        schedule.num_steps,
        num_fixed_point_steps,
        device=x_next.device,
        dtype=torch.float32,
    )
    per_sample_residuals = torch.empty(
        clean_action.shape[0],
        schedule.num_steps,
        num_fixed_point_steps,
        device=x_next.device,
        dtype=torch.float32,
    )

    for reverse_index, index in enumerate(
        range(schedule.num_steps - 1, -1, -1)
    ):
        timestep = schedule.timesteps[index].to(x_next.device)
        dt = (
            schedule.sigmas[index + 1] - schedule.sigmas[index]
        ).to(device=x_next.device, dtype=torch.float32)
        z = x_next.clone()

        for iteration in range(num_fixed_point_steps):
            velocity = velocity_fn(z, timestep)
            if velocity.shape != z.shape:
                raise ValueError(
                    f"velocity shape {velocity.shape} does not match sample {z.shape}"
                )
            if not velocity.is_floating_point():
                raise ValueError("velocity must be a floating-point tensor")
            updated = x_next - dt * velocity.to(dtype=torch.float32)
            sample_residuals = (
                (updated - z).square().reshape(clean_action.shape[0], -1)
            ).mean(dim=1)
            per_sample_residuals[:, reverse_index, iteration] = (
                sample_residuals
            )
            residuals[reverse_index, iteration] = sample_residuals.mean()
            z = updated

        x_next = z

    _validate_flow_tensor("inverted noise", x_next)
    _validate_flow_tensor("fixed-point residuals", residuals)
    _validate_flow_tensor(
        "per-sample fixed-point residuals", per_sample_residuals
    )
    return FlowInversionResult(
        noise=x_next,
        residuals=residuals,
        per_sample_residuals=per_sample_residuals,
    )
