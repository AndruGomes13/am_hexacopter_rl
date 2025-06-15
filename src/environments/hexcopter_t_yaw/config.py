# config.py
import math
from environments.hexcopter_t_yaw.action_models import ActionModelType
from environments.hexcopter_t_yaw.observation_models import ActorObservationType, CriticObservationType
import numpy as np
from pydantic import BaseModel, Field, field_validator, model_validator, validator
from typing import Literal, Optional, Self, Sequence, Tuple, List, Union
import json
from datetime import datetime
from pathlib import Path
import jax.numpy as jp

# --- Constants ---
DEG = np.pi / 180
G = 9.81
HOVER_THRUST_PER_PROP = 1.32 * G / 4
# --- Config Definition ---
FloatRange = Tuple[float, float]


class Schedule(BaseModel):
    kind: Literal["const", "const_e", "linear", "exp", "zero"]
    start: Optional[Union[float, Sequence[float]]] = None
    end: Optional[Union[float, Sequence[float]]] = None
    p_range: FloatRange = (0.0, 1.0)  # progress ∈ [s,e]

    @model_validator(mode="after")
    def _validate_and_build_fn(self) -> Self:
        s, e = self.p_range
        # --- Check valid range ---
        if not 0.0 <= s < e <= 1.0:
            raise ValueError("p_range must satisfy 0.0 ≤ start < end ≤ 1.0")

        # --- Check start/end are correctly set
        if self.kind in ["linear", "exp"]:
            if self.start is None:
                raise ValueError(f"Start must be assigned for {self.kind} schedule.")
            if self.end is None:
                raise ValueError(f"End must be assigned for {self.kind} schedule.")
            self._start = jp.array(self.start, dtype=jp.float32)
            self._end = jp.array(self.end, dtype=jp.float32)

        if self.kind == "const":
            if self.start is None:
                raise ValueError(f"Start must be assigned for {self.kind} schedule.")
            self._start = jp.array(self.start, dtype=jp.float32)
            self._end = self._start

        if self.kind == "const_e":
            if self.end is None:
                raise ValueError(f"End must be assigned for {self.kind} schedule.")
            self._end = jp.array(self.end, dtype=jp.float32)
            self._start = self._end

        if self.kind == "zero":
            if self.end is None and self.start is None:
                raise ValueError(f"Either start or end must be assigned for {self.kind} schedule.")
            if self.start is not None:
                self._start = jp.array(self.start, dtype=jp.float32)
                self._end = self._start
            elif self.end is not None:
                self._end = jp.array(object=self.end, dtype=jp.float32)
                self._start = self._end

        if self._start.shape != self._end.shape:
            raise ValueError("start/end must have the same shape for linear and exp")

        # --- Check start/end values are valid
        if self.kind == "exp":
            if jp.any(self._start * self._end < 0):
                raise ValueError(f"Start and end must have the same sign (got start={self.start}, end={self.end})")
            if jp.any(self._start == 0).any() or jp.any(self._end == 0).any():
                raise ValueError("Start and end must be different than zero.")

        # --- Precompute some things ---
        self._inv_span = 1.0 / (e - s)

        # --- Create function ---
        if self.kind == "linear":
            fn = self._linear
        elif self.kind == "exp":
            fn = self._exponential
        elif self.kind == "zero":
            fn = self._zero
        elif self.kind == "const":
            fn = self._constant_s
        elif self.kind == "const_e":
            fn = self._constant_e
        self._fn = fn
        return self

    def _p_norm(self, p: jp.ndarray) -> jp.ndarray:
        s, e = self.p_range
        return (jp.clip(p, s, e) - s) * self._inv_span

    def _linear(self, p: jp.ndarray) -> jp.ndarray:
        p_n = self._p_norm(p)
        return self._start + p_n * (self._end - self._start)

    def _exponential(self, p: jp.ndarray) -> jp.ndarray:
        p_n = self._p_norm(p)
        mag_vs, mag_ve = jp.abs(self._start), jp.abs(self._end)
        ratio = mag_ve / mag_vs
        mag_val = mag_vs * (ratio**p_n)
        return jp.sign(self._start) * mag_val

    def _zero(self, p: jp.ndarray) -> jp.ndarray:
        return jp.zeros_like(self._start, dtype=jp.float32)

    def _constant_s(self, p: jp.ndarray) -> jp.ndarray:
        return self._start

    def _constant_e(self, p: jp.ndarray) -> jp.ndarray:
        return self._end

    def __call__(self, p: jp.ndarray) -> jp.ndarray:
        return self._fn(p)


class BoolSchedule(BaseModel):
    """
    A boolean gate that either stays constant or flips once.
    """

    start: bool = False
    switch_at: Optional[float] = None

    @model_validator(mode="after")
    def _build_fn(self) -> "BoolSchedule":
        if self.switch_at is None:
            self._start_val = self.start
            self._fn = self._constant
        else:
            if not (0.0 <= self.switch_at <= 1.0):
                raise ValueError("`switch_at` must lie between 0.0 and 1.0")
            self._start_val = self.start
            self._fn = self._threshold

        self._start_val = jp.asarray(self._start_val, dtype=jp.bool_)
        return self

    def _constant(self, p=None):
        # ignore p entirely
        return self._start_val

    def _threshold(self, p):
        # p can be scalar or array; returns same shape
        return jp.where(p < self.switch_at, self._start_val, jp.logical_not(self._start_val))

    def __call__(self, p=None):
        if self.switch_at is not None and p is None:
            raise ValueError("`p` must be provided when `switch_at` is set")
        return self._fn(p)


ACTION_REPEAT: int = 1


class DomainRandomizationConfig(BaseModel):
    use_domain_randomization: bool = False

    # --- At reset ---
    # NOTE: These values will be randomizes at the start of each episode. These can be progress dependent.
    # Aerodynamics
    density_mult: Schedule = Schedule(kind="const_e", start=(0.01, 0.01), end=(0.8, 1.2), p_range=(0.7, 1.0))
    viscosity_mult: Schedule = Schedule(kind="const", start=(0.9, 1.1))
    thrust_map_coefficients_mult: Schedule = Schedule(kind="const", start=(1.0, 1.0))

    # --- At environement setup ---
    # NOTE: These values will be randomizes at the start of the environment (will be fixed during the whole training). These can't be progress dependent.
    drone_mass_mult_range: FloatRange = (0.95, 1.05) # Multiplier
    contact_solref_damping_range: FloatRange = (-10.0, -15.0)
    gear_variation_range: Tuple[float, ...] = (0.05, 0.05, 0.05, 0., 0., 0.001)  # Gear variation for each propeller. Original gear: "0 0 1 0 0  0.022"


class StageConfig(BaseModel):
    # --- Reward weights ---
    # --- Drone
    reward_drone_survive: Schedule = Schedule(kind="const", start=2.0)
    reward_drone_dist_to_target: Schedule = Schedule(kind="zero", start=0.0)
    reward_drone_ang_from_v: Schedule = Schedule(kind="zero", start=0.0)  # const
    reward_drone_ang_from_des: Schedule = Schedule(kind="zero", start=0.0)  # const
    reward_drone_vel: Schedule = Schedule(kind="zero", start=0.0)
    reward_drone_ang_vel: Schedule = Schedule(kind="zero", start=0.0)
    reward_drone_yaw_rate: Schedule = Schedule(kind="zero", start=0.0)
    reward_drone_action: Schedule = Schedule(kind="zero", start=0.0)
    reward_drone_action_change: Schedule = Schedule(kind="zero", start=0.0)
    reward_drone_action_deviation: Schedule = Schedule(kind="zero", start=0.0)
    reward_drone_outside_playground: Schedule = Schedule(kind="zero", start=0.0)

    # --- End effector ---
    reward_end_effector_dist_to_target: Schedule = Schedule(kind="zero", start=0.0)

    # NOTE: At the moment, this is not used. "reset_drone_pos_start" is used as a target.
    reward_drone_target_position: Tuple[float, float, float] = (0, 0, 0)

    # --- Zones
    zones_drone_playzone_min: Schedule = Schedule(kind="const", start=(-2, -2.0, -2), p_range=(0.15, 0.5))
    zones_drone_playzone_max: Schedule = Schedule(kind="const", start=(2, 2, 2), p_range=(0.15, 0.5))

    # --- Drone Reset Config ---
    # --- Mean values
    reset_drone_pos_start: Tuple[float, float, float] = (0.0, 0.0, 0.0)

    # --- Randomisation
    reset_drone_orientation_rand: Schedule = Schedule(kind="zero", start=0.0)
    reset_drone_ang_vel_rand: Schedule = Schedule(kind="zero", start=0.0)
    reset_drone_pos_rand: Schedule = Schedule(kind="zero", start=(0.0, 0.0, 0.0))
    reset_drone_vel_rand: Schedule = Schedule(kind="zero", start=(0.0, 0.0, 0.0))
    reset_drone_ori_target: Schedule = Schedule(kind="zero", start=np.deg2rad(10)) 
    env_rand: DomainRandomizationConfig = Field(default_factory=DomainRandomizationConfig)


# --- The curriculum steps for gameplay ---
stage_config_curriculum_0 = StageConfig(
    # --- Reward ---
    # --- Drone
    reward_drone_survive=Schedule(kind="const", start=2.0),
    reward_drone_dist_to_target=Schedule(kind="const", start=-0.25, end=-1.5, p_range=(0, 0.3)),
    # reward_drone_ang_from_v=Schedule(kind="const", start=-0.1),  # const
    reward_drone_ang_from_des=Schedule(kind="const", start=-1.0),
    reward_drone_vel=Schedule(kind="exp", start=-0.005, end=-0.1, p_range=(0.0, 0.3)),
    reward_drone_ang_vel=Schedule(kind="exp", start=-0.002, end=-0.1, p_range=(0.0, 0.3)),
    reward_drone_action=Schedule(kind="exp", start=-0.002, end=-0.1, p_range=(0.0, 0.4)),
    reward_drone_action_change=Schedule(kind="exp", start=-0.002, end=-0.1, p_range=(0.0, 0.4)),
    reward_drone_action_deviation=Schedule(kind="exp", start=-0.002, end=-0.2, p_range=(0.0, 0.4)),
    reward_drone_outside_playground=Schedule(kind="const", start=-20.0 * ACTION_REPEAT),
    # --- End effector ---
    reward_end_effector_dist_to_target=Schedule(kind="const", start=-0.1, end=-1.0, p_range=(0.0, 0.3)),

    # # --- Radomization ---
    reset_drone_orientation_rand=Schedule(kind="const", start=90 * DEG, p_range=(0.0, 0.4)),
    reset_drone_ang_vel_rand=Schedule(kind="const", start=50 * DEG, p_range=(0.0, 0.4)),
    reset_drone_pos_rand=Schedule(kind="const", start=(1., 1., 1.), p_range=(0.4, 0.8)),
    reset_drone_vel_rand=Schedule(kind="const", start=(0.3, 0.3, 0.3), p_range=(0.4, 0.8)),
    reset_drone_ori_target = Schedule(kind="const", start=np.deg2rad(10)) ,
   
    env_rand=DomainRandomizationConfig(
        use_domain_randomization=False,
        density_mult=Schedule(kind="const", start=(0.0, 0.0), end=(0.9, 1.1), p_range=(0.6, 0.9)),
        viscosity_mult=Schedule(kind="const", start=(0.0, 0.0), end=(0.9, 1.1), p_range=(0.6, 0.9)),
    ),
    
    zones_drone_playzone_min=Schedule(kind="const", start=(-2, -2.0, -2)),
    zones_drone_playzone_max=Schedule(kind="const", start=(2, 2, 2.0)),
)
# -----------------------------------------


class EnvConfig(BaseModel):
    # --- Sim config ---
    backend: str = "mjx"
    physics_steps_per_control_step: int = 1
    sim_time_limit: float = 100.0
    env_name: str = "hexcopter_env"
    opt_timestep: float = 1 / 100 / ACTION_REPEAT

    # stage_config: StageConfig = StageConfig()
    stage_config: StageConfig = stage_config_curriculum_0
    # --- Obs/Act model ---
    actor_observation_model: ActorObservationType = ActorObservationType.FULL_HIST
    critic_observation_model: CriticObservationType = CriticObservationType.FULL
    # observation_model: ObservationType = ObservationType.FULL
    action_model: ActionModelType = ActionModelType.F_WZ_JOINT_POS

    # --- History sizes ---
    action_history_len: int = 4
    sensor_data_history_len: int = 4

    # --- Actuator Dynamics ---
    rotor_tau: float = 0.033
    action_delay: float = 0.05
    thrust_map_coefficients: tuple[float, float, float] = (1.562522e-6, 0.0, 0.0)  # Coefficients for thrust map polynomial

    # --- Arm Dynamics ---
    arm_pitch_1_limits: FloatRange = (0.0, np.deg2rad(180))  # Pitch 1 limits in radians
    arm_pitch_2_limits: FloatRange = (-np.deg2rad(10), np.deg2rad(45))  # Pitch 1 limits in radians

    # --- Auxiliary ---
    action_buffer_size: int = 0  # NOTE: This value has no effect. It will be overwriden depending on action history length and action delay.
    action_repeat : int = ACTION_REPEAT  
    action_delay_discrete:int  = 0 # NOTE: This value has no effect. It will be overwriden depending on action history length and action delay.

    def __init__(self, **data):
        super().__init__(**data)
        
        # Calculate action buffer size
        self.action_delay_discrete = int(math.ceil(self.action_delay / (self.opt_timestep * self.action_repeat)))
        self.action_buffer_size = max(self.action_history_len, self.action_delay_discrete)


class NetworkConfig(BaseModel):
    policy_hidden_layer_sizes: Tuple[int, ...] = (128, 128)
    value_hidden_layer_sizes: Tuple[int, ...] = (128, 128)
    # Add other network params if needed, e.g., activation function


class TrainConfig(BaseModel):
    num_timesteps: int = 100_000_000 * ACTION_REPEAT
    num_evals: int = 10  # NOTE: This value has no effect. It will be overwriden further.
    episode_length: int = 1000 * ACTION_REPEAT
    num_envs: int = 3072 // 1  # Parallel environments
    batch_size: int = 512 // 1
    num_minibatches: int = 24
    num_updates_per_batch: int = 8
    unroll_length: int = 128
    learning_rate: float = 3e-3
    entropy_cost: float = 1e-4  # NOTE
    discounting: float = 0.97 ** (1 / ACTION_REPEAT)
    normalize_observations: bool = True
    seed: int = 0
    reward_scaling: float = 1 / ACTION_REPEAT
    clip_epsilon: float = 0.2
    gae_lambda: float = 0.95 ** (1 / ACTION_REPEAT)
    action_repeat: int = ACTION_REPEAT
    restore_value_fn: bool = True

    def __init__(self, **data):
        super().__init__(**data)

        # Basic validation checks
        if (self.num_envs * self.unroll_length) % self.batch_size != 0:
            print(
                f"Warning: (num_envs * unroll_length) ({self.num_envs * self.unroll_length}) is not perfectly divisible by batch_size ({self.batch_size})."
            )
        if self.batch_size % self.num_minibatches != 0:
            print(f"Warning: batch_size ({self.batch_size}) is not perfectly divisible by num_minibatches ({self.num_minibatches}).")


class LoggingConfig(BaseModel):
    log_dir_base: str = "logs_orbax"
    checkpoint_dir_base: str = "runs"
    run_name_prefix: str = "brax_ppo"
    tensorboard_interval_steps: int = 4_000_000 * ACTION_REPEAT  # How often to save checkpoints
    max_checkpoints_to_keep: int = 3  # Keep last N checkpoints


class ExperimentConfig(BaseModel):
    """Main configuration container."""

    env: EnvConfig = Field(default_factory=EnvConfig)
    network: NetworkConfig = Field(default_factory=NetworkConfig)
    train: TrainConfig = Field(default_factory=TrainConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    timestamp: str = Field(default_factory=lambda: datetime.now().strftime("%Y%m%d_%H%M%S"))
    run_name: Optional[str] = Field(default=None, description="Generated run name")  # Will be set after init

    def __init__(self, **data):
        super().__init__(**data)
        # Calculate dependent fields
        self.run_name = f"{self.logging.run_name_prefix}_{self.timestamp}"
        self.train.num_evals = self.train.num_timesteps // self.logging.tensorboard_interval_steps


# --- Helper Functions ---


def save_config(config: ExperimentConfig, path: Path):
    """Saves the config model to a JSON file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(config.model_dump(), f, indent=4)


def load_config(path: Path) -> ExperimentConfig:
    """Loads the config model from a JSON file."""
    if not path.exists():
        raise FileNotFoundError(f"Config file not found at {path}")
    with open(path, "r") as f:
        config_dict = json.load(f)
    return ExperimentConfig(**config_dict)
