

"""Metric Distillation config."""
import dataclasses
from typing import Optional, Union, Tuple

from acme.adders import reverb as adders_reverb


@dataclasses.dataclass
class MetricDistillationConfig:
    """Configuration options for Metric Distillation."""

    env_name: str = ''
    max_number_of_steps: int = 1_000_000
    num_actors: int = 4

    batch_size: int = 256
    contrastive_learning_rate: float = 3e-4
    quasimetric_learning_rate: float = 3e-4
    critic_learning_rate: float = 3e-4
    actor_learning_rate: float = 3e-4
    discount: float = 0.99
    tau: float = 0.005
    hidden_layer_sizes: Tuple[int, ...] = (512, 512, 512, 512)

    min_replay_size: int = 10000
    max_replay_size: int = 1000000
    replay_table_name: str = adders_reverb.DEFAULT_PRIORITY_TABLE
    prefetch_size: int = 4
    num_parallel_calls: Optional[int] = 4
    samples_per_insert: float = 256

    samples_per_insert_tolerance_rate: float = 0.1
    num_sgd_steps_per_step: int = 64

    repr_dim: Union[int, str] = 64
    use_random_actor: bool = True
    repr_norm: bool = True
    repr_norm_temp: float = 0.01
    adaptive_repr_norm_temp: bool = True
    local: bool = False
    twin_q: bool = False
    use_cpc: bool = True
    use_nce: bool = False
    use_gcbc: bool = False
    use_image_obs: bool = False
    random_goals: float = 1.0
    jit: bool = True
    bc_coef: float = 0.0
    bc_loss: str = 'mse'
    lam_init: float = 1.0
    lam_scale: float = 1.0
    margin: float = 0.35
    quasimetric: str = 'max'
    quasimetric_hidden_dim: int = 256
    quasimetric_num_groups: int = 32
    quasimetric_action_constraint: bool = True
    quasimetric_action_constraint_coef: float = 1e-5
    contrastive_only: bool = False
    triangle_ineq_coef: float = 0.0
    awr: bool = False
    awr_temp: float = 1.0

    obs_dim: int = -1
    max_episode_steps: int = -1
    start_index: int = 0
    end_index: int = -1
