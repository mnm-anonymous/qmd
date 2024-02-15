

"""Program definition for a distributed layout based on a builder."""

import dataclasses
import logging
from typing import Any, Callable, Optional, Sequence

from acme import core
from acme import environment_loop
from acme import specs
from acme.agents.jax import builders
from acme.jax import networks as networks_lib
from acme.jax import savers
from acme.jax import types
from acme.jax import utils
from acme.utils import counting
from acme.utils import loggers
from acme.utils import lp_utils
from acme.utils import observers as observers_lib
import dm_env
import jax
import launchpad as lp
import numpy as np
import reverb
import tqdm

ActorId = int
AgentNetwork = Any
PolicyNetwork = Any
NetworkFactory = Callable[[specs.EnvironmentSpec], AgentNetwork]
PolicyFactory = Callable[[AgentNetwork], PolicyNetwork]
Seed = int
EnvironmentFactory = Callable[[Seed], dm_env.Environment]
MakeActorFn = Callable[[types.PRNGKey, PolicyNetwork, core.VariableSource],
                       core.Actor]
LoggerLabel = str
LoggerStepsKey = str
LoggerFn = Callable[[LoggerLabel, LoggerStepsKey], loggers.Logger]
EvaluatorFactory = Callable[[
    types.PRNGKey,
    core.VariableSource,
    counting.Counter,
    MakeActorFn,
], core.Worker]


def get_default_logger_fn(
        log_to_bigtable: bool = False,
        log_every: float = 10) -> Callable[[ActorId], loggers.Logger]:
    """Creates an actor logger."""

    def create_logger(actor_id: ActorId):
        return loggers.make_default_logger(
            'actor',
            save_data=(log_to_bigtable and actor_id == 0),
            time_delta=log_every,
            steps_key='actor_steps')

    return create_logger


def default_evaluator_factory(
        environment_factory: EnvironmentFactory,
        network_factory: NetworkFactory,
        policy_factory: PolicyFactory,
        observers: Sequence[observers_lib.EnvLoopObserver] = (),
        log_to_bigtable: bool = False,
        logger_fn: Optional[LoggerFn] = None) -> EvaluatorFactory:
    """Returns a default evaluator process."""

    def evaluator(
            random_key: networks_lib.PRNGKey,
            variable_source: core.VariableSource,
            counter: counting.Counter,
            make_actor: MakeActorFn,
    ):
        """The evaluation process."""

        environment_key, actor_key = jax.random.split(random_key)

        environment = environment_factory(utils.sample_uint32(environment_key))
        networks = network_factory(specs.make_environment_spec(environment))

        actor = make_actor(actor_key, policy_factory(
            networks), variable_source)

        counter = counting.Counter(counter, 'evaluator')
        if logger_fn is not None:
            logger = logger_fn('evaluator', log_to_bigtable, 'actor_steps')
        else:
            logger = loggers.make_default_logger(
                'evaluator', log_to_bigtable, steps_key='actor_steps')

        return environment_loop.EnvironmentLoop(environment, actor, counter,
                                                logger, observers=observers)

    return evaluator


@dataclasses.dataclass
class CheckpointingConfig:
    """Configuration options for learner checkpointer."""

    max_to_keep: int = 1

    directory: str = '~/acme'

    add_uid: bool = True


class DistributedLayout:
    """Program definition for a distributed agent based on a builder."""

    def __init__(
            self,
            seed: int,
            environment_factory: EnvironmentFactory,
            network_factory: NetworkFactory,
            builder: builders.GenericActorLearnerBuilder,
            policy_network: PolicyFactory,
            num_actors: int,
            environment_spec: Optional[specs.EnvironmentSpec] = None,
            actor_logger_fn: Optional[Callable[[
                ActorId], loggers.Logger]] = None,
            evaluator_factories: Sequence[EvaluatorFactory] = (),
            device_prefetch: bool = True,
            prefetch_size: int = 1,
            log_to_bigtable: bool = False,
            max_number_of_steps: Optional[int] = None,
            observers: Sequence[observers_lib.EnvLoopObserver] = (),
            multithreading_colocate_learner_and_reverb: bool = False,
            checkpointing_config: Optional[CheckpointingConfig] = None):

        if prefetch_size < 0:
            raise ValueError(
                f'Prefetch size={prefetch_size} should be non negative')

        actor_logger_fn = actor_logger_fn or get_default_logger_fn(
            log_to_bigtable)

        self._seed = seed
        self._builder = builder
        self._environment_factory = environment_factory
        self._network_factory = network_factory
        self._policy_network = policy_network
        self._environment_spec = environment_spec
        self._num_actors = num_actors
        self._device_prefetch = device_prefetch
        self._log_to_bigtable = log_to_bigtable
        self._prefetch_size = prefetch_size
        self._max_number_of_steps = max_number_of_steps
        self._actor_logger_fn = actor_logger_fn
        self._evaluator_factories = evaluator_factories
        self._observers = observers
        self._multithreading_colocate_learner_and_reverb = (
            multithreading_colocate_learner_and_reverb)
        self._checkpointing_config = checkpointing_config

    def replay(self):
        """The replay storage."""
        dummy_seed = 1
        environment_spec = (
            self._environment_spec or
            specs.make_environment_spec(self._environment_factory(dummy_seed)))
        return self._builder.make_replay_tables(environment_spec)

    def counter(self):
        kwargs = {}
        if self._checkpointing_config:
            kwargs = vars(self._checkpointing_config)
        return savers.CheckpointingRunner(
            counting.Counter(),
            key='counter',
            subdirectory='counter',
            time_delta_minutes=5,
            **kwargs)

    def learner(
            self,
            random_key: networks_lib.PRNGKey,
            replay: reverb.Client,
            counter: counting.Counter,
    ):
        """The Learning part of the agent."""

        if self._builder._config.env_name.startswith(
                'offline'):
            adder = self._builder.make_adder(replay)
            env = self._environment_factory(0)
            env_spec = specs.make_environment_spec(env)
            dataset = env.get_dataset()
            for t in tqdm.trange(dataset['observations'].shape[0]):
                discount = 1.0
                if t == 0 or dataset['timeouts'][t - 1]:
                    step_type = dm_env.StepType.FIRST
                elif dataset['timeouts'][t]:
                    step_type = dm_env.StepType.LAST
                    discount = 0.0
                else:
                    step_type = dm_env.StepType.MID

                ts = dm_env.TimeStep(
                    step_type=step_type,
                    reward=dataset['rewards'][t].astype(
                        env_spec.rewards.dtype),
                    discount=discount,
                    observation=np.concatenate([dataset['observations'][t],
                                                dataset['infos/goal'][t]],
                                               dtype=env_spec.observations.dtype),
                )
                if t == 0 or dataset['timeouts'][t - 1]:
                    adder.add_first(ts)
                else:
                    adder.add(action=dataset['actions'][t - 1].astype(env_spec.actions.dtype),
                              next_timestep=ts)

                if self._builder._config.local and t > 10_000:
                    break

        iterator = self._builder.make_dataset_iterator(replay)

        dummy_seed = 1
        environment_spec = (
            self._environment_spec or
            specs.make_environment_spec(self._environment_factory(dummy_seed)))

        networks = self._network_factory(environment_spec)

        if self._prefetch_size > 1:

            device = jax.devices()[0] if self._device_prefetch else None
            iterator = utils.prefetch(
                iterator, buffer_size=self._prefetch_size, device=device)
        else:
            logging.info('Not prefetching the iterator.')

        counter = counting.Counter(counter, 'learner')

        learner = self._builder.make_learner(random_key, networks, iterator, replay,
                                             counter)
        kwargs = {}
        if self._checkpointing_config:
            kwargs = vars(self._checkpointing_config)

        return savers.CheckpointingRunner(
            learner,
            key='learner',
            subdirectory='learner',
            time_delta_minutes=5,
            **kwargs)

    def actor(self, random_key: networks_lib.PRNGKey, replay: reverb.Client,
              variable_source: core.VariableSource, counter: counting.Counter,
              actor_id: ActorId) -> environment_loop.EnvironmentLoop:
        """The actor process."""
        adder = self._builder.make_adder(replay)

        environment_key, actor_key = jax.random.split(random_key)

        environment = self._environment_factory(
            utils.sample_uint32(environment_key))

        networks = self._network_factory(
            specs.make_environment_spec(environment))
        policy_network = self._policy_network(networks)
        actor = self._builder.make_actor(actor_key, policy_network, adder,
                                         variable_source)

        counter = counting.Counter(counter, 'actor')

        if actor_id == 0:
            counter.increment(steps=0)

        logger = self._actor_logger_fn(actor_id)

        return environment_loop.EnvironmentLoop(environment, actor, counter,
                                                logger, observers=self._observers)

    def coordinator(self, counter: counting.Counter, max_steps: int):
        if self._builder._config.env_name.startswith(
                'offline'):
            steps_key = 'learner_steps'
        else:
            steps_key = 'actor_steps'
        return lp_utils.StepsLimiter(counter, max_steps, steps_key=steps_key)

    def build(self, name='agent', program: Optional[lp.Program] = None):
        """Build the distributed agent topology."""
        if not program:
            program = lp.Program(name=name)

        key = jax.random.PRNGKey(self._seed)

        replay_node = lp.ReverbNode(self.replay)
        with program.group('replay'):
            if self._multithreading_colocate_learner_and_reverb:
                replay = replay_node.create_handle()
            else:
                replay = program.add_node(replay_node)

        with program.group('counter'):
            counter = program.add_node(lp.CourierNode(self.counter))
            if self._max_number_of_steps is not None:
                _ = program.add_node(
                    lp.CourierNode(self.coordinator, counter,
                                   self._max_number_of_steps))

        learner_key, key = jax.random.split(key)
        learner_node = lp.CourierNode(
            self.learner, learner_key, replay, counter)
        with program.group('learner'):
            if self._multithreading_colocate_learner_and_reverb:
                learner = learner_node.create_handle()
                program.add_node(
                    lp.MultiThreadingColocation([learner_node, replay_node]))
            else:
                learner = program.add_node(learner_node)

        def make_actor(random_key: networks_lib.PRNGKey,
                       policy_network: PolicyNetwork,
                       variable_source: core.VariableSource) -> core.Actor:
            return self._builder.make_actor(
                random_key, policy_network, variable_source=variable_source)

        with program.group('evaluator'):
            for evaluator in self._evaluator_factories:
                evaluator_key, key = jax.random.split(key)
                program.add_node(
                    lp.CourierNode(evaluator, evaluator_key, learner, counter,
                                   make_actor))

        with program.group('actor'):
            for actor_id in range(self._num_actors):
                actor_key, key = jax.random.split(key)
                program.add_node(
                    lp.CourierNode(self.actor, actor_key, replay, learner, counter,
                                   actor_id))

        return program
