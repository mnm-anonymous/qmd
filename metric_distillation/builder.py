

"""Metric Distillation builder."""
import functools

import optax
import reverb
import tensorflow as tf
import tree
from acme import types
from acme.adders import reverb as adders_reverb
from acme.agents.jax import actor_core as actor_core_lib
from acme.agents.jax import actors
from acme.agents.jax import builders
from acme.jax import variable_utils

from metric_distillation import learning
from metric_distillation import utils as metric_distillation_utils
from reverb import rate_limiters


class MetricDistillationBuilder(builders.ActorLearnerBuilder):
    """Metric Distillation builder."""

    def __init__(
            self,
            config,
            log_dir=None,
            logger_fn=lambda: None,
    ):
        """Creates a Metric Distillation learner, a behavior policy and an eval actor.

        Args:
          config: a config with Metric Distillation hyperparameters
          logger_fn: a logger factory for the learner
        """
        self._config = config
        self._logger_fn = logger_fn
        self._log_dir = log_dir

    def make_learner(
            self,
            random_key,
            networks,
            dataset,
            replay_client=None,
            counter=None,
    ):
        """Create optimizers and the Metric Distillation learner."""
        contrastive_optimizer = optax.adam(
            learning_rate=self._config.contrastive_learning_rate)
        quasimetric_optimizer = optax.adam(
            learning_rate=self._config.quasimetric_learning_rate)
        critic_optimizer = optax.adam(
            learning_rate=self._config.critic_learning_rate)
        policy_optimizer = optax.adam(
            learning_rate=self._config.actor_learning_rate)

        return learning.MetricDistillationLearner(
            networks=networks,
            rng=random_key,
            contrastive_optimizer=contrastive_optimizer,
            quasimetric_optimizer=quasimetric_optimizer,
            critic_optimizer=critic_optimizer,
            policy_optimizer=policy_optimizer,
            iterator=dataset,
            counter=counter,
            logger=self._logger_fn(),
            log_dir=self._log_dir,
            obs_to_goal=functools.partial(metric_distillation_utils.obs_to_goal_2d,
                                          start_index=self._config.start_index,
                                          end_index=self._config.end_index),
            goal_to_obs=functools.partial(metric_distillation_utils.goal_to_obs_2d,
                                          obs_dim=self._config.obs_dim,
                                          start_index=self._config.start_index,
                                          end_index=self._config.end_index),
            config=self._config)

    def make_actor(
            self,
            random_key,
            policy_network,
            adder=None,
            variable_source=None):
        """Create the behavior policy and the eval actor."""
        assert variable_source is not None
        actor_core = actor_core_lib.batched_feed_forward_to_actor_core(
            policy_network)
        variable_client = variable_utils.VariableClient(variable_source, 'policy',
                                                        device='cpu')
        if self._config.use_random_actor:
            ACTOR = metric_distillation_utils.InitiallyRandomActor
        else:
            ACTOR = actors.GenericActor
        return ACTOR(
            actor_core, random_key, variable_client, adder, backend='cpu')

    def make_replay_tables(
            self,
            environment_spec,
    ):
        """Create tables to insert data into."""
        samples_per_insert_tolerance = (
            self._config.samples_per_insert_tolerance_rate
            * self._config.samples_per_insert)
        min_replay_traj = self._config.min_replay_size // self._config.max_episode_steps
        max_replay_traj = self._config.max_replay_size // self._config.max_episode_steps
        error_buffer = min_replay_traj * samples_per_insert_tolerance
        limiter = rate_limiters.SampleToInsertRatio(
            min_size_to_sample=min_replay_traj,
            samples_per_insert=self._config.samples_per_insert,
            error_buffer=error_buffer)
        return [
            reverb.Table(
                name=self._config.replay_table_name,
                sampler=reverb.selectors.Uniform(),
                remover=reverb.selectors.Fifo(),
                max_size=max_replay_traj,
                rate_limiter=limiter,
                signature=adders_reverb.EpisodeAdder.signature(environment_spec, {}))
        ]

    def make_dataset_iterator(
            self, replay_client):
        """Create a dataset iterator to use for learning/updating the agent."""

        @tf.function
        def future_goal_relabeling(sample):
            seq_len = tf.shape(sample.data.observation)[0]
            arange = tf.range(seq_len)
            is_future_mask = tf.cast(
                arange[:, None] < arange[None], tf.float32)
            discount = self._config.discount ** tf.cast(arange[None] - arange[:, None],
                                                        tf.float32)
            probs = is_future_mask * discount

            goal_index = tf.random.categorical(logits=tf.math.log(probs),
                                               num_samples=1)[:, 0]
            state = sample.data.observation[:-1, :self._config.obs_dim]
            next_state = sample.data.observation[1:, :self._config.obs_dim]

            goal = sample.data.observation[:, :self._config.obs_dim]
            goal = metric_distillation_utils.obs_to_goal_2d(
                goal, start_index=self._config.start_index,
                end_index=self._config.end_index)
            goal = tf.gather(goal, goal_index[:-1])
            future_state = tf.gather(sample.data.observation[:, :self._config.obs_dim],
                                     goal_index[:-1])
            goal_action = tf.gather(sample.data.action, goal_index[:-1])
            new_obs = tf.concat([state, goal], axis=1)
            new_next_obs = tf.concat([next_state, goal], axis=1)
            transition = types.Transition(
                observation=new_obs,
                action=sample.data.action[:-1],
                reward=sample.data.reward[:-1],
                discount=sample.data.discount[:-1],
                next_observation=new_next_obs,
                extras={
                    'next_action': sample.data.action[1:],
                    'future_goal': goal,
                    'future_state': future_state,
                    'future_action': goal_action,
                })

            shift = tf.random.uniform((), 0, seq_len, tf.int32)
            transition = tree.map_structure(lambda t: tf.roll(t, shift, axis=0),
                                            transition)

            return transition

        @tf.function
        def random_goal_relabeling(sample):
            batch_size = tf.shape(sample.observation)[0]
            state = sample.observation[:, :self._config.obs_dim]
            shift = tf.random.uniform((), 0, batch_size, tf.int32)
            goal = tf.roll(state, shift, axis=0)

            goal = metric_distillation_utils.obs_to_goal_2d(
                goal, start_index=self._config.start_index,
                end_index=self._config.end_index)

            extras = dict(sample.extras)
            extras['random_goal'] = goal
            transition = sample._replace(extras=extras)

            return transition

        if self._config.num_parallel_calls:
            num_parallel_calls = self._config.num_parallel_calls
        else:
            num_parallel_calls = tf.data.AUTOTUNE

        def _make_dataset(unused_idx):
            dataset = reverb.TrajectoryDataset.from_table_signature(
                server_address=replay_client.server_address,
                table=self._config.replay_table_name,
                max_in_flight_samples_per_worker=100)

            dataset = dataset.map(future_goal_relabeling)

            def _transpose_fn(t):
                dims = tf.range(tf.shape(tf.shape(t))[0])
                perm = tf.concat([[1, 0], dims[2:]], axis=0)
                return tf.transpose(t, perm)

            dataset = dataset.batch(
                self._config.batch_size, drop_remainder=True)
            dataset = dataset.map(
                lambda transition: tree.map_structure(_transpose_fn, transition))
            dataset = dataset.unbatch()

            dataset = dataset.map(random_goal_relabeling)

            dataset = dataset.unbatch()

            return dataset

        dataset = tf.data.Dataset.from_tensors(0).repeat()
        dataset = dataset.interleave(
            map_func=_make_dataset,
            cycle_length=num_parallel_calls,
            num_parallel_calls=num_parallel_calls,
            deterministic=False)

        dataset = dataset.batch(
            self._config.batch_size * self._config.num_sgd_steps_per_step,
            drop_remainder=True)

        @tf.function
        def add_info_fn(data):
            info = reverb.SampleInfo(key=0,
                                     probability=0.0,
                                     table_size=0,
                                     priority=0.0)
            return reverb.ReplaySample(info=info, data=data)

        dataset = dataset.map(add_info_fn, num_parallel_calls=tf.data.AUTOTUNE,
                              deterministic=False)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        return dataset.as_numpy_iterator()

    def make_adder(self,
                   replay_client):
        """Create an adder to record data generated by the actor/environment."""
        return adders_reverb.EpisodeAdder(
            client=replay_client,
            priority_fns={self._config.replay_table_name: None},
            max_sequence_length=self._config.max_episode_steps + 1)