

"""Metric Distillation learner implementation."""
import time
from typing import NamedTuple, Optional

import acme
import jax
import jax.numpy as jnp
import optax
from acme import types
from acme.jax import networks as networks_lib
from acme.jax import utils
from acme.utils import counting
from acme.utils import loggers
import os
import launchpad as lp


class TrainingState(NamedTuple):
    """Contains training state for the learner."""
    contrastive_optimizer_state: optax.OptState
    quasimetric_optimizer_state: optax.OptState
    critic_optimizer_state: optax.OptState
    policy_optimizer_state: optax.OptState
    contrastive_params: networks_lib.Params
    quasimetric_params: networks_lib.Params
    critic_params: networks_lib.Params
    policy_params: networks_lib.Params
    key: networks_lib.PRNGKey


class MetricDistillationLearner(acme.Learner):
    """Metric Distillation learner."""

    _state: TrainingState

    def __init__(
            self,
            networks,
            rng,
            contrastive_optimizer,
            quasimetric_optimizer,
            critic_optimizer,
            policy_optimizer,
            iterator,
            counter,
            logger,
            log_dir,
            obs_to_goal,
            goal_to_obs,
            config):
        """Initialize the Metric Distillation learner.

        Args:
          networks: Metric Distillation networks.
          rng: a key for random number generation.
          contrastive_optimizer: the contrastive function optimizer.
          quasimetric_optimizer: the quasimetric function optimizer.
          critic_optimizer: the critic optimizer.
          policy_optimizer: the policy optimizer.
          iterator: an iterator over training data.
          counter: counter object used to keep track of steps.
          logger: logger object to be used by learner.
          config: the experiment config file.
        """
        self._num_sgd_steps_per_step = config.num_sgd_steps_per_step
        self._obs_dim = config.obs_dim

        if not os.path.exists(log_dir):
            lp.stop()
        self.alive_file = log_dir + f'/.alive{int(time.time())}'
        open(self.alive_file, 'w').close()

        def mixing_obs_and_goal(transitions):
            obs, _ = jnp.split(transitions.observation,
                               [config.obs_dim], axis=1)
            future_goal = transitions.extras['future_state']
            future_action = transitions.extras['future_action']

            random_goal = jnp.roll(future_goal, 1, axis=0)
            random_action = jnp.roll(future_action, 1, axis=0)

            if config.random_goals == 0.0:
                new_obs = obs
                new_goal = future_goal
                new_action = future_action
            elif config.random_goals == 0.5:
                new_obs = jnp.concatenate([obs, obs], axis=0)
                new_goal = jnp.concatenate([future_goal, random_goal], axis=0)
                new_action = jnp.concatenate(
                    [future_action, random_action], axis=0)
            else:
                assert config.random_goals == 1.0
                new_obs = obs
                new_goal = random_goal
                new_action = random_action

            return new_obs, new_goal, new_action

        def contrastive_loss_fn(contrastive_params,
                                transitions):
            batch_size = transitions.observation.shape[0]
            obs, _ = jnp.split(transitions.observation,
                               [config.obs_dim], axis=1)
            goal = transitions.extras['future_state']
            logits, _, _ = networks.contrastive_network.apply(
                contrastive_params, obs, transitions.action, goal)

            I = jnp.eye(batch_size)

            def loss_fn(_logits):
                if config.use_cpc:
                    return (optax.softmax_cross_entropy(logits=_logits, labels=I)
                            + 0.01 * jax.nn.logsumexp(_logits, axis=1) ** 2)
                elif config.use_nce:
                    return optax.sigmoid_binary_cross_entropy(logits=_logits, labels=I)
                else:

                    return optax.softmax_cross_entropy(logits=_logits, labels=I)

            if config.use_cpc or config.use_nce:

                loss = jax.vmap(loss_fn, in_axes=2, out_axes=-1)(logits)
            else:

                loss = (jax.vmap(loss_fn, in_axes=2, out_axes=-1)(logits) +
                        jax.vmap(loss_fn, in_axes=2, out_axes=-1)(jnp.swapaxes(logits, 0, 1))) / 2.0
            loss = jnp.mean(loss)

            loss_triangle_ineq = 0.0

            I = I[:, :, None].repeat(logits.shape[-1], axis=-1)
            binary_acc = jnp.mean((logits > 0) == I)
            categorial_acc = jnp.mean(jnp.argmax(
                logits, axis=1) == jnp.arange(batch_size)[:, None])
            logits_pos = jnp.sum(logits * I) / jnp.sum(I)
            logits_neg = jnp.sum(logits * (1 - I)) / jnp.sum(1 - I)
            logsumexp = jnp.mean(jax.nn.logsumexp(
                logits[:, :, 0], axis=1) ** 2)

            logs = {


                "binary_accuracy": binary_acc,
                "categorical_accuracy": categorial_acc,
                "logits_pos": logits_pos,
                "logits_neg": logits_neg,
                "logsumexp": logsumexp,
                "triangle_ineq": loss_triangle_ineq,
            }

            if config.repr_norm and config.adaptive_repr_norm_temp:
                logs.update({
                    "repr_norm_temp": jnp.exp(contrastive_params['~']['log_repr_norm_temp'])
                })

            return loss, logs

        def quasimetric_loss_fn(quasimetric_params,
                                contrastive_params,
                                policy_params,
                                transitions,
                                key):
            lam = quasimetric_params['lam']
            lam2 = quasimetric_params['lam2']
            batch_size = transitions.observation.shape[0]
            obs, _ = jnp.split(transitions.observation,
                               [config.obs_dim], axis=1)
            goal = transitions.extras['future_state']
            future_action = transitions.extras['future_action']

            obs_goal_logits, _, _ = networks.contrastive_network.apply(
                contrastive_params, obs, transitions.action, goal)
            goal_goal_logits, _, _ = networks.contrastive_network.apply(
                contrastive_params, goal, future_action, goal)
            goal_goal_logits = jax.vmap(jnp.diag, -1, -1)(goal_goal_logits)

            if config.twin_q:
                assert obs_goal_logits.shape == (batch_size, batch_size, 2)
                assert goal_goal_logits.shape == (batch_size, 2)
            else:
                assert obs_goal_logits.shape == (batch_size, batch_size, 1)
                assert goal_goal_logits.shape == (batch_size, 1)

            d_target = jnp.min(goal_goal_logits - obs_goal_logits, axis=-1)
            d_target = jax.lax.stop_gradient(d_target)[..., None]

            key, key0 = jax.random.split(key)
            actions_perm = jax.random.permutation(
                key0, transitions.extras['future_action'])

            d_sg = networks.quasimetric_network.apply(
                quasimetric_params['network_params'],
                obs[:, None], transitions.action[:, None], goal[None], actions_perm[None])

            I = jnp.eye(batch_size)
            loss_hinge_pos = jnp.mean(jnp.maximum(0, jax.vmap(
                jnp.diag, -1, -1)(d_sg - d_target)) ** 2, axis=0)
            loss_hinge_neg = jnp.mean(jnp.maximum(
                0, (1 - I)[..., None] * (d_target - d_sg)) ** 2, axis=(0, 1))
            assert loss_hinge_pos.shape == loss_hinge_neg.shape == (
                (2,) if config.twin_q else (1,))
            if config.twin_q:
                lams = jnp.stack([lam, lam2], axis=0)
            else:
                lams = jnp.array([lam])
            lams = jax.nn.softplus(lams) + 0.01
            lams_scaled = jax.lax.stop_gradient(config.lam_scale * lams)
            loss_hinge = (loss_hinge_pos + loss_hinge_neg / lams)
            loss_dual = lams * \
                (config.margin - jax.lax.stop_gradient(loss_hinge_pos))
            loss = loss_hinge + loss_dual
            loss = jnp.mean(loss)

            if config.contrastive_only:
                loss *= 0.0

            if config.quasimetric_action_constraint:

                obs_as_g = obs_to_goal(obs)
                rand_g = jnp.roll(goal, 1, axis=0)
                new_obs_and_goal = jnp.concatenate([obs, rand_g], axis=-1)
                dist_params = networks.policy_network.apply(
                    policy_params, new_obs_and_goal)
                action = networks.sample(dist_params, key)

                d_ss = networks.quasimetric_network.apply(
                    quasimetric_params['network_params'],
                    obs, action, goal)

                loss_action_constraint = config.quasimetric_action_constraint_coef * \
                    jnp.mean(d_ss)
                loss += loss_action_constraint

            else:
                d_ss = 0.0

            logs = {
                "loss_hinge_pos": loss_hinge_pos,
                "loss_hinge_neg": loss_hinge_neg,
                "loss_hinge": loss_hinge,
                "loss_dual": loss_dual,
                "d": jnp.mean(jnp.diag(d_sg.mean(axis=-1))),
                "d_ss": jnp.mean(d_ss),
                "gg_logits_minus_sg_logits": jnp.mean(jnp.diag(d_target.mean(axis=-1))),
                "lambda": lams.mean(),
                "lambda_scale": config.lam_scale,
                "margin": config.margin
            }

            return loss, logs

        def actor_loss_fn(policy_params,
                          contrastive_params,
                          quasimetric_params,
                          transitions,
                          key):
            obs, goal, future_action = mixing_obs_and_goal(transitions)

            obs_and_goal = jnp.concatenate([obs, obs_to_goal(goal)], axis=1)

            if config.use_gcbc:
                dist_params = networks.policy_network.apply(
                    policy_params, obs_and_goal)
                log_prob = networks.log_prob(dist_params, transitions.action)

                train_mask = jnp.float32(
                    (transitions.action * 1E8 % 10)[:, 0] != 4)
                val_mask = 1.0 - train_mask

                actor_loss = -1.0 * jnp.mean(train_mask * log_prob)
                val_actor_loss = -1.0 * jnp.mean(val_mask * log_prob)

                logs = {
                    "actor_loss": actor_loss,
                    "val_actor_loss": val_actor_loss,
                }
            else:
                dist_params = networks.policy_network.apply(
                    policy_params, obs_and_goal)
                action = networks.sample(dist_params, key)

                if config.contrastive_only:

                    q_action, _, _ = networks.contrastive_network.apply(
                        contrastive_params, obs, action, goal)
                    q_action = jnp.min(q_action, axis=-1)
                    actor_loss = -jnp.diag(q_action)

                    d_sg = 0.0
                else:
                    d_sg = networks.quasimetric_network.apply(
                        quasimetric_params['network_params'],
                        obs, action, goal, future_action).max(axis=-1)

                    actor_loss = jnp.mean(d_sg)

                adv_mean = 0.
                weight_mean = 0.
                if config.awr:
                    orig_action = transitions.action
                    action = action[:orig_action.shape[0]]

                    if config.contrastive_only:
                        q_action, _, _ = networks.contrastive_network.apply(
                            contrastive_params, obs, orig_action, goal)
                        q_action = jnp.diag(jnp.min(q_action, axis=-1))
                        v_action, _, _ = networks.contrastive_network.apply(
                            contrastive_params, obs, action, goal)
                        v_action = jnp.diag(jnp.min(v_action, axis=-1))
                        adv = q_action - v_action
                    else:
                        q_action = -networks.quasimetric_network.apply(
                            quasimetric_params['network_params'],
                            obs, orig_action, goal, future_action).max(axis=-1)
                        v_action = -networks.quasimetric_network.apply(
                            quasimetric_params['network_params'],
                            obs, action, goal, future_action).max(axis=-1)
                        adv = q_action - v_action

                    assert adv.shape == (orig_action.shape[0],)
                    adv = jax.lax.stop_gradient(adv)
                    norm_adv = (adv - jnp.mean(adv)) / (jnp.std(adv) + 1e-8)

                    train_mask = jnp.float32(
                        (orig_action * 1E8 % 10)[:, 0] != 4)
                    val_mask = 1.0 - train_mask
                    weight = jnp.exp(norm_adv / config.awr_temp)
                    weight = jnp.clip(weight, a_max=100)

                    if config.bc_loss == 'mse':

                        bc_loss = train_mask * \
                            jnp.mean((action - orig_action)
                                     ** 2, axis=1) * weight
                        bc_val_loss = val_mask * \
                            jnp.mean((action - orig_action)
                                     ** 2, axis=1) * weight
                    elif config.bc_loss == 'mle':

                        bc_loss = -1.0 * train_mask * \
                            networks.log_prob(
                                dist_params, orig_action) * weight
                        bc_val_loss = -1.0 * val_mask * \
                            networks.log_prob(
                                dist_params, orig_action) * weight
                    else:
                        raise NotImplementedError

                    assert bc_loss.shape == (orig_action.shape[0],)

                    actor_loss = bc_loss
                    adv_mean = jnp.mean(adv)
                    adv_std = jnp.std(adv)
                    weight_mean = jnp.mean(weight)

                elif config.bc_coef > 0 and not config.awr:
                    assert 0.0 <= config.bc_coef <= 1.0
                    orig_action = transitions.action
                    action = action[:orig_action.shape[0]]

                    train_mask = jnp.float32(
                        (orig_action * 1E8 % 10)[:, 0] != 4)
                    val_mask = 1.0 - train_mask

                    if config.bc_loss == 'mse':

                        bc_loss = train_mask * \
                            jnp.mean((action - orig_action) ** 2, axis=1)
                        bc_val_loss = val_mask * \
                            jnp.mean((action - orig_action) ** 2, axis=1)
                    elif config.bc_loss == 'mle':

                        bc_loss = -1.0 * train_mask * \
                            networks.log_prob(dist_params, orig_action)
                        bc_val_loss = -1.0 * val_mask * \
                            networks.log_prob(dist_params, orig_action)
                    else:
                        raise NotImplementedError

                    actor_loss = config.bc_coef * bc_loss + \
                        (1 - config.bc_coef) * actor_loss
                else:
                    bc_loss = 0.0
                    bc_val_loss = 0.0
                    train_mask = 1.0
                    val_mask = 0.0

                actor_loss = jnp.mean(actor_loss)

                logs = {
                    "d_sg_actor": jnp.mean(d_sg),
                    "gcbc_loss": jnp.sum(bc_loss) / (jnp.sum(train_mask) + 1e-8),
                    "gcbc_val_loss": jnp.sum(bc_val_loss) / (jnp.sum(val_mask) + 1e-8),
                    "random_goals": config.random_goals,
                    "bc_coef": config.bc_coef,
                    "adv_mean": adv_mean,
                    "weight_mean": weight_mean,
                    "adv_std": adv_std,
                }

            return actor_loss, logs

        contrastive_grad = jax.value_and_grad(
            contrastive_loss_fn, has_aux=True)
        quasimetric_grad = jax.value_and_grad(
            quasimetric_loss_fn, has_aux=True)

        actor_grad = jax.value_and_grad(actor_loss_fn, has_aux=True)

        def update_step(
            state,
            transitions
        ):
            if not os.path.exists(self.alive_file):
                lp.stop()

            key_actor, key_quasimetric, key = jax.random.split(state.key, 3)

            if not config.use_gcbc:

                (contrastive_loss, contrastive_logs), contrastive_grads = contrastive_grad(
                    state.contrastive_params, transitions)
                logs = contrastive_logs

                (quasimetric_loss, quasimetric_logs), quasimetric_grads = quasimetric_grad(
                    state.quasimetric_params, state.contrastive_params, state.policy_params,
                    transitions, key_quasimetric)
                logs.update(quasimetric_logs)
            else:
                logs = {}

            (actor_loss, actor_logs), actor_grads = actor_grad(
                state.policy_params, state.contrastive_params,
                state.quasimetric_params, transitions, key_actor)
            logs.update(actor_logs)

            if config.use_gcbc:
                contrastive_params = state.contrastive_params
                contrastive_optimizer_state = state.contrastive_optimizer_state
                quasimetric_params = state.quasimetric_params
                quasimetric_optimizer_state = state.quasimetric_optimizer_state
            else:

                contrastive_update, contrastive_optimizer_state = contrastive_optimizer.update(
                    contrastive_grads, state.contrastive_optimizer_state)
                contrastive_params = optax.apply_updates(
                    state.contrastive_params, contrastive_update)

                quasimetric_update, quasimetric_optimizer_state = quasimetric_optimizer.update(
                    quasimetric_grads, state.quasimetric_optimizer_state)
                quasimetric_params = optax.apply_updates(
                    state.quasimetric_params, quasimetric_update)

                logs.update({
                    "contrastive_loss": contrastive_loss,
                    "quasimetric_loss": quasimetric_loss,
                    "actor_loss": actor_loss,
                })

            actor_update, policy_optimizer_state = policy_optimizer.update(
                actor_grads, state.policy_optimizer_state)
            policy_params = optax.apply_updates(
                state.policy_params, actor_update)

            new_state = TrainingState(
                contrastive_optimizer_state=contrastive_optimizer_state,
                quasimetric_optimizer_state=quasimetric_optimizer_state,
                critic_optimizer_state=state.critic_optimizer_state,
                policy_optimizer_state=policy_optimizer_state,
                contrastive_params=contrastive_params,
                quasimetric_params=quasimetric_params,
                critic_params=state.critic_params,
                policy_params=policy_params,
                key=key,
            )

            return new_state, logs

        self._counter = counter or counting.Counter()

        self._counter.increment(steps=0, walltime=0)
        self._logger = logger or loggers.make_default_logger(
            'learner', asynchronous=True, serialize_fn=utils.fetch_devicearray,
            time_delta=10.0)

        self._iterator = iterator

        update_step = utils.process_multiple_batches(update_step,
                                                     config.num_sgd_steps_per_step)

        if config.jit:
            self._update_step = jax.jit(update_step)
        else:
            self._update_step = update_step

        def make_initial_state(key):
            """Initialises the training state (parameters and optimiser state)."""
            key_contrastive, key_quasimetric, key_critic, key_policy, key = jax.random.split(
                key, 5)

            contrastive_params = networks.contrastive_network.init(
                key_contrastive)
            contrastive_optimizer_state = contrastive_optimizer.init(
                contrastive_params)

            quasimetric_params = dict(
                lam=jnp.asarray(config.lam_init, dtype=jnp.float32),
                lam2=jnp.asarray(config.lam_init, dtype=jnp.float32),
                network_params=networks.quasimetric_network.init(
                    key_quasimetric)
            )
            quasimetric_optimizer_state = quasimetric_optimizer.init(
                quasimetric_params)

            critic_params = networks.critic_network.init(key_critic)
            critic_optimizer_state = critic_optimizer.init(critic_params)

            policy_params = networks.policy_network.init(key_policy)
            policy_optimizer_state = policy_optimizer.init(policy_params)

            state = TrainingState(
                contrastive_optimizer_state=contrastive_optimizer_state,
                quasimetric_optimizer_state=quasimetric_optimizer_state,
                critic_optimizer_state=critic_optimizer_state,
                policy_optimizer_state=policy_optimizer_state,
                contrastive_params=contrastive_params,
                quasimetric_params=quasimetric_params,
                critic_params=critic_params,
                policy_params=policy_params,
                key=key)

            return state

        self._state = make_initial_state(rng)

        self._timestamp = None

    def step(self):
        with jax.profiler.StepTraceAnnotation('step', step_num=self._counter):
            sample = next(self._iterator)
            transitions = types.Transition(*sample.data)
            self._state, logs = self._update_step(self._state, transitions)

        timestamp = time.time()
        elapsed_time = timestamp - self._timestamp if self._timestamp else 0
        self._timestamp = timestamp

        counts = self._counter.increment(
            steps=self._num_sgd_steps_per_step, walltime=elapsed_time)
        if elapsed_time > 0:
            logs['steps_per_second'] = (
                self._num_sgd_steps_per_step / elapsed_time)
        else:
            logs['steps_per_second'] = 0.

        if not os.path.exists(self.alive_file):
            lp.stop()
        self._logger.write({**logs, **counts})

    def get_variables(self, names):
        variables = {
            'contrastive': self._state.contrastive_params,
            'quasimetric': self._state.quasimetric_params,
            'critic': self._state.critic_params,
            'policy': self._state.policy_params,
        }
        return [variables[name] for name in names]

    def save(self):
        return self._state

    def restore(self, state):
        self._state = state
