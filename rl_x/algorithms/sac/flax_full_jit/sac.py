import os
import shutil
import json
from copy import deepcopy
from functools import partial
import logging
import time
import tree
import numpy as np
import jax
import jax.numpy as jnp
from jax.lax import stop_gradient
from flax.training.train_state import TrainState
from flax.training import orbax_utils
import orbax.checkpoint
import optax
import wandb

from rl_x.algorithms.sac.flax_full_jit.general_properties import GeneralProperties
from rl_x.algorithms.sac.flax_full_jit.policy import get_policy
from rl_x.algorithms.sac.flax_full_jit.critic import get_critic
from rl_x.algorithms.sac.flax_full_jit.entropy_coefficient import EntropyCoefficient
from rl_x.algorithms.sac.flax_full_jit.rl_train_state import RLTrainState

rlx_logger = logging.getLogger("rl_x")


class SAC:
    def __init__(self, config, env, run_path, writer):
        self.config = config
        self.env = env
        self.writer = writer

        self.save_model = config.runner.save_model
        self.save_path = os.path.join(run_path, "models")
        self.track_console = config.runner.track_console
        self.track_tb = config.runner.track_tb
        self.track_wandb = config.runner.track_wandb
        self.seed = config.environment.seed
        self.total_timesteps = config.algorithm.total_timesteps
        self.nr_parallel_seeds = config.algorithm.nr_parallel_seeds
        self.nr_envs = config.environment.nr_envs
        self.render = config.environment.render
        self.learning_rate = config.algorithm.learning_rate
        self.anneal_learning_rate = config.algorithm.anneal_learning_rate
        self.buffer_size = config.algorithm.buffer_size
        self.learning_starts = config.algorithm.learning_starts
        self.batch_size = config.algorithm.batch_size
        self.tau = config.algorithm.tau
        self.gamma = config.algorithm.gamma
        self.target_entropy = config.algorithm.target_entropy
        self.logging_frequency = config.algorithm.logging_frequency
        self.evaluation_and_save_frequency = config.algorithm.evaluation_and_save_frequency
        self.evaluation_active = config.algorithm.evaluation_active
        self.evaluation_episodes = config.algorithm.evaluation_episodes
        self.total_training_timesteps = self.total_timesteps - self.learning_starts
        if config.algorithm.evaluation_and_save_frequency == -1:
            self.evaluation_and_save_frequency = self.nr_envs * (self.total_training_timesteps // self.nr_envs)
        self.nr_eval_save_iterations = self.total_training_timesteps // self.evaluation_and_save_frequency
        self.nr_loggings_per_eval_save_iteration = self.evaluation_and_save_frequency // self.logging_frequency
        self.nr_updates_per_logging_iteration = self.logging_frequency // self.nr_envs

        if self.nr_parallel_seeds > 1:
            raise ValueError("Parallel seeds are not supported yet. This is mainly limited by not being able to log mutliple wandb runs at the same time.")

        rlx_logger.info(f"Using device: {jax.default_backend()}")
        
        self.key = jax.random.PRNGKey(self.seed)
        self.key, reset_key, policy_key, critic_key, entropy_coefficient_key = jax.random.split(self.key, 5)
        reset_key = jax.random.split(reset_key, 1)

        self.env_as_low = env.single_action_space.low
        self.env_as_high = env.single_action_space.high

        self.policy, self.get_processed_action = get_policy(config, env)
        self.critic = get_critic(config, env)
        
        if self.target_entropy == "auto":
            self.target_entropy = -np.prod(env.single_action_space.shape).item()
        else:
            self.target_entropy = float(self.target_entropy)
        self.entropy_coefficient = EntropyCoefficient(1.0)

        self.policy.apply = jax.jit(self.policy.apply)
        self.critic.apply = jax.jit(self.critic.apply)
        self.entropy_coefficient.apply = jax.jit(self.entropy_coefficient.apply)

        def linear_schedule(count):
            step = (count * self.nr_envs) - self.learning_starts
            total_steps = self.total_timesteps - self.learning_starts
            fraction = 1.0 - (step / total_steps)
            return self.learning_rate * fraction
        
        self.q_learning_rate = linear_schedule if self.anneal_learning_rate else self.learning_rate
        self.policy_learning_rate = linear_schedule if self.anneal_learning_rate else self.learning_rate
        self.entropy_learning_rate = linear_schedule if self.anneal_learning_rate else self.learning_rate

        env_state = self.env.reset(reset_key)
        self.dummy_state = env_state.next_observation
        self.dummy_action = jnp.array([self.env.single_action_space.sample(reset_key[0])])

        self.policy_state = TrainState.create(
            apply_fn=self.policy.apply,
            params=self.policy.init(policy_key, self.dummy_state),
            tx=optax.inject_hyperparams(optax.adam)(learning_rate=self.policy_learning_rate)
        )

        self.critic_state = RLTrainState.create(
            apply_fn=self.critic.apply,
            params=self.critic.init(critic_key, self.dummy_state, self.dummy_action),
            target_params=self.critic.init(critic_key, self.dummy_state, self.dummy_action),
            tx=optax.inject_hyperparams(optax.adam)(learning_rate=self.q_learning_rate)
        )

        self.entropy_coefficient_state = TrainState.create(
            apply_fn=self.entropy_coefficient.apply,
            params=self.entropy_coefficient.init(entropy_coefficient_key),
            tx=optax.inject_hyperparams(optax.adam)(learning_rate=self.entropy_learning_rate)
        )

        if self.save_model:
            os.makedirs(self.save_path)
            self.latest_model_file_name = "latest.model"
            self.latest_model_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
        
    
    def train(self):
        def jitable_train_function(key, parallel_seed_id):
            key, reset_key = jax.random.split(key, 2)
            reset_keys = jax.random.split(reset_key, self.nr_envs)
            env_state = self.env.reset(reset_keys)

            policy_state = self.policy_state
            critic_state = self.critic_state
            entropy_coefficient_state = self.entropy_coefficient_state

            # Replay buffer
            capacity = int(self.buffer_size // self.nr_envs)
            states_buffer = jnp.zeros((capacity, self.nr_envs) + (self.dummy_state.shape[1],), dtype=jnp.float32)
            next_states_buffer = jnp.zeros((capacity, self.nr_envs) + (self.dummy_state.shape[1],), dtype=jnp.float32)
            actions_buffer = jnp.zeros((capacity, self.nr_envs) + (self.dummy_action.shape[1],), dtype=jnp.float32)
            rewards_buffer = jnp.zeros((capacity, self.nr_envs), dtype=jnp.float32)
            terminations_buffer = jnp.zeros((capacity, self.nr_envs), dtype=jnp.float32)
            replay_buffer = {
                "states": states_buffer,
                "next_states": next_states_buffer,
                "actions": actions_buffer,
                "rewards": rewards_buffer,
                "terminations": terminations_buffer,
                "pos": jnp.zeros((), dtype=jnp.int32),
                "size": jnp.zeros((), dtype=jnp.int32)
            }

            # Fill replay buffer until learning_starts
            ...

            def eval_save_iteration(eval_save_iteration_carry, eval_save_iteration_step):
                policy_state, critic_state, entropy_coefficient_state, replay_buffer, env_state, key = eval_save_iteration_carry

                def logging_iteration(logging_iteration_carry, logging_iteration_step):
                    policy_state, critic_state, entropy_coefficient_state, replay_buffer, env_state, key = logging_iteration_carry

                    def learning_iteration(learning_iteration_carry, learning_iteration_step):
                        policy_state, critic_state, entropy_coefficient_state, replay_buffer, env_state, key = learning_iteration_carry

                        # Acting
                        key, subkey = jax.random.split(key)
                        observation = env_state.next_observation
                        action = self.policy.apply(policy_state.params, observation).sample(seed=subkey)
                        processed_action = self.get_processed_action(action)
                        env_state = self.env.step(env_state, processed_action)

                        # Adding to replay buffer
                        replay_buffer["states"] = replay_buffer["states"].at[replay_buffer["pos"]].set(observation)
                        replay_buffer["next_states"] = replay_buffer["next_states"].at[replay_buffer["pos"]].set(env_state.actual_next_observation)
                        replay_buffer["actions"] = replay_buffer["actions"].at[replay_buffer["pos"]].set(action)
                        replay_buffer["rewards"] = replay_buffer["rewards"].at[replay_buffer["pos"]].set(env_state.reward)
                        replay_buffer["terminations"] = replay_buffer["terminations"].at[replay_buffer["pos"]].set(env_state.terminated)
                        replay_buffer["pos"] = (replay_buffer["pos"] + 1) % capacity
                        replay_buffer["size"] = jnp.minimum(replay_buffer["size"] + 1, capacity)

                        if self.render:
                            def render(env_state):
                                return self.env.render(env_state)
                            
                            env_state = jax.experimental.io_callback(render, env_state, env_state)


                        # Optimizing
                        def loss_fn(policy_params, critic_params, entropy_coefficient_params, state, next_state, action, reward, terminated, key1, key2):
                            # Critic loss
                            dist = stop_gradient(self.policy.apply(policy_params, next_state))
                            next_action = dist.sample(seed=key1)
                            next_log_prob = dist.log_prob(next_action)

                            alpha_with_grad = self.entropy_coefficient.apply(entropy_coefficient_params)
                            alpha = stop_gradient(alpha_with_grad)

                            next_q_target = self.critic.apply(critic_state.target_params, next_state, next_action)
                            min_next_q_target = jnp.min(next_q_target)

                            y = reward + self.gamma * (1 - terminated) * (min_next_q_target - alpha * next_log_prob)

                            q = self.critic.apply(critic_params, state, action)
                            q_loss = (q - y) ** 2

                            # Policy loss
                            dist = self.policy.apply(policy_params, state)
                            current_action = dist.sample(seed=key2)
                            current_log_prob = dist.log_prob(current_action)
                            entropy = stop_gradient(-current_log_prob)

                            q = self.critic.apply(stop_gradient(critic_params), state, current_action)
                            min_q = jnp.min(q)

                            policy_loss = alpha * current_log_prob - min_q

                            # Entropy loss
                            entropy_loss = alpha_with_grad * (entropy - self.target_entropy)

                            # Combine losses
                            loss = q_loss + policy_loss + entropy_loss

                            # Create metrics
                            metrics = {
                                "loss/q_loss": q_loss,
                                "loss/policy_loss": policy_loss,
                                "loss/entropy_loss": entropy_loss,
                                "entropy/entropy": entropy,
                                "entropy/alpha": alpha,
                                "q_value/q_value": min_q,
                            }

                            return loss, (metrics)
                        

                        vmap_loss_fn = jax.vmap(loss_fn, in_axes=(None, None, None, 0, 0, 0, 0, 0, 0, 0), out_axes=0)
                        safe_mean = lambda x: jnp.mean(x) if x is not None else x
                        mean_vmapped_loss_fn = lambda *a, **k: tree.map_structure(safe_mean, vmap_loss_fn(*a, **k))
                        grad_loss_fn = jax.value_and_grad(mean_vmapped_loss_fn, argnums=(0, 1, 2), has_aux=True)

                        keys = jax.random.split(key, (self.batch_size * 2) + 2)
                        key, replay_buffer_key, update_keys = keys[0], keys[1], keys[2:]
                        update_keys1, update_keys2 = update_keys[:self.batch_size], update_keys[self.batch_size:]

                        idx1 = jax.random.randint(replay_buffer_key, (self.batch_size,), 0, replay_buffer["size"])
                        idx2 = jax.random.randint(replay_buffer_key, (self.batch_size,), 0, self.nr_envs)
                        states = replay_buffer["states"][idx1, idx2]
                        next_states = replay_buffer["next_states"][idx1, idx2]
                        actions = replay_buffer["actions"][idx1, idx2]
                        rewards = replay_buffer["rewards"][idx1, idx2]
                        terminations = replay_buffer["terminations"][idx1, idx2]

                        (loss, (metrics)), (policy_gradients, critic_gradients, entropy_gradients) = grad_loss_fn(
                            policy_state.params, critic_state.params, entropy_coefficient_state.params,
                            states, next_states, actions, rewards, terminations, update_keys1, update_keys2)

                        policy_state = policy_state.apply_gradients(grads=policy_gradients)
                        critic_state = critic_state.apply_gradients(grads=critic_gradients)
                        entropy_coefficient_state = entropy_coefficient_state.apply_gradients(grads=entropy_gradients)

                        critic_state = critic_state.replace(target_params=optax.incremental_update(critic_state.params, critic_state.target_params, self.tau))

                        metrics["lr/learning_rate"] = policy_state.opt_state.hyperparams["learning_rate"]
                        metrics["gradients/policy_grad_norm"] = optax.global_norm(policy_gradients)
                        metrics["gradients/critic_grad_norm"] = optax.global_norm(critic_gradients)
                        metrics["gradients/entropy_grad_norm"] = optax.global_norm(entropy_gradients)
                        

                        return (policy_state, critic_state, entropy_coefficient_state, replay_buffer, env_state, key), (env_state.info, metrics)
                        
                    key, subkey = jax.random.split(key)
                    learning_iteration_carry, info_and_optimization_metrics = jax.lax.scan(learning_iteration, (policy_state, critic_state, entropy_coefficient_state, replay_buffer, env_state, subkey), jnp.arange(self.nr_updates_per_logging_iteration))
                    policy_state, critic_state, entropy_coefficient_state, replay_buffer, env_state, key = learning_iteration_carry
                    infos, optimization_metrics = info_and_optimization_metrics
                    infos = {key: jnp.mean(infos[key]) for key in infos}
                    optimization_metrics = {key: jnp.mean(optimization_metrics[key]) for key in optimization_metrics}


                    # Logging
                    nr_update_iteration = (eval_save_iteration_step * self.nr_loggings_per_eval_save_iteration * self.nr_updates_per_logging_iteration) + (logging_iteration_step+1) * self.nr_updates_per_logging_iteration
                    steps_metrics = {
                        "steps/nr_env_steps": nr_update_iteration * self.nr_envs,
                        "steps/nr_updates": nr_update_iteration,
                    }

                    combined_metrics = {**infos, **steps_metrics, **optimization_metrics}
                    combined_metrics = tree.map_structure(lambda x: jnp.mean(x), combined_metrics)

                    def callback(carry):
                        metrics, parallel_seed_id = carry
                        current_time = time.time()
                        metrics["time/sps"] = int((self.nr_envs * self.nr_updates_per_logging_iteration) / (current_time - self.last_time[parallel_seed_id]))
                        self.last_time[parallel_seed_id] = current_time
                        global_step = int(metrics["steps/nr_env_steps"])
                        self.start_logging(global_step)
                        for key, value in metrics.items():
                            self.log(f"{key}", np.asarray(value), global_step)
                        self.end_logging()

                    jax.debug.callback(callback, (combined_metrics, parallel_seed_id))

                    return (policy_state, critic_state, entropy_coefficient_state, replay_buffer, env_state, key), None

                key, subkey = jax.random.split(key)
                logging_iteration_carry, _ = jax.lax.scan(logging_iteration, (policy_state, critic_state, entropy_coefficient_state, replay_buffer, env_state, subkey), jnp.arange(self.nr_loggings_per_eval_save_iteration))
                policy_state, critic_state, entropy_coefficient_state, replay_buffer, env_state, key = logging_iteration_carry


                # Evaluating
                ...
                

                # Saving
                ...

                
                return (policy_state, critic_state, entropy_coefficient_state, replay_buffer, env_state, key), None

            jax.lax.scan(eval_save_iteration, (policy_state, critic_state, entropy_coefficient_state, replay_buffer, env_state, key), jnp.arange(self.nr_eval_save_iterations))
            

        self.key, subkey = jax.random.split(self.key)
        seed_keys = jax.random.split(subkey, self.nr_parallel_seeds)
        train_function = jax.jit(jax.vmap(jitable_train_function))
        self.last_time = [time.time() for _ in range(self.nr_parallel_seeds)]
        self.start_time = deepcopy(self.last_time)
        jax.block_until_ready(train_function(seed_keys, jnp.arange(self.nr_parallel_seeds)))
        rlx_logger.info(f"Average time: {max([time.time() - t for t in self.start_time]):.2f} s")


    def log(self, name, value, step):
        if self.track_tb:
            self.writer.add_scalar(name, value, step)
        if self.track_console:
            self.log_console(name, value)
    

    def log_console(self, name, value):
        value = np.format_float_positional(value, trim="-")
        rlx_logger.info(f"│ {name.ljust(30)}│ {str(value).ljust(14)[:14]} │", flush=False)

    
    def start_logging(self, step):
        if self.track_console:
            rlx_logger.info("┌" + "─" * 31 + "┬" + "─" * 16 + "┐", flush=False)
        else:
            rlx_logger.info(f"Step: {step}")


    def end_logging(self):
        if self.track_console:
            rlx_logger.info("└" + "─" * 31 + "┴" + "─" * 16 + "┘")


    def save(self, type):
        ...


    def load(config, env, run_path, writer, explicitly_set_algorithm_params):
        ...
    

    def test(self, episodes):
        ...
    

    def set_train_mode(self):
        ...


    def set_eval_mode(self):
        ...


    def general_properties():
        return GeneralProperties
