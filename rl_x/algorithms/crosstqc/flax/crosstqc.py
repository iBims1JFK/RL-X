import os
import logging
import time
from collections import deque
import tree
import numpy as np
import jax
from jax.lax import stop_gradient
import jax.numpy as jnp
import flax
from flax.training.train_state import TrainState
from flax.training import orbax_utils
import orbax.checkpoint
import optax
import wandb

from rl_x.algorithms.crosstqc.flax.general_properties import GeneralProperties
from rl_x.algorithms.crosstqc.flax.policy import get_policy
from rl_x.algorithms.crosstqc.flax.critic import get_critic
from rl_x.algorithms.crosstqc.flax.entropy_coefficient import EntropyCoefficient
from rl_x.algorithms.crosstqc.flax.replay_buffer import ReplayBuffer
from rl_x.algorithms.crosstqc.flax.rl_train_state import RLTrainState

rlx_logger = logging.getLogger("rl_x")


class CrossTQC:
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
        self.nr_envs = config.environment.nr_envs
        self.learning_rate = config.algorithm.learning_rate
        self.anneal_learning_rate = config.algorithm.anneal_learning_rate
        self.policy_adam_b1 = config.algorithm.policy_adam_b1
        self.critic_adam_b1 = config.algorithm.critic_adam_b1
        self.buffer_size = config.algorithm.buffer_size
        self.learning_starts = config.algorithm.learning_starts
        self.batch_size = config.algorithm.batch_size
        self.gamma = config.algorithm.gamma
        self.policy_delay = config.algorithm.policy_delay
        self.target_entropy = config.algorithm.target_entropy
        self.logging_frequency = config.algorithm.logging_frequency
        self.evaluation_frequency = config.algorithm.evaluation_frequency
        self.evaluation_episodes = config.algorithm.evaluation_episodes

        rlx_logger.info(f"Using device: {jax.default_backend()}")
        
        self.rng = np.random.default_rng(self.seed)
        self.key = jax.random.PRNGKey(self.seed)
        self.key, policy_key, policy_batch_renorm_key, critic_key, critic_batch_renorm_key, entropy_coefficient_key \
            = jax.random.split(self.key, 6)

        self.env_as_low = env.single_action_space.low
        self.env_as_high = env.single_action_space.high

        self.policy, self.get_processed_action = get_policy(config, env)
        self.critic = get_critic(config, env)
        
        if self.target_entropy == "auto":
            self.target_entropy = -np.prod(env.single_action_space.shape).item()
        else:
            self.target_entropy = float(self.target_entropy)
        self.entropy_coefficient = EntropyCoefficient(1.0)

        self.entropy_coefficient.apply = jax.jit(self.entropy_coefficient.apply)

        def linear_schedule(step):
            total_steps = self.total_timesteps
            fraction = 1.0 - (step / total_steps)
            return self.learning_rate * fraction
        
        self.q_learning_rate = linear_schedule if self.anneal_learning_rate else self.learning_rate
        self.policy_learning_rate = linear_schedule if self.anneal_learning_rate else self.learning_rate
        self.entropy_learning_rate = linear_schedule if self.anneal_learning_rate else self.learning_rate

        state = jnp.array([self.env.single_observation_space.sample()])
        action = jnp.array([self.env.single_action_space.sample()])

        policy_init = self.policy.init(
            {"params": policy_key, "batch_stats": policy_batch_renorm_key},
            state, train=False
        )
        self.policy_state = RLTrainState.create(
            apply_fn=self.policy.apply,
            params=policy_init["params"],
            batch_stats=policy_init["batch_stats"],
            tx=optax.inject_hyperparams(optax.adam)(
                learning_rate=self.policy_learning_rate,
                b1=self.policy_adam_b1,
            )
        )

        critic_init = self.critic.init(
            {"params": critic_key, "batch_stats": critic_batch_renorm_key},
            state, action, train=False
        )
        self.critic_state = RLTrainState.create(
            apply_fn=self.critic.apply,
            params=critic_init["params"],
            batch_stats=critic_init["batch_stats"],
            tx=optax.inject_hyperparams(optax.adam)(
                learning_rate=self.q_learning_rate,
                b1=self.critic_adam_b1,
            )
        )

        self.entropy_coefficient_state = TrainState.create(
            apply_fn=self.entropy_coefficient.apply,
            params=self.entropy_coefficient.init(entropy_coefficient_key),
            tx=optax.inject_hyperparams(optax.adam)(learning_rate=self.entropy_learning_rate)
        )

        if self.save_model:
            os.makedirs(self.save_path)
            self.best_mean_return = -np.inf
            self.best_model_file_name = "model_best_jax"
            best_model_check_point_handler = orbax.checkpoint.PyTreeCheckpointHandler(aggregate_filename=self.best_model_file_name)
            self.best_model_checkpointer = orbax.checkpoint.Checkpointer(best_model_check_point_handler)
        
    
    def train(self):
        @jax.jit
        def get_action(policy_state: TrainState, state: np.ndarray, key: jax.random.PRNGKey):
            dist = self.policy.apply(
                {"params": policy_state.params, "batch_stats": policy_state.batch_stats},
                state, train=False
            )
            key, subkey = jax.random.split(key)
            action = dist.sample(seed=subkey)
            return action, key


        @jax.jit
        def update_critic(
                policy_state: TrainState, critic_state: TrainState, entropy_coefficient_state: TrainState,
                states: np.ndarray, next_states: np.ndarray, actions: np.ndarray, rewards: np.ndarray, terminations: np.ndarray, key: jax.random.PRNGKey
            ):
            def loss_fn(critic_params: flax.core.FrozenDict,
                        state: np.ndarray, next_state: np.ndarray, action: np.ndarray, reward: np.ndarray, terminated: np.ndarray,
                        subkey: jax.random.PRNGKey
                ):
                # Critic loss
                dist = self.policy.apply(
                    {"params": policy_state.params, "batch_stats": policy_state.batch_stats},
                    next_state, train=False
                )
                next_action = dist.sample(seed=subkey)
                next_log_prob = dist.log_prob(next_action)

                alpha = self.entropy_coefficient.apply(entropy_coefficient_state.params)

                current_and_next_q, critic_state_update = self.critic.apply(
                    {"params": critic_params, "batch_stats": critic_state.batch_stats},
                    jnp.concatenate([state, next_state], axis=0),
                    jnp.concatenate([action, next_action], axis=0),
                    train=True, mutable=["batch_stats"]
                )
                q, next_q = jnp.split(jnp.squeeze(current_and_next_q, 2), 2, axis=1)

                min_next_q_target = jnp.min(stop_gradient(next_q), axis=0)

                y = reward + self.gamma * (1 - terminated) * (min_next_q_target - alpha * next_log_prob)

                q_loss = jnp.mean((q - y) ** 2)

                # Create metrics
                metrics = {
                    "loss/q_loss": q_loss,
                }

                return q_loss, (critic_state_update, metrics)


            grad_loss_fn = jax.value_and_grad(loss_fn, argnums=(0,), has_aux=True)

            key, subkey = jax.random.split(key, 2)

            (loss, (critic_state_update, metrics)), (critic_gradients,) = grad_loss_fn(
                critic_state.params,
                states, next_states, actions, rewards, terminations, subkey)

            critic_state = critic_state.apply_gradients(grads=critic_gradients)

            critic_state = critic_state.replace(batch_stats=critic_state_update["batch_stats"])

            metrics["lr/learning_rate"] = critic_state.opt_state.hyperparams["learning_rate"]
            metrics["gradients/critic_grad_norm"] = optax.global_norm(critic_gradients)

            return critic_state, metrics, key
        

        @jax.jit
        def update_policy_and_entropy_coefficient(
                policy_state: TrainState, critic_state: TrainState, entropy_coefficient_state: TrainState,
                states: np.ndarray, key: jax.random.PRNGKey
            ):
            def loss_fn(policy_params: flax.core.FrozenDict, entropy_coefficient_params: flax.core.FrozenDict,
                        state: np.ndarray, subkey: jax.random.PRNGKey
                ):
                # Policy loss
                dist, policy_state_update = self.policy.apply(
                    {"params": policy_params, "batch_stats": policy_state.batch_stats},
                    state, train=True, mutable=["batch_stats"]
                )
                current_action = dist.sample(seed=subkey)
                current_log_prob = dist.log_prob(current_action)
                entropy = stop_gradient(-current_log_prob)

                q = self.critic.apply(
                    {"params": critic_state.params, "batch_stats": critic_state.batch_stats},
                    state, current_action, train=False
                )
                min_q = jnp.min(jnp.squeeze(q, 2), axis=0)

                alpha_with_grad = self.entropy_coefficient.apply(entropy_coefficient_params)
                alpha = stop_gradient(alpha_with_grad)

                policy_loss = jnp.mean(alpha * current_log_prob - min_q)

                # Entropy loss
                entropy_loss = jnp.mean(alpha_with_grad * (entropy - self.target_entropy))

                # Combine losses
                loss = policy_loss + entropy_loss

                # Create metrics
                metrics = {
                    "loss/policy_loss": policy_loss,
                    "loss/entropy_loss": entropy_loss,
                    "entropy/entropy": jnp.mean(entropy),
                    "entropy/alpha": jnp.mean(alpha),
                    "q_value/q_value": jnp.mean(min_q),
                }

                return loss, (policy_state_update, metrics)
            

            grad_loss_fn = jax.value_and_grad(loss_fn, argnums=(0, 1), has_aux=True)

            key, subkey = jax.random.split(key, 2)

            (loss, (policy_state_update, metrics)), (policy_gradients, entropy_gradients) = grad_loss_fn(
                policy_state.params, entropy_coefficient_state.params, states, subkey)

            policy_state = policy_state.apply_gradients(grads=policy_gradients)
            entropy_coefficient_state = entropy_coefficient_state.apply_gradients(grads=entropy_gradients)

            policy_state = policy_state.replace(batch_stats=policy_state_update["batch_stats"])

            metrics["gradients/policy_grad_norm"] = optax.global_norm(policy_gradients)
            metrics["gradients/entropy_grad_norm"] = optax.global_norm(entropy_gradients)

            return policy_state, entropy_coefficient_state, metrics, key
        

        @jax.jit
        def get_deterministic_action(policy_state: TrainState, state: np.ndarray):
            dist = self.policy.apply(
                {"params": policy_state.params, "batch_stats": policy_state.batch_stats},
                state, train=False
            )
            action = dist.mode()
            return self.get_processed_action(action)


        self.set_train_mode()

        replay_buffer = ReplayBuffer(int(self.buffer_size), self.nr_envs, self.env.single_observation_space.shape, self.env.single_action_space.shape, self.rng)

        saving_return_buffer = deque(maxlen=100 * self.nr_envs)

        state, _ = self.env.reset()
        global_step = 0
        nr_critic_updates = 0
        nr_policy_updates = 0
        nr_episodes = 0
        time_metrics_collection = {}
        step_info_collection = {}
        optimization_metrics_collection = {}
        evaluation_metrics_collection = {}
        steps_metrics = {}
        while global_step < self.total_timesteps:
            start_time = time.time()


            # Acting
            dones_this_rollout = 0
            if global_step < self.learning_starts:
                processed_action = np.array([self.env.single_action_space.sample() for _ in range(self.nr_envs)])
                action = (processed_action - self.env_as_low) / (self.env_as_high - self.env_as_low) * 2.0 - 1.0
            else:
                action, self.key = get_action(self.policy_state, state, self.key)
                processed_action = self.get_processed_action(action)
            
            next_state, reward, terminated, truncated, info = self.env.step(jax.device_get(processed_action))
            done = terminated | truncated
            actual_next_state = next_state.copy()
            for i, single_done in enumerate(done):
                if single_done:
                    actual_next_state[i] = self.env.get_final_observation_at_index(info, i)
                    saving_return_buffer.append(self.env.get_final_info_value_at_index(info, "episode_return", i))
                    dones_this_rollout += 1
            for key, info_value in self.env.get_logging_info_dict(info).items():
                step_info_collection.setdefault(key, []).extend(info_value)
            
            replay_buffer.add(state, actual_next_state, action, reward, terminated)

            state = next_state
            global_step += self.nr_envs
            nr_episodes += dones_this_rollout

            acting_end_time = time.time()
            time_metrics_collection.setdefault("time/acting_time", []).append(acting_end_time - start_time)


            # What to do in this step after acting
            should_learning_start = global_step > self.learning_starts
            should_optimize_critic = should_learning_start
            should_optimize_policy = should_learning_start and global_step % self.policy_delay == 0
            should_evaluate = global_step % self.evaluation_frequency == 0 and self.evaluation_frequency != -1
            should_try_to_save = should_learning_start and self.save_model and dones_this_rollout > 0
            should_log = global_step % self.logging_frequency == 0


            # Optimizing - Prepare batches
            if should_optimize_critic:
                batch_states, batch_next_states, batch_actions, batch_rewards, batch_terminations = replay_buffer.sample(self.batch_size)


            # Optimizing - Q-functions
            if should_optimize_critic:
                self.critic_state, optimization_metrics, self.key = update_critic(self.policy_state, self.critic_state, self.entropy_coefficient_state, batch_states, batch_next_states, batch_actions, batch_rewards, batch_terminations, self.key)
                for key, value in optimization_metrics.items():
                    optimization_metrics_collection.setdefault(key, []).append(value)
                nr_critic_updates += 1


            # Optimizing - Policy and entropy coefficient
            if should_optimize_policy:
                self.policy_state, self.entropy_coefficient_state, optimization_metrics, self.key = update_policy_and_entropy_coefficient(self.policy_state, self.critic_state, self.entropy_coefficient_state, batch_states, self.key)
                for key, value in optimization_metrics.items():
                    optimization_metrics_collection.setdefault(key, []).append(value)
                nr_policy_updates += 1
            
            optimizing_end_time = time.time()
            time_metrics_collection.setdefault("time/optimizing_time", []).append(optimizing_end_time - acting_end_time)


            # Evaluating
            if should_evaluate:
                self.set_eval_mode()
                state, _ = self.env.reset()
                eval_nr_episodes = 0
                while True:
                    processed_action = get_deterministic_action(self.policy_state, state)
                    state, reward, terminated, truncated, info = self.env.step(jax.device_get(processed_action))
                    done = terminated | truncated
                    for i, single_done in enumerate(done):
                        if single_done:
                            eval_nr_episodes += 1
                            evaluation_metrics_collection.setdefault("eval/episode_return", []).append(self.env.get_final_info_value_at_index(info, "episode_return", i))
                            evaluation_metrics_collection.setdefault("eval/episode_length", []).append(self.env.get_final_info_value_at_index(info, "episode_length", i))
                            if eval_nr_episodes == self.evaluation_episodes:
                                break
                    if eval_nr_episodes == self.evaluation_episodes:
                        break
                state, _ = self.env.reset()
                self.set_train_mode()
            
            evaluating_end_time = time.time()
            time_metrics_collection.setdefault("time/evaluating_time", []).append(evaluating_end_time - optimizing_end_time)


            # Saving
            if should_try_to_save:
                mean_return = np.mean(saving_return_buffer)
                if mean_return > self.best_mean_return:
                    self.best_mean_return = mean_return
                    self.save()
            
            saving_end_time = time.time()
            time_metrics_collection.setdefault("time/saving_time", []).append(saving_end_time - evaluating_end_time)

            time_metrics_collection.setdefault("time/sps", []).append(self.nr_envs / (saving_end_time - start_time))


            # Logging
            if should_log:
                self.start_logging(global_step)

                steps_metrics["steps/nr_env_steps"] = global_step
                steps_metrics["steps/nr_critic_updates"] = nr_critic_updates
                steps_metrics["steps/nr_policy_updates"] = nr_policy_updates
                steps_metrics["steps/nr_episodes"] = nr_episodes

                rollout_info_metrics = {}
                env_info_metrics = {}
                if step_info_collection:
                    info_names = list(step_info_collection.keys())
                    for info_name in info_names:
                        metric_group = "rollout" if info_name in ["episode_return", "episode_length"] else "env_info"
                        metric_dict = rollout_info_metrics if metric_group == "rollout" else env_info_metrics
                        mean_value = np.mean(step_info_collection[info_name])
                        if mean_value == mean_value:  # Check if mean_value is NaN
                            metric_dict[f"{metric_group}/{info_name}"] = mean_value
                
                time_metrics = {key: np.mean(value) for key, value in time_metrics_collection.items()}
                optimization_metrics = {key: np.mean(value) for key, value in optimization_metrics_collection.items()}
                evaluation_metrics = {key: np.mean(value) for key, value in evaluation_metrics_collection.items()}
                combined_metrics = {**rollout_info_metrics, **evaluation_metrics, **env_info_metrics, **steps_metrics, **time_metrics, **optimization_metrics}
                for key, value in combined_metrics.items():
                    self.log(f"{key}", value, global_step)

                time_metrics_collection = {}
                step_info_collection = {}
                optimization_metrics_collection = {}
                evaluation_metrics_collection = {}

                self.end_logging()


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


    def save(self):
        checkpoint = {
            "config_algorithm": self.config.algorithm.to_dict(),
            "policy": self.policy_state,
            "critic": self.critic_state,
            "entropy_coefficient": self.entropy_coefficient_state,           
        }
        save_args = orbax_utils.save_args_from_target(checkpoint)
        self.best_model_checkpointer.save(f"{self.save_path}/tmp", checkpoint, save_args=save_args)
        os.rename(f"{self.save_path}/tmp/{self.best_model_file_name}", f"{self.save_path}/{self.best_model_file_name}")
        os.remove(f"{self.save_path}/tmp/_METADATA")
        os.rmdir(f"{self.save_path}/tmp")

        if self.track_wandb:
            wandb.save(f"{self.save_path}/{self.best_model_file_name}", base_path=self.save_path)


    def load(config, env, run_path, writer, explicitly_set_algorithm_params):
        splitted_path = config.runner.load_model.split("/")
        checkpoint_dir = "/".join(splitted_path[:-1])
        checkpoint_file_name = splitted_path[-1]

        check_point_handler = orbax.checkpoint.PyTreeCheckpointHandler(aggregate_filename=checkpoint_file_name)
        checkpointer = orbax.checkpoint.Checkpointer(check_point_handler)

        loaded_algorithm_config = checkpointer.restore(checkpoint_dir)["config_algorithm"]
        for key, value in loaded_algorithm_config.items():
            if f"algorithm.{key}" not in explicitly_set_algorithm_params:
                config.algorithm[key] = value
        model = CrossTQC(config, env, run_path, writer)

        target = {
            "config_algorithm": config.algorithm.to_dict(),
            "policy": model.policy_state,
            "critic": model.critic_state,
            "entropy_coefficient": model.entropy_coefficient_state
        }
        checkpoint = checkpointer.restore(checkpoint_dir, item=target)

        model.policy_state = checkpoint["policy"]
        model.critic_state = checkpoint["critic"]
        model.entropy_coefficient_state = checkpoint["entropy_coefficient"]

        return model
    

    def test(self, episodes):
        @jax.jit
        def get_action(policy_state: TrainState, state: np.ndarray):
            dist = self.policy.apply(
                {"params": policy_state.params, "batch_stats": policy_state.batch_stats},
                state, train=False
            )
            action = dist.mode()
            return self.get_processed_action(action)
        
        self.set_eval_mode()
        for i in range(episodes):
            done = False
            episode_return = 0
            state, _ = self.env.reset()
            while not done:
                processed_action = get_action(self.policy_state, state)
                state, reward, terminated, truncated, info = self.env.step(jax.device_get(processed_action))
                done = terminated | truncated
                episode_return += reward
            rlx_logger.info(f"Episode {i + 1} - Return: {episode_return}")
    

    def set_train_mode(self):
        ...


    def set_eval_mode(self):
        ...


    def general_properties():
        return GeneralProperties
