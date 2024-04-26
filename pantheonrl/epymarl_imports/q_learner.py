import torch as th
from torch.optim import Adam
import copy
from typing import Type, Optional, Dict, Any
from .non_shared_controller import NonSharedMAC

# Assuming these components are correctly implemented in your environment
from .episode_buffer import EpisodeBatch
from .vdn import VDNMixer
from .qmix import QMixer
from .standarize_stream import RunningMeanStd
from stable_baselines3.common.base_class import BaseAlgorithm

class QLearner(BaseAlgorithm):
    def __init__(self,
                 mac: Type[NonSharedMAC],
                 scheme: Dict[str, Any],
                 logger,
                 args,
                 device: str = "auto",
                 verbose: int = 0):
        super().__init__(
            policy=None,  # Q-Learning does not use a separate 'policy' class
            env=None,  # Environment is typically passed separately if needed
            verbose=verbose,
            device=device)

        self.args = args
        self.mac = mac(scheme, args)
        self.logger = logger
        self.device = device if device != "auto" else ("cuda" if th.cuda.is_available() else "cpu")

        self.params = list(mac.parameters())
        self.mixer = None
        if args.mixer:
            self.mixer = VDNMixer() if args.mixer == "vdn" else QMixer(args)
            self.params += list(self.mixer.parameters())
            self.target_mixer = copy.deepcopy(self.mixer)

        self.optimiser = Adam(self.params, lr=args.lr)
        self.target_mac = copy.deepcopy(mac)
        self.training_steps = 0
        self.last_target_update_step = 0

        if args.standardise_returns:
            self.ret_ms = RunningMeanStd(shape=(args.n_agents,), device=self.device)
        if args.standardise_rewards:
            self.rew_ms = RunningMeanStd(shape=(1,), device=self.device)

    def _setup_model(self):
        """Setup required models and parameters."""
        # This is a placeholder to demonstrate where initial setup logic might go.
        self.target_mac = copy.deepcopy(self.mac)
        if self.mixer:
            self.target_mixer = copy.deepcopy(self.mixer)

    # def train(self, batch: EpisodeBatch, t_env: int, episode_num: int):
    #     # Placeholder for actual training logic
    #     pass

    def train(self, batch: EpisodeBatch, t_env: int, episode_num: int):
        # Extract the relevant quantities from the batch
        rewards = batch["reward"][:, :-1]
        actions = batch["actions"][:, :-1]
        terminated = batch["terminated"][:, :-1].float()
        mask = batch["filled"][:, :-1].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])  # Apply termination mask
        avail_actions = batch["avail_actions"]

        # Standardize rewards if required
        if self.args.standardise_rewards:
            self.rew_ms.update(rewards)  # Update the running mean/std for rewards
            rewards = (rewards - self.rew_ms.mean) / th.sqrt(self.rew_ms.var + 1e-8)

        # Compute Q values for all actions at all times
        self.mac.init_hidden(batch.batch_size)
        all_q_values = []
        for t in range(batch.max_seq_length):
            agent_outs = self.mac.forward(batch, t=t)
            all_q_values.append(agent_outs)
        all_q_values = th.stack(all_q_values, dim=1)  # Shape: [batch_size, seq_length, num_agents, num_actions]

        # Select the Q-values for the actions taken
        chosen_action_qvals = th.gather(all_q_values[:, :-1], dim=3, index=actions).squeeze(3)  # Remove last dim

        # Compute target Q values
        self.target_mac.init_hidden(batch.batch_size)
        target_q_values = []
        for t in range(1, batch.max_seq_length):  # Start from 1 to align with targets
            target_agent_outs = self.target_mac.forward(batch, t=t)
            target_q_values.append(target_agent_outs)
        target_q_values = th.stack(target_q_values, dim=1)

        # Mask out unavailable actions
        target_q_values[avail_actions[:, 1:] == 0] = -9999999  # Ensuring these are never selected

        # Compute max Q-value across actions for each agent at the next time step
        if self.args.double_q:
            # Double Q-Learning: Use the current network to select actions
            cur_q_values = all_q_values.clone().detach()
            cur_q_values[avail_actions == 0] = -9999999
            cur_max_actions = cur_q_values[:, 1:].max(dim=3, keepdim=True)[1]
            target_max_qvals = th.gather(target_q_values, 3, cur_max_actions).squeeze(3)
        else:
            # Regular Q-Learning
            target_max_qvals = target_q_values.max(dim=3)[0]

        # Optionally apply a mixer to both chosen and target Q-values
        if self.mixer is not None:
            chosen_action_qvals = self.mixer(chosen_action_qvals, batch["state"][:, :-1])
            target_max_qvals = self.target_mixer(target_max_qvals, batch["state"][:, 1:])

        # Calculate 1-step Q-Learning targets
        targets = rewards + self.args.gamma * (1 - terminated) * target_max_qvals.detach()

        # Standardize returns if required
        if self.args.standardise_returns:
            self.ret_ms.update(targets)
            targets = (targets - self.ret_ms.mean) / th.sqrt(self.ret_ms.var + 1e-8)

        # Compute TD-error
        td_error = (chosen_action_qvals - targets.detach())
        mask = mask.expand_as(td_error)  # Make sure mask is the same shape as td_error

        # Calculate the loss (mean squared error)
        masked_td_error = td_error * mask  # Apply mask
        loss = (masked_td_error ** 2).sum() / mask.sum()  # Normalize by the number of unmasked entries

        # Backpropagation of loss and optimization
        self.optimiser.zero_grad()
        loss.backward()
        grad_norm = th.nn.utils.clip_grad_norm_(self.params, self.args.grad_norm_clip)
        self.optimiser.step()

        # Log training information if necessary
        if t_env - self.log_stats_t >= self.args.learner_log_interval:
            self.logger.log_stat("loss", loss.item(), t_env)
            self.logger.log_stat("grad_norm", grad_norm.item(), t_env)
            mask_elems = mask.sum().item()
            self.logger.log_stat("td_error_abs", (masked_td_error.abs().sum().item() / mask_elems), t_env)
            self.logger.log_stat("q_taken_mean", (chosen_action_qvals * mask).sum().item() / (mask_elems), t_env)
            self.logger.log_stat("target_mean", (targets * mask).sum().item() / (mask_elems), t_env)
            self.log_stats_t = t_env  # Update the time of the last logging

    def _update_targets(self):
        """Updates the target network weights."""
        self.target_mac.load_state_dict(self.mac.state_dict())
        if self.mixer is not None:
            self.target_mixer.load_state_dict(self.mixer.state_dict())

    def save_models(self, path: str):
        """Save models to the given path."""
        self.mac.save("{}/mac.pth".format(path))
        if self.mixer is not None:
            th.save(self.mixer.state_dict(), "{}/mixer.pth".format(path))
        th.save(self.optimiser.state_dict(), "{}/optimizer.pth".format(path))

    def load_models(self, path: str):
        """Load models from the given path."""
        self.mac.load("{}/mac.pth".format(path))
        if self.mixer is not None:
            self.mixer.load_state_dict(th.load("{}/mixer.pth".format(path)))
        self.optimiser.load_state_dict(th.load("{}/optimizer.pth".format(path)))

