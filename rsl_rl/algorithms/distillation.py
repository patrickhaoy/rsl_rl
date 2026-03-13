# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import torch
import torch.nn as nn
from tensordict import TensorDict
from torch.distributions import kl_divergence

from rsl_rl.modules import StudentTeacher, StudentTeacherRecurrent
from rsl_rl.storage import RolloutStorage
from rsl_rl.utils import resolve_optimizer


class Distillation:
    """Distillation algorithm for training a student model to mimic a teacher model."""

    policy: StudentTeacher | StudentTeacherRecurrent
    """The student teacher model."""

    def __init__(
        self,
        policy: StudentTeacher | StudentTeacherRecurrent,
        num_learning_epochs: int = 1,
        num_mini_batches: int | None = None,
        gradient_length: int = 15,
        learning_rate: float = 1e-3,
        max_grad_norm: float | None = None,
        loss_type: str = "mse",
        auxiliary_loss_weight: float = 1.0,
        teacher_act_ratio: float = 0.0,
        optimizer: str = "adam",
        device: str = "cpu",
        # Distributed training parameters
        multi_gpu_cfg: dict | None = None,
    ) -> None:
        # Device-related parameters
        self.device = device
        self.is_multi_gpu = multi_gpu_cfg is not None

        # Multi-GPU parameters
        if multi_gpu_cfg is not None:
            self.gpu_global_rank = multi_gpu_cfg["global_rank"]
            self.gpu_world_size = multi_gpu_cfg["world_size"]
        else:
            self.gpu_global_rank = 0
            self.gpu_world_size = 1

        # Distillation components
        self.policy = policy
        self.policy.to(self.device)
        self.storage = None  # Initialized later

        # Initialize the optimizer
        self.optimizer = resolve_optimizer(optimizer)(self.policy.parameters(), lr=learning_rate)

        # Initialize the transition
        self.transition = RolloutStorage.Transition()
        self.last_hidden_states = (None, None)

        # Distillation parameters
        self.num_learning_epochs = num_learning_epochs
        self.num_mini_batches = num_mini_batches
        self.gradient_length = gradient_length
        self.learning_rate = learning_rate
        self.max_grad_norm = max_grad_norm
        self.auxiliary_loss_weight = auxiliary_loss_weight
        self.teacher_act_ratio = teacher_act_ratio

        # Initialize the loss function
        self.loss_type = loss_type
        loss_fn_dict = {
            "mse": nn.functional.mse_loss,
            "huber": nn.functional.huber_loss,
        }
        if loss_type == "kl":
            self.loss_fn = None  # KL handled separately via distributions
        elif loss_type in loss_fn_dict:
            self.loss_fn = loss_fn_dict[loss_type]
        else:
            raise ValueError(f"Unknown loss type: {loss_type}. Supported: {list(loss_fn_dict.keys()) + ['kl']}")

        self.num_updates = 0

    def init_storage(
        self,
        training_type: str,
        num_envs: int,
        num_transitions_per_env: int,
        obs: TensorDict,
        actions_shape: tuple[int],
    ) -> None:
        # Create rollout storage
        self.storage = RolloutStorage(
            training_type,
            num_envs,
            num_transitions_per_env,
            obs,
            actions_shape,
            self.device,
        )

    def act(self, obs: TensorDict) -> torch.Tensor:
        student_actions = self.policy.act(obs).detach()
        teacher_actions = self.policy.evaluate(obs).detach()

        # DAgger-BC mixture: teacher_act_ratio fraction of envs use teacher actions,
        # remaining use student actions. Teacher labels are recorded for ALL envs.
        if self.teacher_act_ratio > 0.0:
            actions = student_actions.clone()
            n_teacher = int(actions.shape[0] * self.teacher_act_ratio)
            if n_teacher > 0:
                actions[-n_teacher:] = teacher_actions[-n_teacher:]
        else:
            actions = student_actions

        self.transition.actions = actions
        self.transition.privileged_actions = teacher_actions
        self.transition.observations = obs
        return actions

    def process_env_step(
        self, obs: TensorDict, rewards: torch.Tensor, dones: torch.Tensor, extras: dict[str, torch.Tensor]
    ) -> None:
        # Update the normalizers
        self.policy.update_normalization(obs)

        # Record the rewards and dones
        self.transition.rewards = rewards
        self.transition.dones = dones
        # Record the transition
        self.storage.add_transitions(self.transition)
        self.transition.clear()
        self.policy.reset(dones)

    def update(self) -> dict[str, float]:
        self.num_updates += 1
        mean_behavior_loss = 0
        mean_auxiliary_loss = 0
        cnt = 0

        if self.num_mini_batches is not None:
            mean_behavior_loss, mean_auxiliary_loss, cnt = self._update_feedforward()
        else:
            mean_behavior_loss, mean_auxiliary_loss, cnt = self._update_recurrent()

        mean_behavior_loss /= cnt
        mean_auxiliary_loss /= cnt
        self.storage.clear()
        self.last_hidden_states = self.policy.get_hidden_states()
        self.policy.detach_hidden_states()

        return {"behavior": mean_behavior_loss, "auxiliary": mean_auxiliary_loss}

    def _update_feedforward(self) -> tuple[float, float, int]:
        """Shuffled mini-batch update for feedforward (non-recurrent) student policies."""
        mean_behavior_loss = 0
        mean_auxiliary_loss = 0
        cnt = 0

        for obs_batch, privileged_actions_batch in self.storage.distillation_mini_batch_generator(
            self.num_mini_batches, self.num_learning_epochs
        ):
            if self.loss_type == "kl":
                student_dist = self.policy.get_student_distribution(obs_batch)
                with torch.no_grad():
                    teacher_dist = self.policy.get_teacher_distribution(obs_batch)
                behavior_loss = kl_divergence(teacher_dist, student_dist).sum(dim=-1).mean()
            else:
                actions = self.policy.act_inference(obs_batch)
                behavior_loss = self.loss_fn(actions, privileged_actions_batch)

            if hasattr(self.policy, 'auxiliary_head') and self.policy.auxiliary_head is not None:
                aux_pred = self.policy.predict_auxiliary(obs_batch)
                aux_loss_fn = self.loss_fn if self.loss_fn is not None else nn.functional.mse_loss
                auxiliary_loss = aux_loss_fn(aux_pred, obs_batch['auxiliary'])
                mean_auxiliary_loss += auxiliary_loss.item()
            else:
                auxiliary_loss = 0.0

            loss = behavior_loss + self.auxiliary_loss_weight * auxiliary_loss
            mean_behavior_loss += behavior_loss.item()
            cnt += 1

            self.optimizer.zero_grad()
            loss.backward()
            if self.is_multi_gpu:
                self.reduce_parameters()
            if self.max_grad_norm:
                nn.utils.clip_grad_norm_(self.policy.student.parameters(), self.max_grad_norm)
            self.optimizer.step()

        return mean_behavior_loss, mean_auxiliary_loss, cnt

    def _update_recurrent(self) -> tuple[float, float, int]:
        """Sequential update for recurrent student policies (preserves temporal ordering)."""
        mean_behavior_loss = 0
        mean_auxiliary_loss = 0
        loss = 0
        cnt = 0

        for epoch in range(self.num_learning_epochs):
            self.policy.reset(hidden_states=self.last_hidden_states)
            self.policy.detach_hidden_states()
            for obs, _, privileged_actions, dones in self.storage.generator():
                if self.loss_type == "kl":
                    student_dist = self.policy.get_student_distribution(obs)
                    with torch.no_grad():
                        teacher_dist = self.policy.get_teacher_distribution(obs)
                    behavior_loss = kl_divergence(teacher_dist, student_dist).sum(dim=-1).mean()
                else:
                    actions = self.policy.act_inference(obs)
                    behavior_loss = self.loss_fn(actions, privileged_actions)

                if hasattr(self.policy, 'auxiliary_head') and self.policy.auxiliary_head is not None:
                    aux_pred = self.policy.predict_auxiliary(obs)
                    aux_loss_fn = self.loss_fn if self.loss_fn is not None else nn.functional.mse_loss
                    auxiliary_loss = aux_loss_fn(aux_pred, obs['auxiliary'])
                    mean_auxiliary_loss += auxiliary_loss.item()
                else:
                    auxiliary_loss = 0.0

                loss = loss + behavior_loss + self.auxiliary_loss_weight * auxiliary_loss
                mean_behavior_loss += behavior_loss.item()
                cnt += 1

                if cnt % self.gradient_length == 0:
                    self.optimizer.zero_grad()
                    loss.backward()
                    if self.is_multi_gpu:
                        self.reduce_parameters()
                    if self.max_grad_norm:
                        nn.utils.clip_grad_norm_(self.policy.student.parameters(), self.max_grad_norm)
                    self.optimizer.step()
                    self.policy.detach_hidden_states()
                    loss = 0

                self.policy.reset(dones.view(-1))
                self.policy.detach_hidden_states(dones.view(-1))

        return mean_behavior_loss, mean_auxiliary_loss, cnt

    def broadcast_parameters(self) -> None:
        """Broadcast model parameters to all GPUs."""
        # Obtain the model parameters on current GPU
        model_params = [self.policy.state_dict()]
        # Broadcast the model parameters
        torch.distributed.broadcast_object_list(model_params, src=0)
        # Load the model parameters on all GPUs from source GPU
        self.policy.load_state_dict(model_params[0])

    def reduce_parameters(self) -> None:
        """Collect gradients from all GPUs and average them.

        This function is called after the backward pass to synchronize the gradients across all GPUs.
        """
        # Create a tensor to store the gradients
        grads = [param.grad.view(-1) for param in self.policy.parameters() if param.grad is not None]
        all_grads = torch.cat(grads)
        # Average the gradients across all GPUs
        torch.distributed.all_reduce(all_grads, op=torch.distributed.ReduceOp.SUM)
        all_grads /= self.gpu_world_size
        # Update the gradients for all parameters with the reduced gradients
        offset = 0
        for param in self.policy.parameters():
            if param.grad is not None:
                numel = param.numel()
                # Copy data back from shared buffer
                param.grad.data.copy_(all_grads[offset : offset + numel].view_as(param.grad.data))
                # Update the offset for the next parameter
                offset += numel
