# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import os
import statistics
import time
import torch
from collections import deque
from tensordict import TensorDict

import rsl_rl
from rsl_rl.algorithms import Distillation
from rsl_rl.env import VecEnv
from rsl_rl.modules import StudentTeacher, StudentTeacherVision, StudentTeacherRecurrent
from rsl_rl.runners import OnPolicyRunner
from rsl_rl.utils import resolve_obs_groups, store_code_state


class DistillationRunner(OnPolicyRunner):
    """On-policy runner for training and evaluation of teacher-student training."""

    def __init__(self, env: VecEnv, train_cfg: dict, log_dir: str | None = None, device: str = "cpu") -> None:
        self.cfg = train_cfg
        self.alg_cfg = train_cfg["algorithm"]
        self.policy_cfg = train_cfg["policy"]
        self.device = device
        self.env = env

        # Check if multi-GPU is enabled
        self._configure_multi_gpu()

        # Store training configuration
        self.num_steps_per_env = self.cfg["num_steps_per_env"]
        self.save_interval = self.cfg["save_interval"]

        # Query observations from environment for algorithm construction
        obs = self.env.get_observations()
        self.cfg["obs_groups"] = resolve_obs_groups(obs, self.cfg["obs_groups"], default_sets=["teacher"])

        # Create the algorithm
        self.alg = self._construct_algorithm(obs)

        # Decide whether to disable logging
        # Note: We only log from the process with rank 0 (main process)
        self.disable_logs = self.is_distributed and self.gpu_global_rank != 0

        # Logging
        self.log_dir = log_dir
        self.writer = None
        self.tot_timesteps = 0
        self.tot_time = 0
        self.current_learning_iteration = 0
        self.git_status_repos = [rsl_rl.__file__]

    def _get_partition_boundary(self) -> int:
        """Number of student envs (first N envs). Teacher envs are the remaining."""
        n_teacher = int(self.env.num_envs * self.alg.teacher_act_ratio)
        return self.env.num_envs - n_teacher

    def _get_term_manager(self):
        """Access the underlying IsaacLab termination manager, or None."""
        try:
            return self.env.unwrapped.termination_manager
        except AttributeError:
            return None

    def learn(self, num_learning_iterations: int, init_at_random_ep_len: bool = False) -> None:
        # Initialize writer
        self._prepare_logging_writer()
        # Check if teacher is loaded
        if not self.alg.policy.loaded_teacher:
            raise ValueError("Teacher model parameters not loaded. Please load a teacher model to distill.")

        # Randomize initial episode lengths (for exploration)
        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(
                self.env.episode_length_buf, high=int(self.env.max_episode_length)
            )

        # Start learning
        obs = self.env.get_observations().to(self.device)
        self.train_mode()  # switch to train mode (for dropout for example)

        # Book keeping
        ep_infos = []
        rewbuffer = deque(maxlen=100)
        lenbuffer = deque(maxlen=100)
        cur_reward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        cur_episode_length = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)

        # Per-partition (student/teacher) book keeping
        student_boundary = self._get_partition_boundary()
        student_rewbuffer = deque(maxlen=100)
        teacher_rewbuffer = deque(maxlen=100)
        student_lenbuffer = deque(maxlen=100)
        teacher_lenbuffer = deque(maxlen=100)
        term_mgr = self._get_term_manager()
        term_names = list(term_mgr._term_names) if term_mgr is not None else []
        student_term_counts = {name: 0 for name in term_names}
        teacher_term_counts = {name: 0 for name in term_names}
        student_ep_count = 0
        teacher_ep_count = 0

        # Ensure all parameters are in-synced
        if self.is_distributed:
            print(f"Synchronizing parameters for rank {self.gpu_global_rank}...")
            self.alg.broadcast_parameters()

        # Start training
        start_iter = self.current_learning_iteration
        tot_iter = start_iter + num_learning_iterations
        for it in range(start_iter, tot_iter):
            start = time.time()

            # Reset per-iteration partition term counters
            for name in term_names:
                student_term_counts[name] = 0
                teacher_term_counts[name] = 0
            student_ep_count = 0
            teacher_ep_count = 0

            # Rollout
            with torch.inference_mode():
                for _ in range(self.num_steps_per_env):
                    # Sample actions
                    actions = self.alg.act(obs)
                    # Step the environment
                    obs, rewards, dones, extras = self.env.step(actions.to(self.env.device))
                    # Move to device
                    obs, rewards, dones = (obs.to(self.device), rewards.to(self.device), dones.to(self.device))
                    # Process the step
                    self.alg.process_env_step(obs, rewards, dones, extras)
                    # Book keeping
                    if self.log_dir is not None:
                        if "episode" in extras:
                            ep_infos.append(extras["episode"])
                        elif "log" in extras:
                            ep_infos.append(extras["log"])
                        # Update rewards
                        cur_reward_sum += rewards
                        # Update episode length
                        cur_episode_length += 1
                        # Clear data for completed episodes
                        new_ids = (dones > 0).nonzero(as_tuple=False)
                        rewbuffer.extend(cur_reward_sum[new_ids][:, 0].cpu().numpy().tolist())
                        lenbuffer.extend(cur_episode_length[new_ids][:, 0].cpu().numpy().tolist())

                        # Per-partition tracking
                        if new_ids.numel() > 0:
                            done_flat = new_ids[:, 0]
                            s_mask = done_flat < student_boundary
                            t_mask = ~s_mask
                            s_ids = done_flat[s_mask]
                            t_ids = done_flat[t_mask]
                            if s_ids.numel() > 0:
                                student_rewbuffer.extend(cur_reward_sum[s_ids].cpu().numpy().tolist())
                                student_lenbuffer.extend(cur_episode_length[s_ids].cpu().numpy().tolist())
                                student_ep_count += s_ids.numel()
                            if t_ids.numel() > 0:
                                teacher_rewbuffer.extend(cur_reward_sum[t_ids].cpu().numpy().tolist())
                                teacher_lenbuffer.extend(cur_episode_length[t_ids].cpu().numpy().tolist())
                                teacher_ep_count += t_ids.numel()
                            # Per-partition termination term tracking
                            if term_mgr is not None and hasattr(term_mgr, '_last_episode_dones'):
                                led = term_mgr._last_episode_dones  # [num_envs, num_terms]
                                for j, name in enumerate(term_names):
                                    if s_ids.numel() > 0:
                                        student_term_counts[name] += led[s_ids, j].float().sum().item()
                                    if t_ids.numel() > 0:
                                        teacher_term_counts[name] += led[t_ids, j].float().sum().item()

                        cur_reward_sum[new_ids] = 0
                        cur_episode_length[new_ids] = 0

                stop = time.time()
                collection_time = stop - start
                start = stop

            # Update policy
            loss_dict = self.alg.update()

            stop = time.time()
            learn_time = stop - start
            self.current_learning_iteration = it

            if self.log_dir is not None and not self.disable_logs:
                # Log information
                self.log(locals())
                # Save model
                if it % self.save_interval == 0:
                    self.save(os.path.join(self.log_dir, f"model_{it}.pt"))

            # Clear episode infos
            ep_infos.clear()
            # Save code state
            if it == start_iter and not self.disable_logs:
                # Obtain all the diff files
                git_file_paths = store_code_state(self.log_dir, self.git_status_repos)
                # If possible store them to wandb or neptune
                if self.logger_type in ["wandb", "neptune"] and git_file_paths:
                    for path in git_file_paths:
                        self.writer.save_file(path)

        # Save the final model after training
        if self.log_dir is not None and not self.disable_logs:
            self.save(os.path.join(self.log_dir, f"model_{self.current_learning_iteration}.pt"))

    def log(self, locs: dict, width: int = 80, pad: int = 35) -> None:
        """Extended logging with per-partition (student/teacher) metrics."""
        # Call parent log for standard metrics
        super().log(locs, width, pad)

        it = locs["it"]

        # --- Student partition ---
        if len(locs["student_rewbuffer"]) > 0:
            self.writer.add_scalar("Student/mean_reward", statistics.mean(locs["student_rewbuffer"]), it)
            self.writer.add_scalar("Student/mean_episode_length", statistics.mean(locs["student_lenbuffer"]), it)

        if locs["student_ep_count"] > 0:
            for name in locs["term_names"]:
                rate = locs["student_term_counts"][name] / locs["student_ep_count"]
                self.writer.add_scalar(f"Student/Episode_Termination/{name}", rate, it)

        # --- Teacher partition ---
        if len(locs["teacher_rewbuffer"]) > 0:
            self.writer.add_scalar("Teacher/mean_reward", statistics.mean(locs["teacher_rewbuffer"]), it)
            self.writer.add_scalar("Teacher/mean_episode_length", statistics.mean(locs["teacher_lenbuffer"]), it)

        if locs["teacher_ep_count"] > 0:
            for name in locs["term_names"]:
                rate = locs["teacher_term_counts"][name] / locs["teacher_ep_count"]
                self.writer.add_scalar(f"Teacher/Episode_Termination/{name}", rate, it)

    def _construct_algorithm(self, obs: TensorDict) -> Distillation:
        """Construct the distillation algorithm."""
        # Initialize the policy
        student_teacher_class = eval(self.policy_cfg.pop("class_name"))
        student_teacher: StudentTeacher | StudentTeacherVision | StudentTeacherRecurrent = student_teacher_class(
            obs, self.cfg["obs_groups"], self.env.num_actions, **self.policy_cfg
        ).to(self.device)

        # Initialize the algorithm
        alg_class = eval(self.alg_cfg.pop("class_name"))
        alg: Distillation = alg_class(
            student_teacher, device=self.device, **self.alg_cfg, multi_gpu_cfg=self.multi_gpu_cfg
        )

        # Initialize the storage
        alg.init_storage(
            "distillation",
            self.env.num_envs,
            self.num_steps_per_env,
            obs,
            [self.env.num_actions],
        )

        return alg
