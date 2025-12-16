# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
import torch.nn as nn
import torchvision
from tensordict import TensorDict
from torch.distributions import Normal
from typing import Any

from rsl_rl.networks import MLP, EmpiricalNormalization
from rsl_rl.modules.student_teacher import StudentTeacher
from rsl_rl.modules.actor_critic import GSDENoiseDistribution


class StudentTeacherVision(StudentTeacher):
    """StudentTeacher with ResNet image encoders for visual observations.

    Automatically detects 4D tensors in policy observations as images and
    creates ResNet18 encoders for them.
    """

    def __init__(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        num_actions: int,
        student_obs_normalization: bool = False,
        teacher_obs_normalization: bool = False,
        student_hidden_dims: tuple[int] | list[int] = [256, 256, 256],
        teacher_hidden_dims: tuple[int] | list[int] = [256, 256, 256],
        activation: str = "elu",
        init_noise_std: float = 0.1,
        noise_std_type: str = "scalar",
        **kwargs: dict[str, Any],
    ) -> None:
        # Must call nn.Module.__init__ first before assigning any nn.Module
        nn.Module.__init__(self)

        # Separate image keys (4D) from low-dim keys
        self.image_keys = []
        self.low_dim_keys = []
        num_low_dim = 0
        for obs_group in obs_groups["policy"]:
            for key in obs[obs_group].keys():
                if obs[obs_group][key].dim() == 4:
                    self.image_keys.append(key)
                elif obs[obs_group][key].dim() == 2:
                    self.low_dim_keys.append(key)
                    num_low_dim += obs[obs_group][key].shape[-1]
                else:
                    raise ValueError(f"Unknown observation dimension: {obs[obs_group][key].dim()}")
        self.image_keys = sorted(self.image_keys)
        self.low_dim_keys = sorted(self.low_dim_keys)

        # Build image encoders
        self.image_encoders = nn.ModuleDict()
        self.image_feature_dim = 0

        if self.image_keys:
            for key in self.image_keys:
                backbone = torchvision.models.resnet18(weights=None)
                self.image_feature_dim = backbone.fc.in_features
                backbone.fc = nn.Identity()
                self.image_encoders[key] = backbone
            print(f"Detected image observations: {self.image_keys}")

        # Calculate student input dim
        num_image_features = len(self.image_keys) * self.image_feature_dim
        num_student_obs = num_image_features + num_low_dim
        print(f"Student input: {len(self.image_keys)} images * {self.image_feature_dim} + {num_low_dim} low_dim = {num_student_obs}")

        self.loaded_teacher = False
        self.obs_groups = obs_groups

        # ImageNet normalization
        self.register_buffer('imagenet_mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('imagenet_std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

        # Student MLP (takes encoded images + low_dim)
        self.student = MLP(num_student_obs, num_actions, student_hidden_dims, activation)
        print(f"Student MLP: {self.student}")

        # Student normalization (only for low_dim, not image features)
        self.student_obs_normalization = student_obs_normalization
        self.num_low_dim = num_low_dim
        if student_obs_normalization and num_low_dim > 0:
            self.student_obs_normalizer = EmpiricalNormalization(num_low_dim)
        else:
            self.student_obs_normalizer = nn.Identity()

        # Teacher (state-based)
        num_teacher_obs = sum(obs[k].shape[-1] for k in obs_groups["teacher"])
        self.teacher = MLP(num_teacher_obs, num_actions, teacher_hidden_dims, activation)
        self.teacher.eval()
        print(f"Teacher MLP: {self.teacher}")

        # Teacher normalization
        self.teacher_obs_normalization = teacher_obs_normalization
        if teacher_obs_normalization:
            self.teacher_obs_normalizer = EmpiricalNormalization(num_teacher_obs)
        else:
            self.teacher_obs_normalizer = nn.Identity()

        # Action noise
        self.noise_std_type = noise_std_type
        if noise_std_type == "scalar":
            self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
            self.teacher_std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        elif noise_std_type == "log":
            self.log_std = nn.Parameter(torch.log(init_noise_std * torch.ones(num_actions)))
            self.teacher_log_std = nn.Parameter(torch.log(init_noise_std * torch.ones(num_actions)))
        elif noise_std_type == "gsde":
            self.log_std = nn.Parameter(
                torch.ones(student_hidden_dims[-1], num_actions) * torch.log(torch.tensor(init_noise_std))
            )
            self.teacher_log_std = nn.Parameter(
                torch.ones(teacher_hidden_dims[-1], num_actions) * torch.log(torch.tensor(init_noise_std))
            )
        else:
            raise ValueError(f"Unknown noise_std_type: {noise_std_type}. Should be 'scalar', 'log', or 'gsde'")

        # Action distribution
        if noise_std_type == "gsde":
            self.distribution = GSDENoiseDistribution(action_dim=num_actions)
            self.distribution.sample_weights(self.log_std)
        else:
            self.distribution = None
        Normal.set_default_validate_args(False)

    def _encode_images(self, obs: TensorDict) -> torch.Tensor | None:
        """Encode images through ResNet backbones."""
        if not self.image_keys:
            return None
        features = []
        for key in self.image_keys:
            img = obs[key]
            if img.shape[-1] == 3:  # [N, H, W, C] -> [N, C, H, W]
                img = img.permute(0, 3, 1, 2)
            if img.max() > 1.0:
                img = img.float() / 255.0
            img = (img - self.imagenet_mean) / self.imagenet_std
            features.append(self.image_encoders[key](img))
        return torch.cat(features, dim=-1)

    def _get_low_dim_obs(self, obs: TensorDict) -> torch.Tensor:
        low_dim = []
        for obs_group in self.obs_groups["policy"]:
            for key in self.low_dim_keys:
                low_dim.append(obs[obs_group][key])
        return torch.cat(low_dim, dim=-1)

    def get_student_obs(self, obs: TensorDict) -> torch.Tensor:
        """Encode images and concatenate with normalized low-dim obs."""
        parts = []

        image_features = self._encode_images(obs)
        if image_features is not None:
            parts.append(image_features)

        if self.low_dim_keys:
            low_dim = self._get_low_dim_obs(obs)
            low_dim = self.student_obs_normalizer(low_dim)
            parts.append(low_dim)

        return torch.cat(parts, dim=-1)

    def get_student_distribution(self, obs: TensorDict) -> Normal:
        features = self.get_student_obs(obs)
        mean = self.student(features)
        if self.noise_std_type == "scalar":
            std = self.std.expand_as(mean)
        elif self.noise_std_type == "log":
            std = torch.exp(self.log_std).expand_as(mean)
        elif self.noise_std_type == "gsde":
            # For gSDE, compute state-dependent std
            latent_features = self.student[:-1](features)
            log_std = self.log_std
            std_weights = torch.exp(log_std)
            # variance = (phi(s)^2) @ (sigma^2)
            variance = torch.mm(latent_features**2, std_weights**2)
            std = torch.sqrt(variance + 1e-6)
        else:
            raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}")
        return Normal(mean, std)

    @torch.no_grad()
    def get_teacher_distribution(self, obs: TensorDict) -> Normal:
        teacher_obs = self.get_teacher_obs(obs)
        teacher_obs = self.teacher_obs_normalizer(teacher_obs)
        mean = self.teacher(teacher_obs)
        if self.noise_std_type == "scalar":
            std = self.teacher_std.expand_as(mean)
        elif self.noise_std_type == "log":
            std = torch.exp(self.teacher_log_std).expand_as(mean)
        elif self.noise_std_type == "gsde":
            latent_features = self.teacher[:-1](teacher_obs)
            teacher_log_std = self.teacher_log_std.to(latent_features.device)
            std_weights = torch.exp(teacher_log_std)
            variance = torch.mm(latent_features**2, std_weights**2)
            std = torch.sqrt(variance + 1e-6)
        else:
            raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}")
        return Normal(mean, std)

    def update_normalization(self, obs: TensorDict) -> None:
        if self.student_obs_normalization and self.num_low_dim > 0:
            low_dim = self._get_low_dim_obs(obs)
            self.student_obs_normalizer.update(low_dim)

    def load_state_dict(self, state_dict: dict, strict: bool = True) -> bool:
        if any("actor" in key for key in state_dict) and not any("student" in key for key in state_dict):
            if self.noise_std_type == "gsde" or self.noise_std_type == "log":
                self.teacher_log_std.data.copy_(state_dict["log_std"])
            else:
                self.teacher_std.data.copy_(state_dict["std"])
        return super().load_state_dict(state_dict, strict=strict)
