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
        assert obs_groups == {"policy": ["policy"], "teacher": ["teacher"]}, "obs_groups is not supported for StudentTeacherVision"
        self.obs_groups = obs_groups

        # Must call nn.Module.__init__ first before assigning any nn.Module
        nn.Module.__init__(self)

        # Separate image keys (4D) from low-dim keys
        self.image_keys = []
        self.low_dim_keys = []
        num_low_dim = 0
        for key in obs["policy"].keys():
            if obs["policy"][key].dim() == 4:
                self.image_keys.append(key)
            elif obs["policy"][key].dim() == 2:
                self.low_dim_keys.append(key)
                num_low_dim += obs["policy"][key].shape[-1]
            else:
                raise ValueError(f"Unknown observation dimension: {obs['policy'][key].dim()}")
        self.image_keys = sorted(self.image_keys)
        self.low_dim_keys = sorted(self.low_dim_keys)

        # Build image encoders
        self.image_encoders = nn.ModuleDict()
        self.image_feature_dim = 0

        if self.image_keys:
            for key in self.image_keys:
                backbone = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
                self.image_feature_dim = backbone.fc.in_features
                backbone.fc = nn.Identity()
                self.image_encoders[key] = backbone
            print(f"Detected image observations: {self.image_keys}")

        # Calculate student input dim
        num_image_features = len(self.image_keys) * self.image_feature_dim
        num_student_obs = num_image_features + num_low_dim
        print(f"Student input: {len(self.image_keys)} images * {self.image_feature_dim} + {num_low_dim} low_dim = {num_student_obs}")

        self.loaded_teacher = False

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
        num_teacher_obs = obs["teacher"].shape[-1]
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
        elif noise_std_type == "log":
            self.log_std = nn.Parameter(torch.log(init_noise_std * torch.ones(num_actions)))
        else:
            raise ValueError(f"Unknown noise_std_type: {noise_std_type}. Should be 'scalar', 'log', or 'gsde'")

        # Auxiliary prediction head (predicts auxiliary observations from features)
        if "auxiliary" in obs:
            num_auxiliary_obs = obs["auxiliary"].shape[-1]
            self.auxiliary_head = MLP(num_student_obs, num_auxiliary_obs, student_hidden_dims[:2], activation)
            print(f"Auxiliary head MLP: {self.auxiliary_head}")
        else:
            self.auxiliary_head = None

    def _encode_images(self, obs: TensorDict) -> torch.Tensor | None:
        """Encode images through ResNet backbones."""
        if not self.image_keys:
            return None
        features = []
        for key in self.image_keys:
            img = obs["policy"][key]
            if img.shape[-1] == 3:  # [N, H, W, C] -> [N, C, H, W]
                img = img.permute(0, 3, 1, 2)
            if img.max() > 1.0:
                img = img.float() / 255.0
            img = (img - self.imagenet_mean) / self.imagenet_std
            features.append(self.image_encoders[key](img))
        return torch.cat(features, dim=-1)

    def _get_low_dim_obs(self, obs: TensorDict) -> torch.Tensor:
        low_dim = []
        for key in self.low_dim_keys:
            low_dim.append(obs["policy"][key])
        return torch.cat(low_dim, dim=-1)

    def act(self, obs: TensorDict) -> torch.Tensor:
        obs = self.get_student_obs(obs)
        self._update_distribution(obs)
        return self.distribution.sample()

    def act_inference(self, obs: TensorDict) -> torch.Tensor:
        obs = self.get_student_obs(obs)
        return self.student(obs)

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
        """Return student distribution. get_student_obs already normalizes low-dim."""
        student_obs = self.get_student_obs(obs)
        mean = self.student(student_obs)
        if self.noise_std_type == "scalar":
            std = self.std.expand_as(mean)
        elif self.noise_std_type == "log":
            std = torch.exp(self.log_std).expand_as(mean)
        else:
            raise ValueError(f"Unknown noise_std_type: {self.noise_std_type}")
        return Normal(mean, std)

    def update_normalization(self, obs: TensorDict) -> None:
        if self.student_obs_normalization and self.num_low_dim > 0:
            low_dim = self._get_low_dim_obs(obs)
            self.student_obs_normalizer.update(low_dim)

    def predict_auxiliary(self, obs: TensorDict) -> torch.Tensor:
        obs = self.get_student_obs(obs)
        return self.auxiliary_head(obs)
