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
from typing import Any, NoReturn

from rsl_rl.networks import MLP, EmpiricalNormalization, HiddenState


class StudentTeacherImage(nn.Module):
    """Student-Teacher module with image encoder for the student.
    
    The student uses a ResNet backbone to encode images and concatenates with
    proprioceptive state before feeding to an MLP. The teacher uses only
    state observations with an MLP.
    """
    is_recurrent: bool = False

    def __init__(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        num_actions: int,
        # Image encoder config
        image_keys: list[str] = None,
        resnet_type: str = "resnet18",
        # Student config
        student_obs_normalization: bool = False,
        student_hidden_dims: tuple[int] | list[int] = [256, 256, 256],
        # Teacher config
        teacher_obs_normalization: bool = False,
        teacher_hidden_dims: tuple[int] | list[int] = [512, 256, 128, 64],
        activation: str = "elu",
        init_noise_std: float = 0.1,
        noise_std_type: str = "scalar",
        **kwargs: dict[str, Any],
    ) -> None:
        if kwargs:
            print(
                "StudentTeacherImage.__init__ got unexpected arguments, which will be ignored: "
                + str([key for key in kwargs])
            )
        super().__init__()

        self.loaded_teacher = False
        self.obs_groups = obs_groups
        
        # Default image keys if not provided
        if image_keys is None:
            image_keys = ["front_rgb", "side_rgb", "wrist_rgb"]
        self.image_keys = image_keys
        
        # Identify proprioceptive keys (non-image student observations)
        self.proprio_keys = [k for k in obs_groups["policy"] if k not in image_keys]
        
        # Build image encoder (ResNet backbone)
        resnet_models = {
            "resnet18": torchvision.models.resnet18,
            "resnet34": torchvision.models.resnet34,
            "resnet50": torchvision.models.resnet50,
        }
        if resnet_type not in resnet_models:
            raise ValueError(f"Unknown resnet_type: {resnet_type}. Supported: {list(resnet_models.keys())}")
        
        # Train from scratch (no pretrained weights)
        backbone = resnet_models[resnet_type](weights=None)
        
        # Get feature dimension before removing fc layer
        self.image_feature_dim = backbone.fc.in_features
        backbone.fc = nn.Identity()
        self.image_encoder = backbone
        
        # Calculate student input dimension
        num_images = len([k for k in image_keys if k in obs_groups["policy"]])
        num_proprio_obs = 0
        for key in self.proprio_keys:
            if key in obs['policy']:
                num_proprio_obs += obs['policy'][key].shape[-1]
        
        num_student_obs = num_images * self.image_feature_dim + num_proprio_obs
        print(f"Student input dim: {num_images} images * {self.image_feature_dim} + {num_proprio_obs} proprio = {num_student_obs}")
        
        # Student MLP
        self.student = MLP(num_student_obs, num_actions, student_hidden_dims, activation)
        print(f"Student MLP: {self.student}")
        
        # Student proprioceptive normalization
        self.student_obs_normalization = student_obs_normalization
        if student_obs_normalization and num_proprio_obs > 0:
            self.student_obs_normalizer = EmpiricalNormalization(num_proprio_obs)
        else:
            self.student_obs_normalizer = nn.Identity()
        
        # Teacher network (state-based MLP)
        num_teacher_obs = 0
        for obs_group in obs_groups["teacher"]:
            if obs_group in obs:
                assert len(obs[obs_group].shape) == 2, f"Teacher obs must be 1D, got shape {obs[obs_group].shape}"
                num_teacher_obs += obs[obs_group].shape[-1]
        
        self.teacher = MLP(num_teacher_obs, num_actions, teacher_hidden_dims, activation)
        self.teacher.eval()
        print(f"Teacher MLP: {self.teacher}")
        
        # Teacher observation normalization
        self.teacher_obs_normalization = teacher_obs_normalization
        if teacher_obs_normalization:
            self.teacher_obs_normalizer = EmpiricalNormalization(num_teacher_obs)
        else:
            self.teacher_obs_normalizer = nn.Identity()
        
        # Action noise
        self.noise_std_type = noise_std_type
        if self.noise_std_type == "scalar":
            self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        elif self.noise_std_type == "log":
            self.log_std = nn.Parameter(torch.log(init_noise_std * torch.ones(num_actions)))
        else:
            raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}")
        
        self.distribution = None
        Normal.set_default_validate_args(False)

    def reset(
        self, dones: torch.Tensor | None = None, hidden_states: tuple[HiddenState, HiddenState] = (None, None)
    ) -> None:
        pass

    def forward(self) -> NoReturn:
        raise NotImplementedError

    @property
    def action_mean(self) -> torch.Tensor:
        return self.distribution.mean

    @property
    def action_std(self) -> torch.Tensor:
        return self.distribution.stddev

    @property
    def entropy(self) -> torch.Tensor:
        return self.distribution.entropy().sum(dim=-1)

    def _encode_images(self, obs: TensorDict) -> torch.Tensor:
        """Encode all images and concatenate features."""
        features = []
        for key in self.image_keys:
            if key in obs:
                img = obs[key]
                # Handle different input formats
                if img.dim() == 4:  # [N, H, W, C] or [N, C, H, W]
                    if img.shape[-1] == 3:  # [N, H, W, C]
                        img = img.permute(0, 3, 1, 2)
                # Normalize to [0, 1]
                if img.max() > 1.0:
                    img = img.float() / 255.0
                # Encode
                feat = self.image_encoder(img)
                features.append(feat)
        return torch.cat(features, dim=-1) if features else torch.empty(obs[self.proprio_keys[0]].shape[0], 0, device=obs[self.proprio_keys[0]].device)

    def _get_proprio(self, obs: TensorDict) -> torch.Tensor:
        """Get proprioceptive observations."""
        proprio_list = []
        for key in self.proprio_keys:
            if key in obs:
                proprio_list.append(obs[key])
        return torch.cat(proprio_list, dim=-1) if proprio_list else torch.empty(0)

    def _update_distribution(self, features: torch.Tensor) -> None:
        mean = self.student(features)
        if self.noise_std_type == "scalar":
            std = self.std.expand_as(mean)
        elif self.noise_std_type == "log":
            std = torch.exp(self.log_std).expand_as(mean)
        else:
            raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}")
        self.distribution = Normal(mean, std)

    def get_student_obs(self, obs: TensorDict) -> torch.Tensor:
        """Process student observations: encode images + normalize proprio."""
        obs = obs['policy']
        image_features = self._encode_images(obs)
        proprio = self._get_proprio(obs)
        if proprio.numel() > 0:
            proprio = self.student_obs_normalizer(proprio)
        if image_features.numel() > 0 and proprio.numel() > 0:
            return torch.cat([image_features, proprio], dim=-1)
        elif image_features.numel() > 0:
            return image_features
        else:
            return proprio

    def get_teacher_obs(self, obs: TensorDict) -> torch.Tensor:
        obs_list = [obs[obs_group] for obs_group in self.obs_groups["teacher"]]
        return torch.cat(obs_list, dim=-1)

    def act(self, obs: TensorDict) -> torch.Tensor:
        features = self.get_student_obs(obs)
        self._update_distribution(features)
        return self.distribution.sample()

    def act_inference(self, obs: TensorDict) -> torch.Tensor:
        features = self.get_student_obs(obs)
        return self.student(features)

    def evaluate(self, obs: TensorDict) -> torch.Tensor:
        teacher_obs = self.get_teacher_obs(obs)
        teacher_obs = self.teacher_obs_normalizer(teacher_obs)
        with torch.no_grad():
            return self.teacher(teacher_obs)

    def get_hidden_states(self) -> tuple[HiddenState, HiddenState]:
        return None, None

    def detach_hidden_states(self, dones: torch.Tensor | None = None) -> None:
        pass

    def train(self, mode: bool = True) -> None:
        super().train(mode)
        self.teacher.eval()
        self.teacher_obs_normalizer.eval()

    def update_normalization(self, obs: TensorDict) -> None:
        if self.student_obs_normalization:
            proprio = self._get_proprio(obs)
            if proprio.numel() > 0:
                self.student_obs_normalizer.update(proprio)

    def load_state_dict(self, state_dict: dict, strict: bool = True) -> bool:
        """Load the parameters of the student and teacher networks."""
        # Check if state_dict contains teacher and student or just teacher parameters
        if any("actor" in key for key in state_dict):  # Load parameters from rl training
            # Rename keys to match teacher and remove critic parameters
            teacher_state_dict = {}
            teacher_obs_normalizer_state_dict = {}
            for key, value in state_dict.items():
                if "actor." in key:
                    teacher_state_dict[key.replace("actor.", "")] = value
                if "actor_obs_normalizer." in key:
                    teacher_obs_normalizer_state_dict[key.replace("actor_obs_normalizer.", "")] = value
            self.teacher.load_state_dict(teacher_state_dict, strict=strict)
            if teacher_obs_normalizer_state_dict:
                self.teacher_obs_normalizer.load_state_dict(teacher_obs_normalizer_state_dict, strict=strict)
            self.loaded_teacher = True
            self.teacher.eval()
            self.teacher_obs_normalizer.eval()
            return False  # Training does not resume
        elif any("student" in key for key in state_dict):  # Load parameters from distillation training
            super().load_state_dict(state_dict, strict=strict)
            self.loaded_teacher = True
            self.teacher.eval()
            self.teacher_obs_normalizer.eval()
            return True  # Training resumes
        else:
            raise ValueError("state_dict does not contain student or teacher parameters")

