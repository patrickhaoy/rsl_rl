#  Copyright 2021 ETH Zurich, NVIDIA CORPORATION
#  SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
import torch.nn as nn

from rsl_rl.modules.actor_critic_recurrent import ActorCritic, get_activation
from rsl_rl.utils import unpad_trajectories
from rsl_rl.modules.transformer import Transformer


class ActorCriticTransformer(ActorCritic):
    is_recurrent = True

    def __init__(
        self,
        num_actor_obs,
        num_critic_obs,
        num_actions,
        actor_hidden_dims=[256, 256, 256],
        critic_hidden_dims=[256, 256, 256],
        activation="elu",
        transformer_hidden_size=256,
        transformer_block_count=2,
        transformer_context_length=32,
        transformer_head_count=4,
        init_noise_std=1.0,
        **kwargs,
    ):
        if kwargs:
            print(
                "ActorCriticTransformer.__init__ got unexpected arguments, which will be ignored: " + str(kwargs.keys()),
            )

        super().__init__(
            num_actor_obs=transformer_hidden_size,
            num_critic_obs=transformer_hidden_size,
            num_actions=num_actions,
            actor_hidden_dims=actor_hidden_dims,
            critic_hidden_dims=critic_hidden_dims,
            activation=activation,
            init_noise_std=init_noise_std,
        )

        activation = get_activation(activation)

        self.memory_a = TransformerMemory(num_actor_obs, hidden_size=transformer_hidden_size, block_count=transformer_block_count, context_length=transformer_context_length, head_count=transformer_head_count)
        self.memory_c = TransformerMemory(num_critic_obs, hidden_size=transformer_hidden_size, block_count=transformer_block_count, context_length=transformer_context_length, head_count=transformer_head_count)

        print(f"Actor Transformer: {self.memory_a}")
        print(f"Critic Transformer: {self.memory_c}")

    def reset(self, dones=None):
        self.memory_a.reset(dones)
        self.memory_c.reset(dones)

    def act(self, observations, masks=None, hidden_states=None):
        input_a = self.memory_a(observations, masks, hidden_states)
        return super().act(input_a.squeeze(0))

    def act_inference(self, observations, masks=None, hidden_states=None):
        if masks is None:
            input_a = self.memory_a(observations)
        else:
            input_a = self.memory_a(observations, masks, hidden_states)
        return super().act_inference(input_a.squeeze(0))

    def evaluate(self, critic_observations, masks=None, hidden_states=None):
        input_c = self.memory_c(critic_observations, masks, hidden_states)
        return super().evaluate(input_c.squeeze(0))

    def get_hidden_states(self):
        return self.memory_a.hidden_states, self.memory_c.hidden_states


class TransformerMemory(torch.nn.Module):
    def __init__(self, input_size, hidden_size=256, block_count=2, context_length=32, head_count=4):
        super().__init__()
        # RNN
        self.rnn = Transformer(
            input_size, hidden_size, hidden_size, block_count, context_length, head_count
        )
        self.hidden_states = None

    def forward(self, input, masks=None, hidden_states=None):
        batch_mode = masks is not None
        if batch_mode:
            # batch mode (policy update): need saved hidden states
            if hidden_states is None:
                raise ValueError("Hidden states not passed to memory module during policy update")
            out, _ = self.rnn(input, hidden_states)
            out = unpad_trajectories(out, masks)
        else:
            # inference mode (collection): use hidden states of last step
            out, self.hidden_states = self.rnn(input.unsqueeze(0), self.hidden_states)
        return out

    def reset(self, dones=None):
        # When the RNN is an LSTM, self.hidden_states_a is a list with hidden_state and cell_state
        for hidden_state in self.hidden_states:
            hidden_state[..., dones.bool(), :] = 0.0
