from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
from future import standard_library
standard_library.install_aliases()  # NOQA

import chainer
import chainer.links as L
import chainer.functions as F
import numpy as np

import chainerrl
from chainerrl.action_value import DiscreteActionValue
from chainerrl import agents
from chainerrl import experiments
from chainerrl import explorers
from chainerrl import links
from chainerrl import misc
from chainerrl.agents import dqn
from chainerrl import replay_buffer

# to begin from here
from pdb import set_trace

class UBE_DQN(dqn.DQN):
    """Uncertainty Bellman Equation (UBE) for DQN
    See: https://arxiv.org/abs/1709.05380

    Args:
        uncertainty_subnet (StateQFunction): the uncertainty sub-network, output is (u)^0.5.
        optimizer_subnet (Optimizer): Optimizer for the subnetwork that is already setup
        beta (float): Thompson sampling parameter
    For other arguments, see DQN.
    """

    def __init__(self, *args, **kwargs):
        self.uncertainty_subnet = kwargs.pop('uncertainty_subnet')
        self.optimizer_subnet = kwargs.pop('optimizer_subnet')
        self.beta = kwargs.pop('beta',0.01)
        self.Sigma = None
        self.n_features = None
        super().__init__(*args, **kwargs)

    # working on now
    def act_and_train(self, obs, reward):
        with chainer.using_config('train', False):
            with chainer.no_backprop_mode():
                action_value = self.model(
                    self.batch_states([obs], self.xp, self.phi))
                q = float(action_value.max.data)

                # uncertainty_subnet takes input from the hidden layer of the main Q-network
                hidden_layer_value = self.model.layers[0](
                        self.batch_states([obs], self.xp, self.phi))
                uncertainty_sqrt = self.uncertainty_subnet(hidden_layer_value)

                # add noise to Q-value to perform Thompson sampling
                # action_value_adjusted (array): the adjusted value
                assert action_value.n_actions == uncertainty_sqrt.n_actions
                n_actions = action_value.n_actions
                noise = np.random.normal(size=n_actions).astype(np.float32)
                action_value_adjusted = action_value.q_values.data + self.beta * np.multiply(noise,uncertainty_sqrt.q_values.data)
                greedy_action = action_value_adjusted.argmax(axis = 1)

        # initialization of the variances
        if self.Sigma is None:
            self.n_features = hidden_layer_value.shape[1]
            mu = 1 # scale for the initial cov matrix
            self.Sigma = np.zeros((n_actions,self.n_features,self.n_features), dtype=np.float32)
            for act_id in range(n_actions):
                self.Sigma[act_id,:,:] = mu*np.eye(self.n_features)


        # update param for the uncertainty subnet
        Sigma_current = self.Sigma[greedy_action, :, :]
        Sigma_current = Sigma_current.reshape(self.n_features, self.n_features)
        y_step1 = hidden_layer_value.data @ Sigma_current # phi^T Sigma
        y_step2 = y_step1  @ (hidden_layer_value.data.T)    # phi^T Sigma phi
        # the termination check is ignored for now
        y_uncertainty = y_step2 + (self.gamma* uncertainty_sqrt.q_values.data[0,greedy_action])**2

        # the loss function of the subnet
        uncertainty_sqrt_for_update = self.uncertainty_subnet(hidden_layer_value)
        loss_subnet = F.square((F.square(uncertainty_sqrt_for_update.q_values[:,greedy_action]) - y_uncertainty))
        # take a gradient step for the subnet
        self.uncertainty_subnet.cleargrads()
        loss_subnet.backward()
        self.optimizer_subnet.update()

        # update the variances from observations
        Sigma_dif = (y_step1.T @ y_step1) / (1+y_step2)
        Sigma_current = Sigma_current - Sigma_dif
        self.Sigma[greedy_action, :, :] = Sigma_current
        # if self.t % 5000 == 0:
        #     set_trace() # debugging
        #### the rest is the same as in DQN
        # Update stats
        self.average_q *= self.average_q_decay
        self.average_q += (1 - self.average_q_decay) * q

        self.logger.debug('t:%s q:%s action_value:%s', self.t, q, action_value)

        action = self.explorer.select_action(
            self.t, lambda: greedy_action, action_value=action_value)
        self.t += 1

        # Update the target network
        if self.t % self.target_update_interval == 0:
            self.sync_target_network()

        if self.last_state is not None:
            assert self.last_action is not None
            # Add a transition to the replay buffer
            self.replay_buffer.append(
                state=self.last_state,
                action=self.last_action,
                reward=reward,
                next_state=obs,
                next_action=action,
                is_state_terminal=False)

        self.last_state = obs
        self.last_action = action

        self.replay_updater.update_if_necessary(self.t)

        self.logger.debug('t:%s r:%s a:%s', self.t, reward, action)

        return self.last_action