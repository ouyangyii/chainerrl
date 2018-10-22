from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
from future import standard_library
standard_library.install_aliases()  # NOQA

import chainer
import chainer.links as L
import chainer.functions as F
from chainer import cuda

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


class SequenceCachedHiddenValue(links.Sequence):
    """Sequential callable Link that consists of other Links where the value of each link is cached after a call

    """
    def __init__(self, layer_index, *layers):
        super().__init__(*layers)
        # self.layer_cached_values = [None] * len(self.layers)
        self.layer_index = layer_index
        self.layer_cached_value = None

    def __call__(self, x, **kwargs):
        h = x
        for (index,layer), argnames, accept_var_args in zip(enumerate(self.layers),
                                                    self.argnames,
                                                    self.accept_var_args):
            if accept_var_args:
                layer_kwargs = kwargs
            else:
                layer_kwargs = {k: v for k, v in kwargs.items()
                                if k in argnames}
            h = layer(h, **layer_kwargs)
            if index == self.layer_index:
                self.layer_cached_value = h
        return h




class UBE_DQN(dqn.DQN):
    """Uncertainty Bellman Equation (UBE) for DQN
    See: https://arxiv.org/abs/1709.05380

    Args:
        q_function (SequenceCachedValues, StateQFunction): the q-function with access to hidden layers
        uncertainty_subnet (SequenceCachedValues, StateQFunction): the uncertainty sub-network, output is (u)^0.5.
        optimizer_subnet (Optimizer): Optimizer for the subnetwork that is already setup
        beta (float): Thompson sampling parameter

        For other arguments, see DQN.
    """

    def __init__(self, *args, **kwargs):
        self.uncertainty_subnet = kwargs.pop('uncertainty_subnet')
        self.optimizer_subnet = kwargs.pop('optimizer_subnet')
        self.beta = kwargs.pop('beta',0.01)
        #self.n_step = kwargs.pop('n_step', 150)
        super().__init__(*args, **kwargs)
        self.Sigma = None
        self.last_features_vec = None
        self.last_hidden_layer_value = None
        self.bonus = 0
        self.average_loss_subnet = 0
        self.average_q_subnet = 0
        if self.gpu is not None and self.gpu >= 0:
            cuda.get_device(self.gpu).use()
            self.uncertainty_subnet.to_gpu(device=self.gpu)

    def update_uncertainty_subnet(self, features_vec,hidden_layer_value, a, s_next=None, a_next=None, uncertainty_next=None):
        """
        update params for the uncertainty subnet, and also the cov Sigma

        Args:
            (s,a): current state and action
            (s_next,a_next): next state and action
            features_vec: phi(s), the last layer of the subnetwork
            hidden_layer_value: the input to the uncertainty subnetwork
            uncertainty_next: the computed uncertainty value for (s_next,a_next)
        """
        n_features = features_vec.shape[0]
        Sigma_a = self.Sigma[a, :, :]
        Sigma_a = Sigma_a.reshape(n_features, n_features)
        # compute the uncertainty value y, which is the target for the uncertainty subnetwork
        y_step1 = Sigma_a @ features_vec  # Sigma phi
        y_step2 = float(features_vec.T @ y_step1) # phi^T Sigma phi, a scalar
        y_uncertainty = y_step2
        if s_next is not None:
            assert a_next is not None
            y_uncertainty += (self.gamma ** 2) * uncertainty_next

        # the loss function of the subnet
        log_uncertainty_for_update = self.uncertainty_subnet(hidden_layer_value).q_values[:,a]
        loss_subnet = F.square(y_uncertainty - F.exp(log_uncertainty_for_update))

        # Update stats
        self.average_loss_subnet *= self.average_loss_decay
        self.average_loss_subnet += (1 - self.average_loss_decay) * float(loss_subnet.data)
            #debug:
        # print([y_uncertainty,loss_subnet.data])
        # if loss_subnet.data > 1000:
            # set_trace()
        # take a gradient step for the subnet
        self.uncertainty_subnet.cleargrads()
        loss_subnet.backward()
        self.optimizer_subnet.update()

        # update the variances from observations
        Sigma_dif = (y_step1 @ y_step1.T) / (1 + y_step2)
        Sigma_a = Sigma_a - Sigma_dif
        self.Sigma[a, :, :] = Sigma_a

    def act_and_train(self, obs, reward):
        with chainer.using_config('train', False), chainer.no_backprop_mode():
            action_value = self.model(
                self.batch_states([obs], self.xp, self.phi))
            q = float(action_value.max.data)

            # uncertainty_subnet takes input from the first hidden layer of the main Q-network
            hidden_layer_value = self.model.layer_cached_value
            log_uncertainty = self.uncertainty_subnet(hidden_layer_value)

            # add noise to Q-value to perform Thompson sampling for exploration
            # action_value_adjusted (array): the adjusted value
            assert action_value.n_actions == log_uncertainty.n_actions
            n_actions = action_value.n_actions
            # the uncertainty estimates
            uncertainty_estimates = self.xp.exp(log_uncertainty.q_values.data)
            noise = self.xp.random.normal(size=n_actions).astype(self.xp.float32)
            # if self.last_state is None:
            self.bonus = self.beta * self.xp.multiply(noise,self.xp.sqrt(uncertainty_estimates))
            action_value_adjusted = action_value.q_values.data + self.bonus
            self.logger.debug('action_value.q_values.data:%s, action_value_adjusted:%s', action_value.q_values.data, action_value_adjusted)
            greedy_action = cuda.to_cpu(action_value_adjusted.argmax(axis = 1).astype(self.xp.int32))[0]

        # keep this if there is additional exploration
        action = self.explorer.select_action(
            self.t, lambda: greedy_action, action_value=action_value)



        # the value of the last hidden layer is the feature vector used in UBE
        features_vec = self.uncertainty_subnet.layer_cached_value.data
        features_vec = features_vec.reshape([-1, 1])

        # initialization of the cov Sigma for all actions
        if self.Sigma is None:
            n_features = features_vec.shape[0]
            mu = 1 # scale for the initial cov matrix
            self.Sigma = self.xp.zeros((n_actions,n_features,n_features), dtype=self.xp.float32)
            for act_id in range(n_actions):
                self.Sigma[act_id,:,:] = mu*self.xp.eye(n_features)


        # Update stats
        self.average_q *= self.average_q_decay
        self.average_q += (1 - self.average_q_decay) * q

        self.average_q_subnet *= self.average_q_decay
        self.average_q_subnet += (1 - self.average_q_decay) * float(uncertainty_estimates[:,action])

        self.logger.debug('t:%s q:%s action_value:%s', self.t, q, action_value)


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
            # update the uncertainty subnetwork
            self.update_uncertainty_subnet(
                self.last_features_vec,
                self.last_hidden_layer_value,
                self.last_action,
                s_next=obs,
                a_next=action,
                uncertainty_next=float(uncertainty_estimates[:,action]))


        self.last_state = obs
        self.last_action = action
        self.last_features_vec = features_vec
        self.last_hidden_layer_value = hidden_layer_value

        self.replay_updater.update_if_necessary(self.t)

        self.logger.debug('t:%s r:%s a:%s', self.t, reward, action)

        return self.last_action

    def stop_episode_and_train(self, state, reward, done=False):
        """ Need to train the uncertainty subnetwork when an episode ends
        """
        # update the uncertainty subnetwork with no next state
        self.update_uncertainty_subnet(
            self.last_features_vec,
            self.last_hidden_layer_value,
            self.last_action)

        super().stop_episode_and_train(state, reward, done)

    # print also stats
    def get_statistics(self):
        return [
            ('average_q', self.average_q),
            ('average_loss', self.average_loss),
            ('average_q_subnet', self.average_q_subnet),
            ('average_loss_subnet', self.average_loss_subnet)
        ]