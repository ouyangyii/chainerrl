from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
from future import standard_library
standard_library.install_aliases()  # NOQA



import copy


import chainer
import chainer.links as L
import chainer.functions as F
from chainer import cuda

from chainerrl import links
from chainerrl.agents import dqn

# to begin from here
from pdb import set_trace


class SequenceCachedHiddenValue(links.Sequence):
    """Sequential callable Link that consists of other Links where the value of each link is cached after a call

    """
    def __init__(self, *layers, **kwargs):
        """
        Args:
            layer_indices: a list of layer indices whose values will be cached
            layers: layers of the network

            """
        self.layers_to_cache = kwargs.pop('layers_to_cach', [])
        self.layers_to_cache.sort()
        self.cached_values = []
        super().__init__(*layers)

    def to_gpu(self, device=None):
        # move cached values to gpu
        for value in self.cached_values:
            value.to_gpu(device)
        super().to_gpu(device)

    def __call__(self, x, **kwargs):
        h = x
        lay_count = 0
        for (index,layer), argnames, accept_var_args in zip(enumerate(self.layers),
                                                    self.argnames,
                                                    self.accept_var_args):
            if accept_var_args:
                layer_kwargs = kwargs
            else:
                layer_kwargs = {k: v for k, v in kwargs.items()
                                if k in argnames}
            h = layer(h, **layer_kwargs)
            while lay_count < len(self.layers_to_cache) and index == self.layers_to_cache[lay_count]:
                self.cached_values.append(copy.deepcopy(h))
                lay_count += 1

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
        self.n_step = kwargs.pop('n_step', 150)
        super().__init__(*args, **kwargs)
        self.Sigma = None
        self.noise = 0
        self.nu_history = []
        self.action_history = []
        self.hidden_layer_value_history = []

        self.average_loss_subnet = 0
        self.average_q_subnet = 0
        if self.gpu is not None and self.gpu >= 0:
            cuda.get_device(self.gpu).use()
            self.uncertainty_subnet.to_gpu(device=self.gpu)


    def compute_uncertainty_parms(self, a, features_vec):
        """
        compute params used in the update of the uncertainty subnet

        Args:
            (s,a): current state and action, s is omitted
            features_vec: phi(s), the last layer of the subnetwork
        """
        n_features = features_vec.shape[0]
        Sigma_a = self.Sigma[a, :, :]
        Sigma_a = Sigma_a.reshape(n_features, n_features)
        # compute the uncertainty value y, which is the target for the uncertainty subnetwork

        # compute the estimates of instantaneous uncertainty signal nu
        # these values will be used in the loss function of the uncertainty subnetwork
        nu_step1 = Sigma_a @ features_vec  # Sigma phi
        nu_current = float(features_vec.T @ nu_step1) # phi^T Sigma phi, a scalar
        self.nu_history.append(nu_current)
        # update the variances from observations
        Sigma_dif = (nu_step1 @ nu_step1.T) / (1 + nu_current)
        Sigma_a = Sigma_a - Sigma_dif
        self.Sigma[a, :, :] = Sigma_a


    def update_uncertainty_subnet(self, uncertainty_next=0.0):
        """
        update the uncertainty subnetwork

        Args:
            uncertainty_next: the computed uncertainty value for (s_next,a_next)
        """
        y_uncertainty = uncertainty_next

        while self.action_history:
            # compute the accumulated uncertainty signal with discounter factor gamma^2
            y_uncertainty = (self.gamma **2) * y_uncertainty + self.nu_history.pop()
            # the loss function of the subnet
            uncertainty_for_update = self.uncertainty_subnet(self.hidden_layer_value_history.pop()).q_values[:,self.action_history.pop()]
            loss_subnet = F.square(y_uncertainty - uncertainty_for_update)

            # Update stats
            self.average_loss_subnet *= self.average_loss_decay
            self.average_loss_subnet += (1 - self.average_loss_decay) * float(loss_subnet.data)

            # take a gradient step for the subnet
            self.uncertainty_subnet.cleargrads()
            loss_subnet.backward()
            self.optimizer_subnet.update()


    def act_and_train(self, obs, reward):
        with chainer.using_config('train', False), chainer.no_backprop_mode():
            action_value = self.model(
                self.batch_states([obs], self.xp, self.phi))
            q = float(action_value.max.data)

            # uncertainty_subnet takes input from the first hidden layer of the main Q-network
            hidden_layer_value = self.model.cached_values[0]
            uncertainty_estimates = self.uncertainty_subnet(hidden_layer_value)

            # add noise to Q-value to perform Thompson sampling for exploration
            # action_value_adjusted (array): the adjusted value
            assert action_value.n_actions == uncertainty_estimates.n_actions
            n_actions = action_value.n_actions
            # the uncertainty estimates should be positive, so add a lower threshold min_var
            min_var = self.xp.float32(0.001)
            uncertainty_estimates_values = self.xp.maximum(uncertainty_estimates.q_values.data , min_var)
            self.noise = self.xp.random.normal(size=n_actions).astype(self.xp.float32)
            bonus = self.beta * self.xp.multiply(self.noise,self.xp.sqrt(uncertainty_estimates_values))
            action_value_adjusted = action_value.q_values.data + bonus
            self.logger.debug('action_value.q_values.data:%s, action_value_adjusted:%s', action_value.q_values.data, action_value_adjusted)
            greedy_action = cuda.to_cpu(action_value_adjusted.argmax())

        # keep this if there is additional exploration
        action = self.explorer.select_action(
            self.t, lambda: greedy_action, action_value=action_value)


        # update date the uncertainty subnetwork every n_step steps
        if len(self.action_history) >= self.n_step:
            assert len(self.action_history) == len(self.nu_history)
            assert len(self.action_history) == len(self.hidden_layer_value_history)
            uncertainty_next = float(uncertainty_estimates_values[:, action])
            self.update_uncertainty_subnet(uncertainty_next)

        # the value of the last hidden layer of the Q function is the feature vector used in UBE
        features_vec = self.model.cached_values[1].data
        features_vec = features_vec.reshape([-1, 1])

        # initialization of the cov Sigma for all actions
        if self.Sigma is None:
            n_features = features_vec.shape[0]
            mu = 3 # scale for the initial cov matrix
            self.Sigma = self.xp.zeros((n_actions,n_features,n_features), dtype=self.xp.float32)
            for act_id in range(n_actions):
                self.Sigma[act_id,:,:] = mu*self.xp.eye(n_features)


        # compute and store parameters for the uncertainty subnetwork
        self.compute_uncertainty_parms(action, features_vec)
        self.action_history.append(action)
        self.hidden_layer_value_history.append(copy.deepcopy(hidden_layer_value))


        # Update stats
        self.average_q *= self.average_q_decay
        self.average_q += (1 - self.average_q_decay) * q

        self.average_q_subnet *= self.average_q_decay
        self.average_q_subnet += (1 - self.average_q_decay) * float(uncertainty_estimates_values[:,action])

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


        self.last_state = obs
        self.last_action = action

        self.replay_updater.update_if_necessary(self.t)

        self.logger.debug('t:%s r:%s a:%s', self.t, reward, action)

        return self.last_action

    def stop_episode_and_train(self, state, reward, done=False):
        """ Need to train the uncertainty subnetwork when an episode ends
        """
        # update the uncertainty subnetwork with no next state
        self.update_uncertainty_subnet()
        super().stop_episode_and_train(state, reward, done)

    # print also the stats of the subnet
    def get_statistics(self):
        return [
            ('average_q', self.average_q),
            ('average_loss', self.average_loss),
            ('average_q_subnet', self.average_q_subnet),
            ('average_loss_subnet', self.average_loss_subnet)
        ]