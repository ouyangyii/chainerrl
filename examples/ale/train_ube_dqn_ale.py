from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
from __future__ import absolute_import
from builtins import *  # NOQA
from future import standard_library
standard_library.install_aliases()  # NOQA
import argparse
import os

import gym
gym.undo_logger_setup()  # NOQA
import chainer
from chainer import functions as F
from chainer import links as L
from chainer import optimizers
import numpy as np

import chainerrl
from chainerrl.action_value import DiscreteActionValue
from chainerrl import agents
from chainerrl import experiments
from chainerrl import explorers
from chainerrl import links
from chainerrl import misc
from chainerrl.q_functions import DuelingDQN
from chainerrl import replay_buffer

import atari_wrappers


class SingleSharedBias(chainer.Chain):
    """Single shared bias used in the Double DQN paper.

    You can add this link after a Linear layer with nobias=True to implement a
    Linear layer with a single shared bias parameter.

    See http://arxiv.org/abs/1509.06461.
    """

    def __init__(self):
        super().__init__()
        with self.init_scope():
            self.bias = chainer.Parameter(0, shape=1)

    def __call__(self, x):
        return x + F.broadcast_to(self.bias, x.shape)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='BreakoutNoFrameskip-v4')
    parser.add_argument('--outdir', type=str, default='results',
                        help='Directory path to save output files.'
                             ' If it does not exist, it will be created.')
    parser.add_argument('--seed', type=int, default=0,
                        help='Random seed [0, 2 ** 31)')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--demo', action='store_true', default=False)
    parser.add_argument('--load', type=str, default=None)
    parser.add_argument('--final-exploration-frames',
                        type=int, default=10 ** 6)
    parser.add_argument('--final-epsilon', type=float, default=0.1)
    parser.add_argument('--eval-epsilon', type=float, default=0.05)
    parser.add_argument('--steps', type=int, default=5 * 10 ** 7)
    parser.add_argument('--max-episode-len', type=int,
                        default=30 * 60 * 60 // 4,  # 30 minutes with 60/4 fps
                        help='Maximum number of steps for each episode.')
    parser.add_argument('--replay-start-size', type=int, default=5 * 10 ** 4)
    parser.add_argument('--target-update-interval',
                        type=int, default=3 * 10 ** 4)
    parser.add_argument('--eval-interval', type=int, default=10 ** 5)
    parser.add_argument('--update-interval', type=int, default=4)
    parser.add_argument('--eval-n-runs', type=int, default=10)
    parser.add_argument('--no-clip-delta',
                        dest='clip_delta', action='store_false')
    parser.set_defaults(clip_delta=True)
    parser.add_argument('--logging-level', type=int, default=20,
                        help='Logging level. 10:DEBUG, 20:INFO etc.')
    parser.add_argument('--render', action='store_true', default=False,
                        help='Render env states in a GUI window.')
    parser.add_argument('--monitor', action='store_true', default=False,
                        help='Monitor env. Videos and additional information'
                             ' are saved as output files.')
    parser.add_argument('--lr', type=float, default=2.5e-4,
                        help='Learning rate')
    parser.add_argument('--lr-subnet', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--momentum', type=float, default=0.00,
                        help='Momentum')
    parser.add_argument('--momentum-subnet', type=float, default=0.00,
                        help='Momentum for subnet')
    parser.add_argument('--prioritized', action='store_true', default=False,
                        help='Use prioritized experience replay.')

    # added for UBE
    parser.add_argument('--extra-exploration', action='store_true')
    parser.add_argument('--beta', type=float, default=1e-2)
    parser.add_argument('--n-step', type=int, default=1)

    args = parser.parse_args()

    import logging
    logging.basicConfig(level=args.logging_level)

    # Set a random seed used in ChainerRL.
    misc.set_random_seed(args.seed, gpus=(args.gpu,))

    # Set different random seeds for train and test envs.
    train_seed = args.seed
    test_seed = 2 ** 31 - 1 - args.seed

    args.outdir = experiments.prepare_output_dir(args, args.outdir)
    print('Output files are saved in {}'.format(args.outdir))

    def make_env(test):
        # Use different random seeds for train and test envs
        env_seed = test_seed if test else train_seed
        env = atari_wrappers.wrap_deepmind(
            atari_wrappers.make_atari(args.env),
            episode_life=not test,
            clip_rewards=not test)
        env.seed(int(env_seed))
        if args.monitor:
            env = gym.wrappers.Monitor(
                env, args.outdir,
                mode='evaluation' if test else 'training')
        if args.render:
            misc.env_modifiers.make_rendered(env)
        return env

    env = make_env(test=False)
    eval_env = make_env(test=True)

    n_actions = env.action_space.n
    # change the structure of q_func, the first layer's output will be feed to the subnet
    # the first layer's output is also the output of the last hidden layer
    class NatureDQNHead_convpart(chainer.ChainList):
        """DQN's head (Nature version)"""

        def __init__(self, n_input_channels=4, n_output_channels=3136,
                     activation=F.relu, bias=0.1):
            self.n_input_channels = n_input_channels
            self.activation = activation
            self.n_output_channels = n_output_channels

            layers = [
                L.Convolution2D(n_input_channels, 32, 8, stride=4,
                                initial_bias=bias),
                L.Convolution2D(32, 64, 4, stride=2, initial_bias=bias),
                L.Convolution2D(64, 64, 3, stride=1, initial_bias=bias),
            ]
            super().__init__(*layers)

        def __call__(self, state):
            h = state
            for layer in self:
                h = self.activation(layer(h))
            return h


    q_func = chainerrl.agents.ube.SequenceCachedHiddenValue(
        NatureDQNHead_convpart(),
        L.Linear(3136, 512, initial_bias=0.1),
        F.relu,
        L.Linear(512, n_actions),
        DiscreteActionValue,
        layers_to_cach=[0, 2])


    # No explorer for UBE unless extra exploration is used
    explorer = explorers.Greedy()
    if args.extra_exploration is True:
        # Use epsilon-greedy for exploration
        explorer = explorers.LinearDecayEpsilonGreedy(
        1.0, args.final_epsilon,
        args.final_exploration_frames,
        lambda: np.random.randint(n_actions))


    # Draw the computational graph and save it in the output directory.
    chainerrl.misc.draw_computational_graph(
        [q_func(np.zeros((4, 84, 84), dtype=np.float32)[None])],
        os.path.join(args.outdir, 'model'))

    # Use the same hyper parameters as the Nature paper's
    opt = optimizers.RMSpropGraves(
        lr=args.lr, alpha=0.95, momentum=args.momentum, eps=1e-2)


    opt.setup(q_func)

    # Select a replay buffer to use
    if args.prioritized:
        # Anneal beta from beta0 to 1 throughout training
        betasteps = args.steps / args.update_interval
        rbuf = replay_buffer.PrioritizedReplayBuffer(
            10 ** 6, alpha=0.6, beta0=0.4, betasteps=betasteps)
    else:
        rbuf = replay_buffer.ReplayBuffer(10 ** 6)



    def phi(x):
        # Feature extractor
        return np.asarray(x, dtype=np.float32) / 255

    # Agent = parse_agent(args.agent)
    # testing UBE
    # define the uncertainty subnetwork with one hidden layer
    # the bias is initialized with a large positive value
    uncertainty_subnet = links.Sequence(
        L.Linear(3136, 512, initial_bias=0.1),
        F.relu,
        L.Linear(512, n_actions,initial_bias = 1.0*512),
        DiscreteActionValue)
    # the optimizer for the subnetwork
    optimizer_subnet = optimizers.RMSpropGraves(
        lr=args.lr_subnet, alpha=0.9, momentum=args.momentum_subnet, eps=1e-10)
    optimizer_subnet.setup(uncertainty_subnet)


    Agent = agents.UBE_DQN
    agent = Agent(q_func, opt, rbuf, gpu=args.gpu, gamma=0.99,
                  explorer=explorer, replay_start_size=args.replay_start_size,
                  target_update_interval=args.target_update_interval,
                  clip_delta=args.clip_delta,
                  update_interval=args.update_interval,
                  batch_accumulator='sum',
                  phi=phi ,
                  uncertainty_subnet = uncertainty_subnet,
                  optimizer_subnet = optimizer_subnet,
                  beta=args.beta,
                  n_step=args.n_step
                )

    if args.load:
        agent.load(args.load)

    if args.demo:
        eval_stats = experiments.eval_performance(
            env=eval_env,
            agent=agent,
            n_runs=args.eval_n_runs)
        print('n_runs: {} mean: {} median: {} stdev {}'.format(
            args.eval_n_runs, eval_stats['mean'], eval_stats['median'],
            eval_stats['stdev']))
    else:

        # use a different beta in Thompson sampling for evaluation
        eval_explorer = explorer
        experiments.train_agent_with_evaluation(
            agent=agent, env=env, steps=args.steps,
            eval_n_runs=args.eval_n_runs, eval_interval=args.eval_interval,
            outdir=args.outdir, eval_explorer=eval_explorer,
            save_best_so_far_agent=False,
            max_episode_len=args.max_episode_len,
            eval_env=eval_env,
        )


if __name__ == '__main__':
    main()
