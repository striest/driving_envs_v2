from math import log

from drivingenvs.vehicles.ackermann import AckermannSteeredVehicle
from drivingenvs.envs.base_driving_env import BaseDrivingEnv
from drivingenvs.envs.unsupervised_lane_change_env import UnsupervisedLaneChangeEnv

from yarp.envs.torchgymenv import TorchGymEnv
from yarp.envs.unsupervised_env import UnsupervisedEnv
from yarp.policies.tanhgaussianpolicy import TanhGaussianMLPPolicy
from yarp.networks.mlp import MLP
from yarp.networks.valuemlp import SingleHeadQMLP
from yarp.replaybuffers.unsupervisedreplaybuffer import UnsupervisedReplayBuffer
from yarp.algos.sac import SAC
from yarp.algos.diayn import DIAYN
from yarp.experiments.experiment import Experiment

from torch import nn

contexts = 10
max_steps = 100
max_rew = -max_steps * log(1/contexts)

print('contexts = {}, max_steps = {}, max_rew = {}'.format(contexts, max_steps, max_rew))

veh = AckermannSteeredVehicle((4, 2))
env = BaseDrivingEnv(veh, distance=200.0, n_lanes = 5, dt=0.2, max_steps = max_steps, start_lane = 2)
env = UnsupervisedEnv(env, context_dim=contexts)

print(env.reset())

policy = TanhGaussianMLPPolicy(env, hiddens = [300, 300], hidden_activation=nn.ReLU)
qf1 = SingleHeadQMLP(env, hiddens = [300, 300], hidden_activation=nn.ReLU, logscale=True, scale=1.0)
target_qf1 = SingleHeadQMLP(env, hiddens = [300, 300], hidden_activation=nn.ReLU, logscale=True, scale=1.0)
qf2 = SingleHeadQMLP(env, hiddens = [300, 300], hidden_activation=nn.ReLU, logscale=True, scale=1.0)
target_qf2 = SingleHeadQMLP(env, hiddens = [300, 300], hidden_activation=nn.ReLU, logscale=True, scale=1.0)
buf = UnsupervisedReplayBuffer(env)

sac = SAC(env, policy, qf1, target_qf1, qf2, target_qf2, buf, discount = 0.95,  reward_scale=1/max_rew, learn_alpha=True, alpha=0.05, steps_per_epoch=1000, qf_itrs=1000, qf_batch_size=256, target_update_tau=0.005, epochs=int(1e7))

disc = MLP(insize=env.wrapped_env.observation_space.shape[0], outsize=env.context_dim, hiddens = [300, 300])

diayn = DIAYN(env, buf, disc, sac)

experiment = Experiment(diayn, 'diayn', save_every=10, save_logs_every=1)

#import pdb; pdb.set_trace()
experiment.run()
