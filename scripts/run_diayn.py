from math import log

from drivingenvs.vehicles.ackermann import AckermannSteeredVehicle
from drivingenvs.envs.driving_env_with_vehicles import DrivingEnvWithVehicles

from yarp.envs.torchgymenv import TorchGymEnv
from yarp.envs.unsupervised_env import UnsupervisedEnv
from yarp.policies.tanhgaussianpolicy import TanhGaussianMLPPolicy
from yarp.networks.mlp import MLP
from yarp.networks.valuemlp import SingleHeadQMLP
from yarp.networks.mlp_discriminator import MLPDiscriminator
from yarp.replaybuffers.unsupervisedreplaybuffer import UnsupervisedReplayBuffer
from yarp.algos.sac import SAC
from yarp.algos.diayn import DIAYN
from yarp.experiments.experiment import Experiment

from torch import nn

contexts = 10
max_steps = 50
max_rew = -max_steps * log(1/contexts)

print('contexts = {}, max_steps = {}, max_rew = {}'.format(contexts, max_steps, max_rew))

veh = AckermannSteeredVehicle((4, 2))
env = DrivingEnvWithVehicles(veh, distance=200.0, n_lanes = 5, dt=0.2, max_steps = max_steps, start_lane = 2)
env = UnsupervisedEnv(env, context_dim=contexts)

print(env.reset())

policy = TanhGaussianMLPPolicy(env, hiddens = [300, 300], hidden_activation=nn.ReLU)
qf1 = SingleHeadQMLP(env, hiddens = [300, 300], hidden_activation=nn.ReLU, logscale=True, scale=1.0)
target_qf1 = SingleHeadQMLP(env, hiddens = [300, 300], hidden_activation=nn.ReLU, logscale=True, scale=1.0)
qf2 = SingleHeadQMLP(env, hiddens = [300, 300], hidden_activation=nn.ReLU, logscale=True, scale=1.0)
target_qf2 = SingleHeadQMLP(env, hiddens = [300, 300], hidden_activation=nn.ReLU, logscale=True, scale=1.0)
buf = UnsupervisedReplayBuffer(env)

sac = SAC(env, policy, qf1, target_qf1, qf2, target_qf2, buf, discount = 0.99,  reward_scale=1/max_rew, learn_alpha=True, alpha=0.01, steps_per_epoch=1000, qf_itrs=1000, qf_batch_size=256, target_update_tau=0.005, epochs=int(1e7))

disc = MLPDiscriminator(in_idxs=[2, 7, 8, 9, 10, 11], outsize=env.context_dim, hiddens = [300, 300])

print(disc)

diayn = DIAYN(env, buf, disc, sac)

experiment = Experiment(diayn, 'diayn_learn_alpha_v_laneid_full_traffic', save_every=10, save_logs_every=1)

#import pdb; pdb.set_trace()
experiment.run()
