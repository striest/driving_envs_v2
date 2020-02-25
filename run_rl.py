from drivingenvs.vehicles.ackermann import AckermannSteeredVehicle
from drivingenvs.envs.base_driving_env import BaseDrivingEnv

from yarp.envs.torchgymenv import TorchGymEnv
from yarp.policies.tanhgaussianpolicy import TanhGaussianMLPPolicy
from yarp.networks.valuemlp import SingleHeadQMLP
from yarp.replaybuffers.simplereplaybuffer import SimpleReplayBuffer
from yarp.algos.sac import SAC
from yarp.experiments.experiment import Experiment

from torch import nn

veh = AckermannSteeredVehicle((4, 2))
env = BaseDrivingEnv(veh)
print(env.reset())
print(env.ego_state)

policy = TanhGaussianMLPPolicy(env, hiddens = [300, 300], hidden_activation=nn.ReLU)
qf1 = SingleHeadQMLP(env, hiddens = [300, 300], hidden_activation=nn.ReLU, logscale=True, scale=1.0)
target_qf1 = SingleHeadQMLP(env, hiddens = [300, 300], hidden_activation=nn.ReLU, logscale=True, scale=1.0)
qf2 = SingleHeadQMLP(env, hiddens = [300, 300], hidden_activation=nn.ReLU, logscale=True, scale=1.0)
target_qf2 = SingleHeadQMLP(env, hiddens = [300, 300], hidden_activation=nn.ReLU, logscale=True, scale=1.0)
buf = SimpleReplayBuffer(env)

algo = SAC(env, policy, qf1, target_qf1, qf2, target_qf2, buf, discount = 0.99,  reward_scale=0.01, learn_alpha=True, alpha=0.05, steps_per_epoch=1000, qf_itrs=1000, qf_batch_size=256, target_update_tau=0.005, epochs=int(1e7))

experiment = Experiment(algo, 'blah', save_every=100000, save_logs_every=10000000)
experiment.run()
