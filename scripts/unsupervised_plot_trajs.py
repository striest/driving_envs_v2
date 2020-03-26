import pickle
import torch
import argparse
import os
import subprocess
import matplotlib.pyplot as plt

from yarp.collectors.unsupervised_pathcollector import UnsupervisedSARTSCollector
from yarp.utils.os_utils import str2bool

"""
Given a path to experiment output, make a video of the policy acting in the environment.
"""

parser = argparse.ArgumentParser(description='Parse videomaker params')


parser.add_argument('--save_to', type=str, required=False, help='location to save the return curve to. Will just show it if nothing provided.')
parser.add_argument('--experiment_fp', type=str, required=True, help='location where the experiment results are (the base dir).')
parser.add_argument('--env_fp', type=str, required=False, help='location of env file (defaults to the one stored in the experiment dir).')
parser.add_argument('--itr', type=int, required=False, help='the iteration of the policy to test. Defaults to the best one if no argument provided.')
parser.add_argument('--deterministic', type=str2bool, required=False, default=True, help='determines whether the policy takes deterministic actions')
parser.add_argument('--legend', type=str2bool, required=False, default=True, help='bool whether to display a legend on the plot')
parser.add_argument('--lane', type=int, required=False, default=True, help='what lane to start from')

args = parser.parse_args()

print(args)
env_fp = os.path.join(args.env_fp, 'env.cpt') if args.env_fp else os.path.join(args.experiment_fp, 'env.cpt')
policy_fp = os.path.join(args.experiment_fp, 'itr_{}/policy.cpt'.format(args.itr)) if args.itr else os.path.join(args.experiment_fp, '_best/policy.cpt')

env = pickle.load(open(env_fp, 'rb'))
policy = torch.load(policy_fp)

fig, ax = plt.subplots()
collector =  UnsupervisedSARTSCollector(1.0, env)

for c in range(env.context_dim):
	traj = collector.collecttrajs(env=env, policy=policy, ntrajs=1, deterministic=False, context=c, reset_kwargs={'lane': args.lane})
	states = torch.stack([t['state'] for t in traj['info']])
	fig, ax = env.render_traj(states, fig=fig, ax=ax, traj_kwargs={'marker':'.', 'label':'context ={}'.format(c)})

if args.legend:
	plt.legend()

if args.save_to is None:
	plt.show()
else:
	plt.savefig(args.save_to)
