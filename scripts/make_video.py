import pickle
import torch
import argparse
import os
import subprocess
import matplotlib.pyplot as plt

from drivingenvs.vehicles.ackermann import AckermannSteeredVehicle
from drivingenvs.envs.driving_env_with_vehicles import DrivingEnvWithVehicles

"""
Given a path to experiment output, make a video of the policy acting in the environment.
"""

parser = argparse.ArgumentParser(description='Parse videomaker params')


parser.add_argument('--video_name', type=str, required=True, help='name of the output video (actul video will have \'.mp4\' appended to it).')
parser.add_argument('--video_fp', type=str, required=False, default = '', help='location to output the video to (defaults to current dir).')
parser.add_argument('--experiment_fp', type=str, required=True, help='location where the experiment results are (the base dir).')
parser.add_argument('--env_fp', type=str, required=False, help='location of env file (defaults to the one stored in the experiment dir).')
parser.add_argument('--itr', type=int, required=False, help='the iteration of the policy to test. Defaults to the best one if no argument provided.')
parser.add_argument('--framerate', type=int, required=False, default=10, help='framerate of the output video')
parser.add_argument('--deterministic', type=bool, required=False, default=True, help='determines whether the policy takes deterministic actions')

args = parser.parse_args()

print(args)
env_fp = os.path.join(args.env_fp, 'env.cpt') if args.env_fp else os.path.join(args.experiment_fp, 'env.cpt')
policy_fp = os.path.join(args.experiment_fp, 'itr_{}/policy.cpt'.format(args.itr)) if args.itr else os.path.join(args.experiment_fp, '_best/policy.cpt')
output_fp = os.path.join(args.video_fp, '{}.mp4'.format(args.video_name))
tmp_fp = os.path.join(args.video_fp, 'tmp')

env = pickle.load(open(env_fp, 'rb'))
policy = torch.load(policy_fp)

print(env)
print(policy)

subprocess.call(['mkdir', tmp_fp])
o = env.reset(reset_kwargs = {'lane':2})

frame = 0
#import pdb;pdb.set_trace()
while True:
	print('Frame {}'.format(frame), end='\r')
	env.wrapped_env.render(window=50)
	plt.savefig(os.path.join(tmp_fp, 'frame{0:05d}.png'.format(frame)))
	plt.close()
	#o, r, t, i = env.step(policy.action(o, deterministic=args.deterministic))
	o, r, t, i = env.step(torch.tensor([1.0, 0.1]))
	frame += 1

	if t:
		break

os.chdir(tmp_fp)
os.chdir('../')
subprocess.call(['ffmpeg', '-framerate', str(args.framerate), '-i', 'tmp/frame%05d.png', '-pix_fmt', 'yuv420p', output_fp])
subprocess.call(['rm', '-r', 'tmp'])
