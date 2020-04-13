import torch
from torch import nn, distributions

from yarp.policies.base import Policy
from drivingenvs.envs.base_driving_env import BaseDrivingEnv

from drivingenvs.vehicles.ackermann import AckermannSteeredVehicle

class LaneFollowing(Policy):
	"""
	Prior policy for driving that guides the agent to the center of the lane (using a basic P controller)
	"""
	
	def __init__(self, env, lookahead=10.0):
		self.lookahead = lookahead
		self.env = env

	def forward(self, obs, std = 0.1, kp=6.0):
		"""
		Use the y value of the observation to center a gaussian.
		Assuming observation contains left lane dist at idx 4 and right at idx 3.
		Assumes action space is [velocity, turn rate]
		Implement the correction to turn rate as a pure pursuit controller.
		"""
		mean = torch.ones(obs.shape[0], 2).to(obs.device)
		s = torch.ones(obs.shape[0], 2).to(obs.device)
		s[:, 1] *= std #Let acceleration std be one.

		ll_dist = obs[:, 4]
		rl_dist = obs[:, 3]
		vel = obs[:, 2]
		heading = obs[:, 1]
		
		
		lane_pos = rl_dist + self.env.veh.shape[1]/2 - self.env.lane_width/2

		#project the dist to centerline along y-axis of vehicle.
		scaled_lookahead = self.lookahead * vel
		y = -lane_pos/torch.cos(heading)
		l = (scaled_lookahead**2 + lane_pos**2) ** 0.5

		mean[:, 1] *= 2*y/l

		centered_mask = (lane_pos.abs() < 0.2)
		mean[:, 1][centered_mask] = -kp * heading[centered_mask]

		mean[:, 0] *= 0

		dist = distributions.normal.Normal(loc = mean, scale = s)

		return dist

	def action(self, obs, deterministic=False):
		"""
		Given a single observation, return a single action.
		"""
		dist = self.forward(obs.unsqueeze(0))

		if deterministic:
			return dist.mean.squeeze()
		else:
			return dist.sample().squeeze()

	def actions(self, obs, deterministic=False):
		"""
		Given observations in batch, return an action for each observation.
		"""	
		dist = self.forward(obs)

		if deterministic:
			return dist.mean
		else:
			return dist.sample()

	def cuda(self):	
		"""
		Move policy to GPU.
		"""
		pass
	
	def cpu(self):	
		"""
		Move policy to CPU.
		"""
		pass

if __name__ == '__main__':
	veh = AckermannSteeredVehicle((4, 2))
	env = BaseDrivingEnv(veh)
	prior = LaneFollowing()
	obs = env.reset()
	print(obs)
	print(prior.forward(obs.unsqueeze(0)))
