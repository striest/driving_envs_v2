import torch
import matplotlib.pyplot as plt

from drivingenvs.envs.base_driving_env import BaseDrivingEnv
from drivingenvs.utils import torch_utils as ptu

class BaseLaneChangeEnv(BaseDrivingEnv):
	"""
	Basic environment for lane change with no obstacles
	"""
	def __init__(self, veh, distance=1000.0, dt=0.1, lane_width=3.7, n_lanes=3, acc_bounds = (-4, 4), steer_bounds = (-.2618, .2618), max_steps = 100, gpu=False):
		super(BaseLaneChangeEnv, self).__init__(veh, distance, dt, lane_width, n_lanes, acc_bounds, steer_bounds, max_steps, gpu=False)
		self.target_lane = torch.tensor(-1).to(self.device)

	def reset(self, target_lane=None):
		if not target_lane:
			self.target_lane=torch.randint(self.n_lanes, (1, )).squeeze()
		else:
			target_lane = target_lane

		super().reset()
		return self.observation

	@property
	def reward(self):
		#Reward the agent for forward progess only if its in the correct lane.
		correct_lane = (self.target_lane == self.current_lane).float()
		obs = self.observation
		#checking for positive displacements to lane lines i.e. is the vehicle in the lane.
		in_lane = (obs[3] > 0 and obs[4] > 0).float()
		base_rew = super().reward
		if base_rew < 0:
			return base_rew
		else:
			return correct_lane * in_lane * base_rew

	@property
	def observation(self):
		"""
		Append current lane and target lane to the base observation.
		"""
		current_lane = ptu.one_hot(self.current_lane, n_classes = self.n_lanes+1)
		target_lane = ptu.one_hot(self.target_lane, n_classes = self.n_lanes+1)
		base_obs = super().observation
		new_obs = torch.cat([base_obs, current_lane, target_lane], dim=0)
		return new_obs

	@property
	def current_lane(self):
		"""
		Computing current lane based on Y of the vehicle's centrpoint. If vehicle is out-of-lane, return 1+max_lane_id (going non-negative for one-hot tensor).
		"""
		ego_y = self.ego_state[1]
		if ego_y > self.n_lanes*self.lane_width or ego_y < 0:
			return torch.tensor(self.n_lanes).to(self.device)
		else:
			return (ego_y // self.lane_width).long()

	def render(self):
		fig, ax = super().render()
		ax.axhline(self.target_lane.float() * self.lane_width + self.lane_width/2, c='r', label='target lane = {}'.format(self.target_lane.item()), ls=':')
		plt.legend()
		return fig, ax
