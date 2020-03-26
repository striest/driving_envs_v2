import torch
import matplotlib.pyplot as plt

from drivingenvs.envs.base_lane_change_env import BaseLaneChangeEnv
from drivingenvs.utils import torch_utils as ptu

class UnsupervisedLaneChangeEnv(BaseLaneChangeEnv):
	"""
	Lane change env with no target lane.
	"""
	def __init__(self, veh, distance=1000.0, dt=0.1, lane_width=3.7, n_lanes=3, acc_bounds = (-4, 4), steer_bounds = (-.2618, .2618), max_steps = 100, gpu=False):
		super(UnsupervisedLaneChangeEnv, self).__init__(veh, distance, dt, lane_width, n_lanes, acc_bounds, steer_bounds, max_steps, gpu=False)

	@property
	def observation(self):
		"""
		Append current lane and target lane to the base observation.
		"""
		current_lane = ptu.one_hot(self.current_lane, n_classes = self.n_lanes+1)
		base_obs = super().observation
		new_obs = torch.cat([base_obs, current_lane], dim=0)
		return new_obs
