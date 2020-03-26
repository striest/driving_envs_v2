import torch
import numpy as np
from math import pi
import gym
import matplotlib.pyplot as plt

from drivingenvs.utils import torch_utils as ptu
from yarp.envs.base import Env

class BaseDrivingEnv(Env):
	"""
	Basic driving environment with just the ego vehicle and some number of lanes. Ego vehicle position is given as the location of the rear wheel axle. (For ackermann steering)
	Default vehicle width from https://ops.fhwa.dot.gov/freight/publications/size_regs_final_rpt/.
	Length from https://mechanicbase.com/cars/average-car-length/
	Bounds on acceleration and turn rate came from CMU work.
	"""
	def __init__(self, veh, distance=1000.0, dt=0.1, lane_width=3.7, n_lanes=3, acc_bounds = (-4, 4), steer_bounds = (-.2618, .2618), max_steps = 100, start_lane=None, gpu=False):
		self.gpu = gpu
		self.max_distance = distance;
		self.dt = dt
		self.veh = veh
		self.n_lanes = n_lanes
		self.lane_width = lane_width
		self.lane_loc = torch.linspace(0, n_lanes*lane_width, n_lanes+1)
		self.start_lane = start_lane

		#I've left these as tuples so I can use them as args for torch.clamp
		self.steer_bounds = steer_bounds
		self.acc_bounds = acc_bounds

		#Ego state is a tensor of [x_pos, y_pos, heading, vel]. Controls are acceleration and yaw rate.
		self.ego_state = torch.zeros(4).to(self.device)
		self.prev_state = self.ego_state.clone()
		#Gets the state of the vehicle relative to the lane that it's in (this just amounts to changing the y value for straight lanes).
		self.max_steps = max_steps
		self.curr_step = None

	def reset(self, init_speed=21.0, lane=None):
		"""
		Reset the ego vehicle to the center of a random lane. Initialize the vehicle with the given speed.
		"""
		self.curr_step = torch.tensor(0)
		if self.start_lane is None:
			if lane is None:
				y_pos = self.lane_loc[:-1][torch.randint(self.n_lanes, (1, ))] + self.lane_width/2
			else:
				y_pos = self.lane_loc[:-1][lane] + self.lane_width/2
		else:
				y_pos = self.lane_loc[:-1][self.start_lane] + self.lane_width/2

		x_pos = 0.0
		v = init_speed
		heading = 0.0

		self.ego_state = torch.tensor([x_pos, y_pos, heading, v]).to(self.device)
		self.prev_state = self.ego_state.clone()
		return self.observation

	def step(self, action):
		"""
		Step in all current envs. Again, note the vectorization of the env.
		"""
		self.prev_state = self.ego_state.clone()

		action[0] = action[0].clamp(*self.acc_bounds)
		action[1] = action[1].clamp(*self.steer_bounds)

		self.ego_state[3] += action[0]*self.dt
		turn_angle = action[1]*self.dt
		control = torch.tensor([self.ego_state[3], turn_angle]).to(self.device)

		d_state = self.veh.propagate_from_tensor(self.ego_state[:-1], control)
		d_state *= self.dt
		self.ego_state[:-1] += d_state

		self.curr_step += 1

		return self.observation, self.reward, self.terminal, self.info

	@property
	def action_space(self):
		"""
		Action space for this env is a continuous valued acceleration and yaw rate, bounded by the instance variables given in the problem.
		"""
		return gym.spaces.Box(low=np.array([self.acc_bounds[0], self.steer_bounds[0]]), high=np.array([self.acc_bounds[1], self.steer_bounds[1]]))

	@property
	def observation_space(self):
		"""
		Returns the dimension of the observation space
		"""
		obs = self.observation.shape
		return gym.spaces.Box(low = np.ones(obs) * -np.inf, high = np.ones(obs) * np.inf)

	def render(self, fig = None, ax = None, window = 30):
		"""
		for visualization. Go window m in either dir from the ego veh.
		"""
		if fig is None or ax is None:
			fig, ax = plt.subplots()

		ax.set_xlim(self.ego_state[0]-window, self.ego_state[0]+window)
		ax.set_ylim(self.ego_state[1]-window, self.ego_state[1]+window)

		for loc in self.lane_loc[[0, -1]]:
			ax.axhline(loc.item(), ls='-', c='k')
			
		for loc in self.lane_loc[1:-1]:
			ax.axhline(loc.item(), ls='--', c='k')

		vl = self.veh.shape[0]
		vw = self.veh.shape[1]
		vx = self.ego_state[0] - vl/2
		vy = self.ego_state[1] - vw/2
		ax.add_patch(plt.Rectangle([vx, vy], vl, vw, angle=self.ego_state[2] * (180/pi)))

		return fig, ax

	def render_traj(self, traj, fig = None, ax = None, traj_kwargs = {}, window=None):
		x = traj[:, 0]
		y = traj[:, 1]

		if fig is None or ax is None:
			fig, ax = plt.subplots()

		if window:
			ax.set_xlim(x[0]-window, x[0]+window)
			ax.set_ylim(y[1]-window, y[1]+window)

		for loc in self.lane_loc[[0, -1]]:
			ax.axhline(loc.item(), ls='-', c='k')
			
		for loc in self.lane_loc[1:-1]:
			ax.axhline(loc.item(), ls='--', c='k')

		vl = self.veh.shape[0]
		vw = self.veh.shape[1]
		#vx = self.ego_state[0] - vl/2
		#vy = self.ego_state[1] - vw/2
		ax.add_patch(plt.Rectangle([x[0], y[0]], vl, vw, angle=self.ego_state[2] * (180/pi)))
		ax.plot(x, y, **traj_kwargs)

		return fig, ax

	def cuda(self):
		"""
		Move env to GPU.
		"""
		pass
	
	def cpu(self):
		"""
		move env to CPU.
		"""
		pass

	@property
	def device(self):
		return 'cuda' if self.gpu else 'cpu'

	@property
	def lane_relative_state(self):
		"""
		Basically just find the smallest Y that's less than you and subtract it.
		"""
		lane_offsets = self.lane_loc - self.ego_state[1]
		valid_lane_offsets = lane_offsets[lane_offsets>=0]
		if len(valid_lane_offsets) > 0:
			lane_offset = valid_lane_offsets.min()
		else:
			lane_offset = -1

		out = self.ego_state.clone()
		out[1] = lane_offset
		return out

	@property
	def terminal(self):
		"""
		Consider ourselves in a terminal state if we drive off the side of the road or we're past the goal.
		"""
		side_term = self.ego_state[1] < self.lane_loc[0] or self.ego_state[1] > self.lane_loc[-1]
		end_term = self.ego_state[0] > self.max_distance
		time_term = self.curr_step >= self.max_steps
		return side_term or end_term or time_term

	@property
	def reward(self):
		"""
		Give the agent reward for forward progress
		"""
		return self.ego_state[0] - self.prev_state[0]

	@property
	def info(self):
		return {'state': self.state}

	@property
	def state(self):
		"""
		Unlike observation, contains the complete state of the vehicle (position and velocity)
		"""
		return self.ego_state.clone()

	
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

	@property
	def observation(self):
		"""
		For now, let observation be [y_pos, vel, heading, dist to left lane line, dist to right lane line]
		We can make the dist func more sophisticated to handle cars being shifted, but lets start simple.
		"""
		ego = self.ego_state.clone()[1:]
		lane_pos = self.lane_relative_state[1]
		veh_width = self.veh.shape[1]
		ll_dist = self.lane_width - (lane_pos + veh_width/2)
		rl_dist = lane_pos - veh_width/2

		far_left_dist = self.lane_loc[-1] - (self.ego_state[1] + veh_width/2)
		far_right_dist = self.ego_state[1] - veh_width/2

		lane_obs = torch.tensor([ll_dist, rl_dist, far_left_dist, far_right_dist]).to(self.device)
		current_lane = ptu.one_hot(self.current_lane, n_classes = self.n_lanes+1)

		return torch.cat([ego, lane_obs, current_lane], dim=0)
