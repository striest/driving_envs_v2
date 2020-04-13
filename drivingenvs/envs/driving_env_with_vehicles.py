import torch
import numpy as np
from math import pi
import gym
import matplotlib.pyplot as plt

from drivingenvs.utils import torch_utils as ptu
from drivingenvs.envs.base_driving_env import BaseDrivingEnv

class DrivingEnvWithVehicles(BaseDrivingEnv):
	"""
	Basic driving environment with just the ego vehicle and some number of lanes. Ego vehicle position is given as the location of the rear wheel axle. (For ackermann steering)
	Default vehicle width from https://ops.fhwa.dot.gov/freight/publications/size_regs_final_rpt/.
	Length from https://mechanicbase.com/cars/average-car-length/
	Bounds on acceleration and turn rate came from CMU work.
	"""
	def __init__(self, veh, distance=1000.0, dt=0.1, lane_width=3.7, n_lanes=3, acc_bounds = (-4, 4), steer_bounds = (-.2618, .2618), max_steps = 100, start_lane=None, speed_limit=21.0, obs_range=50.0, gpu=False):
		super().__init__(veh, distance, dt, lane_width, n_lanes, acc_bounds, steer_bounds, max_steps, start_lane, speed_limit, gpu)

		#We now need to track the states of a bunch of vehicles. (Since the motion model is stored in self.veh, only track states.)
		self.vehicles = torch.ones(1, 4) * -1
		self.obs_range = obs_range
		self.query_placeholder = torch.tensor([obs_range, obs_range, 0.0, self.speed_limit]).to(self.device)

	def reset(self, init_speed = 21.0, lane=None, init_kwargs = {}):
		"""
		"""
		super().reset(init_speed, lane)
		self.vehicles = self.initialize_scene(**init_kwargs)
		#Controls are given as acceleration, turn rate
		self.controls = torch.zeros(2, 2).to(self.device)
		return self.observation

	def step(self, action):
		o, r, t, i = super().step(action)
		d_vehicles = self.veh.propagate_from_tensor(self.vehicles[:, :-1], self.constant_speed_and_heading(self.vehicles))
		d_vehicles *= self.dt
		self.vehicles[:, :-1] += d_vehicles
		return self.observation, r, self.terminal, i

	def constant_speed_and_heading(self, vehs):
		"""
		Generates velocity and angular velocity commands for driving straight at a constant velocity. This should probably be in its own folder (with IDM and stuff).
		"""
		vels = vehs[:, -1]
		w = torch.zeros(vels.shape).to(vels.device)
		return torch.stack([vels, w], dim=1)

	@property
	def observation(self):
		o = super().observation

		n = self.neighbors
		n[~(n == self.query_placeholder).all(dim=1), 0] -= self.ego_state[0]
		n[~(n == self.query_placeholder).all(dim=1), 1] -= self.ego_state[1]
		n[~(n == self.query_placeholder).all(dim=1), 3] -= self.ego_state[3]
		veh_info = n.flatten()

		return torch.cat([o, veh_info], dim=0)

	@property
	def terminal(self):
		t = super().terminal
		#Also need to collision-check now. Do it roughly and assume that vehicles stay close to 0 degrees heading
		term_coll = self.is_colliding
		return t or term_coll

	@property
	def is_colliding(self):
		lon = (self.vehicles[:, 0] - self.ego_state[0]).abs()
		lat = (self.vehicles[:, 1] - self.ego_state[1]).abs()
		lon_coll = lon < self.veh.shape[0] * 1.1
		lat_coll = lat < self.veh.shape[1] * 1.1
		return (lat_coll & lon_coll).sum() > 0

	def render(self, fig = None, ax = None, window = 30):
		fig, ax = super().render(fig, ax, window)
		n = self.neighbors
		for i in range(self.vehicles.shape[0]):
			vl = self.veh.shape[0]
			vw = self.veh.shape[1]
			vx = self.vehicles[i, 0] - vl/2
			vy = self.vehicles[i, 1] - vw/2

			if (self.vehicles[i] == n).all(dim=1).any():
				ax.add_patch(plt.Rectangle([vx, vy], vl, vw, angle=self.vehicles[i, 2] * (180/pi), color='r'))
			else:
				ax.add_patch(plt.Rectangle([vx, vy], vl, vw, angle=self.vehicles[i, 2] * (180/pi), color='k'))

		return fig, ax
		
	@property
	def scene(self):
		"""
		ALL the vehicles on the highway (including ego-vehicle)
		"""
		return torch.cat([self.ego_state.unsqueeze(0), self.vehicles], dim=0)

	@property
	def neighbors(self):
		return torch.stack([f(self.ego_state, window = self.obs_range) for f in (self.fore, self.rear, self.left_fore, self.left_rear, self.right_fore, self.right_rear)], dim=0)

	def fore(self, vehs, window = 100.0):
		"""
		Get the state of the vehicle immediately in front of each vehicle in vehs
		vehs = [n_vehs x state], or [state]
		"""
		if len(vehs.shape) == 1:
			return self.fore(vehs.unsqueeze(0), window).squeeze()
		else:
			scene = self.scene
			lanes = self.get_lane(scene)
			#Unfortunately, I don't know a good way to do all this in parallel, as number of vehs to query depends on lane.
			fores = []
			for veh in vehs:
				ego_lane = self.get_lane(veh)
				targets = scene[ego_lane == lanes]
				dx = targets[:, 0] - veh[0]
				dx[(dx <= 0) | (dx > window)] = float('inf')
				
				if dx.nelement() == 0 or dx.min() == float('inf'):
					fores.append(self.query_placeholder)
				else:
					idx = dx.argmin()
					fores.append(targets[idx])

			return torch.stack(fores, dim=0)
					
	def rear(self, vehs, window = 100.0):
		"""
		Get the state of the vehicle immediately in front of each vehicle in vehs
		vehs = [n_vehs x state], or [state]
		"""
		if len(vehs.shape) == 1:
			return self.rear(vehs.unsqueeze(0), window).squeeze()
		else:
			scene = self.scene
			lanes = self.get_lane(scene)
			#Unfortunately, I don't know a good way to do all this in parallel, as number of vehs to query depends on lane.
			fores = []
			for veh in vehs:
				ego_lane = self.get_lane(veh)
				targets = scene[ego_lane == lanes]
				dx = targets[:, 0] - veh[0]
				dx[(dx >= 0) | (dx <- window)] = -float('inf')
				
				if dx.nelement() == 0 or dx.max() == -float('inf'):
					fores.append(self.query_placeholder)
				else:
					idx = dx.argmax()
					fores.append(targets[idx])

			return torch.stack(fores, dim=0)

	def left_fore(self, vehs, window):
		if len(vehs.shape) == 1:
			return self.left_fore(vehs.unsqueeze(0), window).squeeze()

		query = vehs.clone()
		query[:, 1] += self.lane_width
		return self.fore(query, window)

	def left_rear(self, vehs, window):
		if len(vehs.shape) == 1:
			return self.left_rear(vehs.unsqueeze(0), window).squeeze()

		query = vehs.clone()
		query[:, 1] += self.lane_width
		return self.rear(query, window)

	def right_fore(self, vehs, window):
		if len(vehs.shape) == 1:
			return self.right_fore(vehs.unsqueeze(0), window).squeeze()

		query = vehs.clone()
		query[:, 1] -= self.lane_width
		return self.fore(query, window)

	def right_rear(self, vehs, window):
		if len(vehs.shape) == 1:
			return self.right_rear(vehs.unsqueeze(0), window).squeeze()

		query = vehs.clone()
		query[:, 1] -= self.lane_width
		return self.rear(query, window)

	def initialize_scene(self, window = 200.0, density = 25.0, prob = 0.4, seed = None):
		"""
		Returns a tensor of vehicle states that can serve as an initialization point for highway driving. 
		Generates vehicles in [-window, window] for all lanes.
		I don't have a great way of doing this, so heuristically, I'm using this algo:
		
		for each lane, but a car every density m with probability dependent on lane.

		Can specify a seed for reproducibility.
		"""
		if seed:
			torch.manual_seed(seed)

		r = torch.rand((self.n_lanes, int(2*window/density)))
		#get the x, y coordinates of start point. Assume v starts at speed limit and heading starts at 0.
		x = torch.stack([torch.arange(-window, window, density) for i in range(self.n_lanes)]).to(self.device)
		x += torch.rand((self.n_lanes, 1)) * density
		y = torch.stack([torch.ones(int(2*window/density)) * (i*self.lane_width + (self.lane_width/2)) for i in range(self.n_lanes)]).to(self.device)
		h = torch.zeros((self.n_lanes, int(2*window/density))).to(self.device)
		v = torch.ones((self.n_lanes, int(2*window/density))).to(self.device) * self.speed_limit

		scene = torch.stack([x, y, h, v], dim=2)
		scene = scene[r < prob]
		
		#remove the vehicle if its on top of ego-vehicle.
		scene = scene[~(scene[:, 1] == self.ego_state[1]) | ~((scene[:, 0] - self.ego_state[0]) < density)]
		return scene
