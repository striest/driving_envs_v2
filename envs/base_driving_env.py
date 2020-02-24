import torch
import gym

from yarp.envs.base import Env

class BaseDrivingEnv(Env):
	"""
	Basic driving environment with just the ego vehicle and some number of lanes. Ego vehicle position is given as the location of the rear wheel axle. (For ackermann steering)
	Default vehicle width from https://ops.fhwa.dot.gov/freight/publications/size_regs_final_rpt/.
	Length from https://mechanicbase.com/cars/average-car-length/
	Bounds on acceleration and turn rate came from CMU work.
	"""
	def __init__(self, distance=1000.0, dt=0.1, veh_dim=(4.5, 2.4), lane_width=3.7, n_lanes=3, acc_bounds = (-4, 4), steer_bounds = (-.2618, .2618), gpu=False):
		self.gpu = gpu
		self.max_distance = distance;
		self.dt = dt
		self.ego_dim = torch.tensor([i for i in veh_dim]).to(self.device)
		self.n_lanes = n_lanes
		self.lane_width = lane_width
		self.lane_loc = torch.linspace(0, n_lanes*lane_width, n_lanes+1)

		#I've left these as tuples so I can use them as args for torch.clamp
		self.steer_bounds = steer_bounds
		self.acc_bounds = acc_bounds

		#Ego state is a tensor of [x_pos, y_pos, vel, heading]. Controls are acceleration and yaw rate.
		self.ego_state = torch.zeros(4).to(self.device)
		#Gets the state of the vehicle relative to the lane that it's in (this just amounts to changing the y value for straight lanes).

	def reset(self, init_speed=21.0):
		"""
		Reset the ego vehicle to the center of a random lane. Initialize the vehicle with the given speed.
		"""
		y_pos = self.lane_loc[:-1][torch.randint(self.n_lanes, (1, ))] + self.lane_width/2
		x_pos = 0.0
		v = init_speed
		heading = 0.0

		self.ego_state = torch.tensor([x_pos, y_pos, v, heading]).to(self.device)

	def step(self, actions):
		"""
		Step in all current envs. Again, note the vectorization of the env.
		"""
		pass

	def action_space(self):
		"""
		Action space for this env is a continuous valued acceleration and yaw rate, bounded by the instance variables given in the problem.
		"""
		return gym.spaces.Box(low=(self.acc_bounds[0], self.steer_bounds[0]), high=(self.acc_bounds[1], self.steer_bounds[1]))

	def observation_space(self):
		"""
		Returns the dimension of the observation space
		"""
		pass

	def render(self):
		"""
		for visualization.
		"""
		pass

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
		lane_offset = lane_offsets[lane_offsets>=0].min()
		out = self.ego_state.clone()
		out[1] = lane_offset
		return out

	@property
	def observation(self):
		"""
		For now, let observation be [y_pos, vel, heading, dist to left lane line, dist to right lane line]
		We can make the dist func more sophisticated to handle cars being shifted, but lets start simple.
		"""
		ego = self.ego_state.clone()[1:]
		lane_pos = self.lane_relative_state[1]
		ll_dist = self.lane_width - (lane_pos + self.ego_dim[1]/2)
		rl_dist = lane_pos - self.ego_dim[1]/2
		lane_obs = torch.tensor([ll_dist, rl_dist]).to(self.device)
		return torch.cat([ego, lane_obs], dim=0)
