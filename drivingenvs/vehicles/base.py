import abc
import torch

class VehicleSpec(object, metaclass=abc.ABCMeta):
	"""
	Class for generating vehicles. I.e. given a vehiclespec, get a vehicle that meets certain parameters.
	"""

	@abc.abstractmethod
	def sample(self):
		pass

class Vehicle(object, metaclass=abc.ABCMeta):
	"""
	A vehicle. We should be able to give it control inputs, and it should give us changes in state.
	"""

	@property
	@abc.abstractmethod
	def shape(self):
		"""
		Returns the shape of the vehicle as a (length, width) tuple
		"""
		pass

	@property
	@abc.abstractmethod
	def state(self):
		"""
		Not the actual state, but a string->int dict that says what position in the state-change vector corresponds to what.
		"""
		pass

	@property
	@abc.abstractmethod
	def control(self):
		"""
		Not the actual state, but a string->int dict that says what position in the state-change vector corresponds to what.
		"""
		pass

	@property
	@abc.abstractmethod
	def variables(self):
		"""
		Returns a dict of name->state var of the values in the vehicle that can vary (i.e. vehicle size, max speed, etc.)
		"""
		pass

	def propagate_from_tensor(self, state_tensor, control_tensor):
		state_dict = {}
		control_dict = {}

		for k, idx in self.state.items():
			if len(state_tensor.shape) == 1:
				state_dict[k] = state_tensor[idx]
			else:
				state_dict[k] = state_tensor[:, idx]

		for k, idx in self.control.items():
			if len(control_tensor.shape) == 1:
				control_dict[k] = control_tensor[idx]
			else:
				control_dict[k] = control_tensor[:, idx]

		return self.propagate(state_dict, control_dict).to(state_tensor.device)

	@abc.abstractmethod
	def propagate(self, state, control):
		"""
		Given a control vector, return the change in vehicle state.
		"""
		pass

