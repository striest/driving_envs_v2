import torch
from drivingenvs.vehicles.base import Vehicle

class AckermannSteeredVehicle(Vehicle):
	"""
	Class for ackermann-steered vehicles, i.e. given linear velocity and turn angle, produce changes in heading, x and y.
	"""
	def __init__(self, veh_shape):
		self.length = veh_shape[0]
		self.width = veh_shape[1]

	@property
	def shape(self):
		return (self.length, self.width)

	@property
	def variables(self):
		return {
			'length': self.length,
			'width':self.width
			}

	@property
	def state(self):
		return {
			'x':0,
			'y':1,
			'theta':2
			}

	@property
	def control(self):
		return {
			'velocity':0,
			'heading':1
			}

	def propagate(self, state, control):
		phi = control['heading']
		vx = control['velocity']
		dx = vx * torch.cos(state['theta'])
		dy = vx * torch.sin(state['theta'])
		dtheta = (vx/self.length) * torch.tan(phi)

		return torch.tensor([dx, dy, dtheta])
		
