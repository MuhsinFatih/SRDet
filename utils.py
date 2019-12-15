class new(object):
	def __init__(self, **kwargs):
		for (k,v) in kwargs.items():
			self.__setattr__(k, v)
	def __repr__(self):
		return 'literal(%s)' % ', '.join('%s = %r' % i for i in sorted(self.__dict__.iteritems()))
	def __str__(self):
		return repr(self)

#%%
from tensorflow.python.client import device_lib
from decimal import Decimal

import tensorflow as tf
import numpy as np
import datetime
import pathlib
import os
def job(name, date=None):
	_outputs_dir="outputs"
	if date is None:
		now = datetime.datetime.now()
		date = now.strftime("%m.%d")
	_jobdir_relative = date + "_" + name
	OUTDIR=os.path.join(_outputs_dir, _jobdir_relative)
	pathlib.Path(OUTDIR).mkdir(parents=True, exist_ok=True)
	print("="*20)
	print("job date: ", date)
	print("job name: ", name)
	print("output directory: ", OUTDIR)
	print("="*20)
	return OUTDIR

def gaussian_kernel(size: int, mean: float, std: float):
	"""Makes 2D gaussian Kernel for convolution."""

	d = tf.distributions.Normal(mean, std)

	vals = d.prob(tf.range(start = -size, limit = size + 1, dtype = tf.float32))

	gauss_kernel = tf.einsum('i,j->ij',
								  vals,
								  vals)

	return gauss_kernel / tf.reduce_sum(gauss_kernel)