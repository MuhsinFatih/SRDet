class new(object):
	def __init__(self, **kwargs):
		for (k,v) in kwargs.items():
			self.__setattr__(k, v)
	def __repr__(self):
		return 'literal(%s)' % ', '.join('%s = %r' % i for i in sorted(self.__dict__.iteritems()))
	def __str__(self):
		return repr(self)