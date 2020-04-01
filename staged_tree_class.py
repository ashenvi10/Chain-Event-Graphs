from collections import defaultdict
import scipy.special as special
from operator import add
import pydotplus as ptp
from IPython.display import Image
import random

class staged_tree(object):
	def __init__(self, dataframe):
		