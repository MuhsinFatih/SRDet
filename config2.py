import os
from utils import *
import sys
self_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(self_dir)

data_dir = os.path.join(os.path.dirname(__file__), 'data')

virat = new(dir = os.path.join(data_dir, 'VIRAT/Public Dataset/VIRAT Video Dataset Release 2.0/'))
virat.aerial = new(dir = os.path.join(virat.dir, 'VIRAT Aerial Dataset'))
virat.ground = new(dir = os.path.join(virat.dir, 'VIRAT Ground Dataset'))
virat.ground.video = new(dir = os.path.join(virat.ground.dir, 'videos_original'))
