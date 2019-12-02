import os
from utils import *
data_dir = 'data'

virat = new(dir = os.path.join(data_dir, 'VIRAT/Public Dataset/VIRAT Video Dataset Release 2.0/'))
virat.aerial = new(dir = os.path.join(virat.dir, 'VIRAT Aerial Dataset'))
virat.ground = new(dir = os.path.join(virat.dir, 'VIRAT Ground Dataset'))
virat.ground.video = new(dir = os.path.join(virat.ground.dir, 'videos_original'))
