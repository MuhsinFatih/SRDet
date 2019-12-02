#%%
import tensorflow as tf
import tensorflow.keras
import tensorflow_datasets as tfds
# from tfds.features import Video
import numpy as np
import pathlib
import glob2
import cv2

import matplotlib.pyplot as plt
from config import *

vidPaths = np.array(glob2.glob(virat.ground.video.dir + '/*.mp4'))

vid = cv2.VideoCapture(vidPaths[0])
totalFrames = vid.get(cv2.CAP_PROP_FRAME_COUNT)
print(totalFrames)
frameNumber = 3000
vid.set(cv2.CAP_PROP_POS_FRAMES, frameNumber)
ret, frame = vid.read()
plt.imshow(frame)