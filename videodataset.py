#%%
import tensorflow as tf
import tensorflow.keras
# import tensorflow_datasets as tfds
# from tfds.features import Video
import numpy as np
import pathlib
import glob2
import time
import cv2
import sys
import os

self_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(self_dir)

import matplotlib.pyplot as plt
from srgan.config import *
from config2 import *


def _preprocess_image(feature):
	img = feature['image']
	img = tf.cast(img, tf.float32)
	img /= 255.0
	feature['image'] = img
	return feature


class FrameGenerator():
	def __init__(self, videoPaths, iteration_size, isTest=False):
		self.videoPaths = videoPaths
		print('getting vids')
		self.videos = [cv2.VideoCapture(path) for path in videoPaths]
		self.totalFrames = np.array([vid.get(cv2.CAP_PROP_FRAME_COUNT) for vid in self.videos]).astype(np.int)
		self.iteration_size = iteration_size
		self.ind_vid = np.arange(len(self.videos[:-1])) # keep the last as test set
		np.random.seed(0)
		np.random.shuffle(self.ind_vid)
		self.isTest = isTest
		if self.isTest:
			self.i_vid = -1
		else:
			self.i_vid = 0

	def call(self):
		vid = self.videos[self.i_vid]
		n_frames = min(self.iteration_size, self.totalFrames[self.i_vid])
		idx_frame = np.random.choice(self.totalFrames[self.i_vid], n_frames, replace=False) # random n frames that are different than each other
		idx_frame = sorted(idx_frame) # read in order to reduce latency introduced by random access
		for i in range(n_frames):
			print('idx:', idx_frame[i], i, len(idx_frame), self.totalFrames[self.i_vid])
			vid.set(cv2.CAP_PROP_POS_FRAMES, idx_frame[i]) # set video to this frame
			# yield {
			# 	'video_index': i_vid,
			# 	'video_path': self.videoPaths[i_vid],
			# 	'frame': vid.read()[1]
			# }
			img = vid.read()[1]
			img = cv2.blur(img,(5,5))
			yield img
		if not self.isTest:
			self.i_vid += 1

class FrameGeneratorInterleaved():
	def __init__(self, videoPaths, iteration_size, isTest=False):
		self.videoPaths = videoPaths
		print('getting vids')
		self.videos = [cv2.VideoCapture(path) for path in videoPaths]
		self.totalFrames = np.array([vid.get(cv2.CAP_PROP_FRAME_COUNT) for vid in self.videos]).astype(np.int)
		self.iteration_size = iteration_size
		self.ind_vid = np.arange(len(self.videos[:-1])) # keep the last as test set
		np.random.seed(0)
		np.random.shuffle(self.ind_vid)
		self.isTest = isTest
		if self.isTest:
			self.i_vid = -1
		else:
			self.i_vid = 0

	def call(self):
		vid = self.videos[self.i_vid]
		n_frames = min(self.iteration_size, self.totalFrames[self.i_vid])
		idx_frame = np.random.choice(self.totalFrames[self.i_vid], n_frames, replace=False) # random n frames that are different than each other
		idx_frame = sorted(idx_frame) # read in order to reduce latency introduced by random access
		for i in range(n_frames):
			print('idx:', idx_frame[i], i, len(idx_frame), self.totalFrames[self.i_vid])
			vid.set(cv2.CAP_PROP_POS_FRAMES, idx_frame[i]) # set video to this frame
			# yield {
			# 	'video_index': i_vid,
			# 	'video_path': self.videoPaths[i_vid],
			# 	'frame': vid.read()[1]
			# }
			img = vid.read()[1]
			img = cv2.blur(img,(5,5))
			yield img
		if not self.isTest:
			self.i_vid += 1


if __name__ == "__main__":
	videoPaths = np.array(glob2.glob(virat.ground.video.dir + '/*.mp4'))
	generator = FrameGenerator(videoPaths, iteration_size=96)
	
	# dataset = tf.data.Dataset.from_generator(generator.call, output_types={'video_index': tf.float32, 'video_path': tf.string, 'frame': tf.uint8})
	dataset = tf.data.Dataset.from_generator(generator.call, output_types=(tf.float32)).batch(8)
	start_time = time.time()
	for sample in dataset.take(96):
		print('got sample')
		# plt.figure()
		# plt.title(f"index: {sample['video_index'].numpy()}, path: {sample['video_path'].numpy()}")
		# plt.imshow(sample['frame'])
	elapsed_time = time.time() - start_time
	print('elapsed_time: ', elapsed_time)