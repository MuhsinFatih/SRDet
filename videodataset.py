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


def _preprocess_image(feature):
	img = feature['image']
	img = tf.cast(img, tf.float32)
	img /= 255.0
	feature['image'] = img
	return feature


class FrameGenerator():
	def __init__(self, videoPaths, iteration_size):
		self.videoPaths = videoPaths
		self.videos = [cv2.VideoCapture(path) for path in videoPaths]
		self.totalFrames = np.array([vid.get(cv2.CAP_PROP_FRAME_COUNT) for vid in self.videos]).astype(np.int)
		self.iteration_size = iteration_size
	def call(self):
		while True:
			# let's just use the first video for testing:
			i_vid = 0
			vid = self.videos[i_vid]
			idx_frame = np.random.choice(self.totalFrames[i_vid], self.totalFrames[i_vid], replace=True)
			for i in range(self.iteration_size):
				vid.set(cv2.CAP_PROP_POS_FRAMES, idx_frame[i]) # set video to this frame
				# yield {
				# 	'video_index': i_vid,
				# 	'video_path': self.videoPaths[i_vid],
				# 	'frame': vid.read()[1]
				# }
				yield vid.read()[1]

if __name__ == "__main__":
	videoPaths = np.array(glob2.glob(virat.ground.video.dir + '/*.mp4'))
	generator = FrameGenerator(videoPaths)
	
	dataset = tf.data.Dataset.from_generator(generator.call, output_types={'video_index': tf.float32, 'video_path': tf.string, 'frame': tf.uint8})
	for sample in dataset.take(3):
		plt.figure()
		plt.title(f"index: {sample['video_index'].numpy()}, path: {sample['video_path'].numpy()}")
		plt.imshow(sample['frame'])