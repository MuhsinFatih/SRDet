#! /usr/bin/python
# -*- coding: utf8 -*-
#%%

import os
import time
import random
import numpy as np
import scipy, multiprocessing
import tensorflow as tf
import tensorlayer as tl
from model import get_G, get_D
from config import config

import sys
sys.path.append('..')
import videodataset
from config2 import *
import glob2

from scipy.misc import imsave
from PIL import Image
import PIL
import pathlib
AUTOTUNE = tf.data.experimental.AUTOTUNE

###====================== HYPER-PARAMETERS ===========================###
## Adam
batch_size = config.TRAIN.batch_size  # use 8 if your GPU memory is small, and change [4, 4] in tl.vis.save_images to [2, 4]
lr_init = config.TRAIN.lr_init
beta1 = config.TRAIN.beta1
## initialize G
n_epoch_init = config.TRAIN.n_epoch_init
## adversarial learning (SRGAN)
n_epoch = config.TRAIN.n_epoch
lr_decay = config.TRAIN.lr_decay
decay_every = config.TRAIN.decay_every
shuffle_buffer_size = 128
iteration_size = 96
# ni = int(np.sqrt(batch_size))

import argparse
parser = argparse.ArgumentParser()

parser.add_argument('--exp', type=str, default='training', help='experiment name')
parser.add_argument('--mode', type=str, default='srgan', help='srgan, evaluate')
parser.add_argument('--inputsize', type=int, default=96)

args = parser.parse_args()

inputsize = args.inputsize
outdir = job(f'{args.exp}')

# create folders to save result images and trained models
save_dir = os.path.join(outdir, "samples")
tl.files.exists_or_mkdir(save_dir)
checkpoint_dir = os.path.join(outdir, "models")
# checkpoint_dir = "models"
tl.files.exists_or_mkdir(checkpoint_dir)

def get_train_data():
	# load dataset
	# train_hr_img_list = sorted(tl.files.load_file_list(path=config.TRAIN.hr_img_path, regx='.*.png', printable=False))#[0:20]
		# train_lr_img_list = sorted(tl.files.load_file_list(path=config.TRAIN.lr_img_path, regx='.*.png', printable=False))
		# valid_hr_img_list = sorted(tl.files.load_file_list(path=config.VALID.hr_img_path, regx='.*.png', printable=False))
		# valid_lr_img_list = sorted(tl.files.load_file_list(path=config.VALID.lr_img_path, regx='.*.png', printable=False))

	## If your machine have enough memory, please pre-load the entire train set.
	# train_hr_imgs = tl.vis.read_images(train_hr_img_list, path=config.TRAIN.hr_img_path, n_threads=32)
		# for im in train_hr_imgs:
		#     print(im.shape)
		# valid_lr_imgs = tl.vis.read_images(valid_lr_img_list, path=config.VALID.lr_img_path, n_threads=32)
		# for im in valid_lr_imgs:
		#     print(im.shape)
		# valid_hr_imgs = tl.vis.read_images(valid_hr_img_list, path=config.VALID.hr_img_path, n_threads=32)
		# for im in valid_hr_imgs:
		#     print(im.shape)
		
	# dataset API and augmentation
	# def generator_train():
	# 	for img in train_hr_imgs:
	# 		yield img

	videoPaths = np.array(glob2.glob(virat.ground.video.dir + '/*.mp4'))
	generator = videodataset.FrameGenerator(videoPaths, iteration_size)

	def _map_fn_train(img):
		hr_patch = tf.image.random_crop(img, [384, 384, 3])
		hr_patch = hr_patch / (255. / 2.)
		hr_patch = hr_patch - 1.
		hr_patch = tf.image.random_flip_left_right(hr_patch)
		lr_patch = tf.image.resize(hr_patch, size=[inputsize, inputsize]) #64, 48, 36
		lr_patch = tf.image.resize(lr_patch, size=[96, 96]) # re-upsample if it was lower than this. 96x96 is the input size of the network
		return lr_patch, hr_patch
	
	train_ds = tf.data.Dataset.from_generator(generator.call, output_types=(tf.float32))
	# train_ds = tf.data.Dataset.from_generator(generator_train, output_types=(tf.float32))
	# print(next(iter(train_ds)).numpy())
	# return
	example = next(iter(train_ds))
	imsave(os.path.join(outdir,"input_example.jpg"), example.numpy())
	train_ds = train_ds.map(_map_fn_train, num_parallel_calls=AUTOTUNE)
	examples = next(iter(train_ds))
	print(examples[1].numpy().shape)
	imsave(os.path.join(outdir,"lowres_example.jpg"), examples[0].numpy())
	imsave(os.path.join(outdir,"highres_example.jpg"), examples[1].numpy())

		# train_ds = train_ds.repeat(n_epoch_init + n_epoch)
	train_ds = train_ds.shuffle(shuffle_buffer_size)
	train_ds = train_ds.prefetch(AUTOTUNE)
	train_ds = train_ds.batch(batch_size)
		# value = train_ds.make_one_shot_iterator().get_next()
	return train_ds

def train():
	G = get_G((batch_size, 96, 96, 3))
	D = get_D((batch_size, 384, 384, 3))
	VGG = tl.models.vgg19(pretrained=True, end_with='pool4', mode='static')

	lr_v = tf.Variable(lr_init)
	g_optimizer_init = tf.optimizers.Adam(lr_v, beta_1=beta1)
	g_optimizer = tf.optimizers.Adam(lr_v, beta_1=beta1)
	d_optimizer = tf.optimizers.Adam(lr_v, beta_1=beta1)

	G.train()
	D.train()
	VGG.train()

	train_ds = get_train_data()

	## initialize learning (G)
	# n_step_epoch = round(n_epoch_init // batch_size)
	# for epoch in range(n_epoch_init):
	# 	for step, (lr_patchs, hr_patchs) in enumerate(train_ds):
	# 		print('lr_patchs.shape: ', lr_patchs.shape, 'batch_size:', batch_size)
	# 		if lr_patchs.shape[0] != batch_size: # if the remaining data in this epoch < batch_size
	# 			break
	# 		step_time = time.time()
	# 		with tf.GradientTape() as tape:
	# 			fake_hr_patchs = G(lr_patchs)
	# 			mse_loss = tl.cost.mean_squared_error(fake_hr_patchs, hr_patchs, is_mean=True)
	# 		grad = tape.gradient(mse_loss, G.trainable_weights)
	# 		g_optimizer_init.apply_gradients(zip(grad, G.trainable_weights))
	# 		print("Epoch: [{}/{}] step: [{}/{}] time: {:.3f}s, mse: {:.3f} ".format(
	# 			epoch, n_epoch_init, step, n_step_epoch, time.time() - step_time, mse_loss))
	# 	if (epoch != 0) and (epoch % 10 == 0):
	# 		tl.vis.save_images(fake_hr_patchs.numpy(), [2, 4], os.path.join(save_dir, 'train_g_init_{}.png'.format(epoch)))

	## adversarial learning (G, D)
	n_step_epoch = round(iteration_size // batch_size)
	for epoch in range(n_epoch):
		for step, (lr_patchs, hr_patchs) in enumerate(train_ds):
			if lr_patchs.shape[0] != batch_size: # if the remaining data in this epoch < batch_size
				break
			step_time = time.time()
			with tf.GradientTape(persistent=True) as tape:
				fake_patchs = G(lr_patchs)
				logits_fake = D(fake_patchs)
				logits_real = D(hr_patchs)
				feature_fake = VGG((fake_patchs+1)/2.) # the pre-trained VGG uses the input range of [0, 1]
				feature_real = VGG((hr_patchs+1)/2.)
				d_loss1 = tl.cost.sigmoid_cross_entropy(logits_real, tf.ones_like(logits_real))
				d_loss2 = tl.cost.sigmoid_cross_entropy(logits_fake, tf.zeros_like(logits_fake))
				d_loss = d_loss1 + d_loss2
				g_gan_loss = 1e-3 * tl.cost.sigmoid_cross_entropy(logits_fake, tf.ones_like(logits_fake))
				mse_loss = tl.cost.mean_squared_error(fake_patchs, hr_patchs, is_mean=True)
				vgg_loss = 2e-6 * tl.cost.mean_squared_error(feature_fake, feature_real, is_mean=True)
				g_loss = mse_loss + vgg_loss + g_gan_loss
			grad = tape.gradient(g_loss, G.trainable_weights)
			g_optimizer.apply_gradients(zip(grad, G.trainable_weights))
			grad = tape.gradient(d_loss, D.trainable_weights)
			d_optimizer.apply_gradients(zip(grad, D.trainable_weights))
			print("Epoch: [{}/{}] step: [{}/{}] time: {:.3f}s, g_loss(mse:{:.3f}, vgg:{:.3f}, adv:{:.3f}) d_loss: {:.3f}".format(
				epoch, n_epoch, step, n_step_epoch, time.time() - step_time, mse_loss, vgg_loss, g_gan_loss, d_loss))

		# update the learning rate
		if epoch != 0 and (epoch % decay_every == 0):
			new_lr_decay = lr_decay**(epoch // decay_every)
			lr_v.assign(lr_init * new_lr_decay)
			log = " ** new learning rate: %f (for GAN)" % (lr_init * new_lr_decay)
			print(log)

		if (epoch != 0) and (epoch % 10 == 0):
			tl.vis.save_images(lr_patchs.numpy(), [2, 4], os.path.join(save_dir, 'inputs_{}.png'.format(epoch)))
			tl.vis.save_images(fake_patchs.numpy(), [2, 4], os.path.join(save_dir, 'train_g_{}.png'.format(epoch)))
			G.save_weights(os.path.join(checkpoint_dir, 'g.h5'))
			D.save_weights(os.path.join(checkpoint_dir, 'd.h5'))
def __evaluate(ds_lowres, ds_highres):
	G = get_G([1, None, None, 3])
	G.load_weights(os.path.join(checkpoint_dir, 'g.h5'))
	G.eval()

	for i,valid_lr_img in enumerate(ds_lowres):
		valid_lr_img = valid_lr_img.numpy() #[img[1].numpy() for img in next(iter(ds_lowres.take(65)))]


		valid_lr_img = np.asarray(valid_lr_img, dtype=np.float32)
		valid_lr_img = valid_lr_img[np.newaxis,:,:,:]
		size = [valid_lr_img.shape[1], valid_lr_img.shape[2]]

		out = G(valid_lr_img).numpy()

		print("LR size: %s /  generated HR size: %s" % (size, out.shape))  # LR size: (339, 510, 3) /  gen HR size: (1, 1356, 2040, 3)
		print("[*] save images")
		pathlib.Path(os.path.join(save_dir, str(i))).mkdir(parents=True, exist_ok=True)
		tl.vis.save_image(out[0], os.path.join(save_dir, str(i), 'valid_gen.png'))
		tl.vis.save_image(valid_lr_img[0], os.path.join(save_dir, str(i), 'valid_lr.png'))
		# tl.vis.save_image(valid_hr_img, os.path.join(save_dir, 'valid_hr.png'))

		out_bicu = scipy.misc.imresize(valid_lr_img[0], [size[0] * 4, size[1] * 4], interp='bicubic', mode=None)
		tl.vis.save_image(out_bicu, os.path.join(save_dir, str(i), 'valid_bicubic.png'))

def evaluate():
	use_test_folder = True
	if use_test_folder:
		filenames = np.array(glob2.glob("Test_images/*.png"))
		path_ds = tf.data.Dataset.from_tensor_slices(filenames)
		def _map_fn_test(path):
			raw_image = tf.io.read_file(path)
			img = tf.image.decode_jpeg(raw_image,channels=3)
			img = tf.cast(img, tf.float32)
			# gaus_kernel = gaussian_kernel(5, 0, 5)[:, :, tf.newaxis, tf.newaxis]
			# img = tf.nn.conv2d(img, gauss_kernel, strides=[1, 1, 1, 1], padding="SAME")
			img = tf.image.resize(img, size=[96, 96])
			img = img / (255. / 2.)
			img = img - 1.
			return img
		ds_lowres = path_ds.map(_map_fn_test, num_parallel_calls=AUTOTUNE)
		ds_highres = None
	else:
		videoPaths = np.array(glob2.glob(virat.ground.video.dir + '/*.mp4'))
		generator = videodataset.FrameGenerator(videoPaths, iteration_size)

		def _map_fn_train(img):
			hr_patch = tf.image.random_crop(img, [384, 384, 3])
			hr_patch = hr_patch / (255. / 2.)
			hr_patch = hr_patch - 1.
			lr_patch = tf.image.resize(hr_patch, size=[96, 96])
			return lr_patch, hr_patch
		
		ds_lowres = tf.data.Dataset.from_generator(generator.call, output_types=(tf.float32))
		ds_highres = tf.data.Dataset.from_generator(generator.call, output_types=(tf.float32))
		ds_lowres = ds_lowres.map(_map_fn_train, num_parallel_calls=AUTOTUNE)
	__evaluate(ds_lowres, ds_highres)

if __name__ == '__main__':
	tl.global_flag['mode'] = args.mode

	if tl.global_flag['mode'] == 'srgan':
		train()
	elif tl.global_flag['mode'] == 'evaluate':
		evaluate()
	else:
		raise Exception("Unknow --mode")