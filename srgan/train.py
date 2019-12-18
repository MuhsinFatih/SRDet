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
iteration_size = 96*4
# ni = int(np.sqrt(batch_size))

import argparse
parser = argparse.ArgumentParser()

parser.add_argument('--exp', type=str, default='training', help='experiment name. this will create an experiment folder with today\'s timestamp')
parser.add_argument('--exp_dir', type=str, default=None, help='exact experiment name (don\'t add today\'s timestamp)')
parser.add_argument('--out_name', type=str, default=None, help='folder name in samples to save outputs (for evaluation)')
parser.add_argument('--mode', type=str, default='srgan', help='srgan, evaluate')
parser.add_argument('--inputsize', type=int, default=96)

args = parser.parse_args()

inputsize = args.inputsize

if args.exp_dir is None:
	outdir = job(f'{args.exp}')
else:
	outdir = os.path.join('outputs', args.exp_dir)
# create folders to save result images and trained models
if args.out_name is None:
	save_dir = os.path.join(outdir, "samples")
else:
	save_dir = os.path.join(outdir, "samples", args.out_name)
tl.files.exists_or_mkdir(save_dir)
checkpoint_dir = os.path.join(outdir, "models")
# checkpoint_dir = "models"
tl.files.exists_or_mkdir(checkpoint_dir)

def _map_fn_path2img(path):
	raw_image = tf.io.read_file(path)
	img = tf.image.decode_jpeg(raw_image,channels=3)
	img = tf.cast(img, tf.float32)
	return img

def _map_fn_gaussian(img):
	# gaus_kernel = gaussian_kernel(5, 0, 5)[:, :, tf.newaxis, tf.newaxis]
	# img = tf.nn.conv2d(img, gauss_kernel, strides=[1, 1, 1, 1], padding="SAME")
	pass

def _map_fn_preprocess(img):
	img = img / (255. / 2.)
	img = img - 1.
	return img

def _map_fn_downsample(img):
	hr_patch = tf.image.random_crop(img, [384, 384, 3], seed=0) # 720, 480, 3
	hr_patch = tf.image.random_flip_left_right(hr_patch)
	lr_patch = tf.image.resize(hr_patch, size=[inputsize, inputsize]) #64, 48, 36
	lr_patch = tf.image.resize(lr_patch, size=[96, 96]) # re-upsample if it was lower than this
	return lr_patch, hr_patch

def _map_fn_downsample_same(img):
	imgshape = tf.shape(img)
	lr_patch = tf.image.resize(img, size=[int(imgshape[0]/8), int(imgshape[1]/8)]) # re-upsample if it was lower than this
	return lr_patch, img

def _map_fn_downsample_centercrop(img):
	hr_patch = tf.image.crop_to_bounding_box(img,
		offset_height = int((1080-560)/2),
		offset_width = int((1920-560)/2),
		target_height = 560,
		target_width = 560
	)
	lr_patch = tf.image.resize(hr_patch, size=[inputsize, inputsize]) #64, 48, 36
	lr_patch = tf.image.resize(lr_patch, size=[140, 140]) # re-upsample if it was lower than this
	return lr_patch, hr_patch

def get_train_data():
	videoPaths = np.array(glob2.glob(virat.ground.video.dir + '/*.mp4'))
	generator = videodataset.FrameGeneratorInterleaved(videoPaths, iteration_size)
	
	train_ds = tf.data.Dataset.from_generator(generator.call, output_types=(tf.float32))
	# train_ds = tf.data.Dataset.from_generator(generator_train, output_types=(tf.float32))
	# print(next(iter(train_ds)).numpy())
	# return
	example = next(iter(train_ds))
	imsave(os.path.join(outdir,"input_example.jpg"), example.numpy())
	train_ds = train_ds.map(_map_fn_preprocess, num_parallel_calls=AUTOTUNE)
	train_ds = train_ds.map(_map_fn_downsample, num_parallel_calls=AUTOTUNE)
	examples = next(iter(train_ds))
	print(examples[1].numpy().shape)
	imsave(os.path.join(outdir,"lowres_example.jpg"), examples[0].numpy())
	imsave(os.path.join(outdir,"highres_example.jpg"), examples[1].numpy())

	# train_ds = train_ds.repeat(n_epoch_init + n_epoch)
	# train_ds = train_ds.shuffle(shuffle_buffer_size)
	train_ds = train_ds.prefetch(AUTOTUNE)
	train_ds = train_ds.batch(batch_size)
	# value = train_ds.make_one_shot_iterator().get_next()

	test_generator = videodataset.FrameGeneratorInterleaved(videoPaths, iteration_size, isTest=True)
	test_ds = tf.data.Dataset.from_generator(generator.call, output_types=(tf.float32))
	test_ds = test_ds.map(_map_fn_preprocess, num_parallel_calls=AUTOTUNE)
	test_ds = test_ds.map(_map_fn_downsample, num_parallel_calls=AUTOTUNE)
	test_ds = test_ds.prefetch(AUTOTUNE)
	test_ds = test_ds.batch(batch_size)
	
	eval_out_path = os.path.join(save_dir, 'test_folder')
	filenames = np.array(glob2.glob("Test_images/*.png") + glob2.glob("Test_images/*.jpg"))
	path_ds = tf.data.Dataset.from_tensor_slices(filenames)
	sample_ds = path_ds.map(_map_fn_path2img, num_parallel_calls=AUTOTUNE)
	sample_ds = sample_ds.map(_map_fn_preprocess, num_parallel_calls=AUTOTUNE)

	return train_ds, test_ds, sample_ds

def train():
	size = [1080, 1920]
	aspect_ratio = size[1] / size[0]

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

	train_ds, test_ds, sample_ds = get_train_data()

	sample_folders = ['train_lr', 'train_hr', 'train_gen', 'test_lr', 'test_hr', 'test_gen', 'sample_lr', 'sample_gen']
	for sample_folder in sample_folders:
		tl.files.exists_or_mkdir(os.path.join(save_dir, sample_folder))
	
	# only take a certain amount of images to save
	test_lr_patchs, test_hr_patchs = next(iter(test_ds))
	valid_lr_imgs = []
	for i,lr_patchs in enumerate(sample_ds):
		valid_lr_img = lr_patchs.numpy()
		valid_lr_img = np.asarray(valid_lr_img, dtype=np.float32)
		valid_lr_img = valid_lr_img[np.newaxis,:,:,:]
		valid_lr_imgs.append(valid_lr_img)
		tl.vis.save_images(valid_lr_img, [1,1], os.path.join(save_dir, 'sample_lr', 'sample_lr_img_{}.jpg'.format(i)))

	tl.vis.save_images(test_lr_patchs.numpy(), [2, 4], os.path.join(save_dir, 'test_lr', 'test_lr.jpg'))
	tl.vis.save_images(test_hr_patchs.numpy(), [2, 4], os.path.join(save_dir, 'test_hr', 'test_hr.jpg'))

	# initialize learning (G)
	n_step_epoch = round(iteration_size // batch_size)
	for epoch in range(n_epoch_init):
		for step, (lr_patchs, hr_patchs) in enumerate(train_ds):
			if lr_patchs.shape[0] != batch_size: # if the remaining data in this epoch < batch_size
				break
			step_time = time.time()
			with tf.GradientTape() as tape:
				fake_hr_patchs = G(lr_patchs)
				mse_loss = tl.cost.mean_squared_error(fake_hr_patchs, hr_patchs, is_mean=True)
			grad = tape.gradient(mse_loss, G.trainable_weights)
			g_optimizer_init.apply_gradients(zip(grad, G.trainable_weights))
			print("Epoch: [{}/{}] step: [{}/{}] time: {:.3f}s, mse: {:.3f} ".format(
				epoch, n_epoch_init, step, n_step_epoch, time.time() - step_time, mse_loss))
		if (epoch != 0) and (epoch % 10 == 0):
			# save training result examples
			tl.vis.save_images(lr_patchs.numpy(), [2, 4], os.path.join(save_dir, 'train_lr', 'train_lr_init_{}.jpg'.format(epoch)))
			tl.vis.save_images(hr_patchs.numpy(), [2, 4], os.path.join(save_dir, 'train_hr', 'train_hr_init_{}.jpg'.format(epoch)))
			tl.vis.save_images(fake_hr_patchs.numpy(), [2, 4], os.path.join(save_dir, 'train_gen', 'train_gen_init_{}.jpg'.format(epoch)))
			# save test results (only save generated, since it's always the same images. Inputs are saved before the training loop)
			fake_hr_patchs = G(test_lr_patchs)
			tl.vis.save_images(fake_hr_patchs.numpy(), [2, 4], os.path.join(save_dir, 'test_gen', 'test_gen_init_{}.jpg'.format(epoch)))
			# save sample results (only save generated, since it's always the same images. Inputs are saved before the training loop)
			for i,lr_patchs in enumerate(valid_lr_imgs):
				fake_hr_patchs = G(lr_patchs)
				tl.vis.save_images(fake_hr_patchs.numpy(), [1,1], os.path.join(save_dir, 'sample_gen', 'sample_gen_init_{}_img_{}.jpg'.format(epoch, i)))

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
			# save training result examples
			tl.vis.save_images(lr_patchs.numpy(), [2, 4], os.path.join(save_dir, 'train_lr', 'train_lr_{}.jpg'.format(epoch)))
			tl.vis.save_images(hr_patchs.numpy(), [2, 4], os.path.join(save_dir, 'train_hr', 'train_hr_{}.jpg'.format(epoch)))
			tl.vis.save_images(fake_patchs.numpy(), [2, 4], os.path.join(save_dir, 'train_gen', 'train_gen_{}.jpg'.format(epoch)))
			# save test results (only save generated, since it's always the same images. Inputs are saved before the training loop)
			fake_hr_patchs = G(test_lr_patchs)
			tl.vis.save_images(fake_hr_patchs.numpy(), [2, 4], os.path.join(save_dir, 'test_gen', 'test_gen_{}.jpg'.format(epoch)))
			# save sample results (only save generated, since it's always the same images. Inputs are saved before the training loop)
			# for i,lr_patchs in enumerate(valid_lr_imgs):
			# 	fake_hr_patchs = G(lr_patchs)
			# 	tl.vis.save_images(fake_hr_patchs.numpy(), [1,1], os.path.join(save_dir, 'sample_gen', 'sample_gen_init_{}_img_{}.jpg'.format(epoch, i)))


			G.save_weights(os.path.join(checkpoint_dir, f'g_epoch_{epoch}.h5'))
			D.save_weights(os.path.join(checkpoint_dir, f'd_epoch_{epoch}.h5'))

			G.save_weights(os.path.join(checkpoint_dir, 'g.h5'))
			D.save_weights(os.path.join(checkpoint_dir, 'd.h5'))


def __evaluate(ds, eval_out_path, filenames=None):
	G = get_G([1, None, None, 3])
	G.load_weights(os.path.join(checkpoint_dir, 'g_epoch_540.h5'))
	G.eval()
	sample_folders = ['lr', 'hr', 'gen', 'bicubic', 'combined']
	for sample_folder in sample_folders:
		tl.files.exists_or_mkdir(os.path.join(eval_out_path, sample_folder))

	# for i,(filename, valid_lr_img) in enumerate(zip(filenames, ds)):
	for i,(valid_lr_img, valid_hr_img) in enumerate(ds):
		valid_lr_img = valid_lr_img.numpy()

		valid_lr_img = np.asarray(valid_lr_img, dtype=np.float32)
		valid_lr_img = valid_lr_img[np.newaxis,:,:,:]
		size = [valid_lr_img.shape[1], valid_lr_img.shape[2]]

		out = G(valid_lr_img).numpy()

		print("LR size: %s /  generated HR size: %s" % (size, out.shape))  # LR size: (339, 510, 3) /  gen HR size: (1, 1356, 2040, 3)
		print("[*] save images")
		if filenames is None:
			tl.vis.save_image(out[0], os.path.join(eval_out_path, 'gen', f'valid_gen_{i}.jpg'))
			tl.vis.save_image(valid_lr_img[0], os.path.join(eval_out_path, 'lr', f'valid_lr_{i}.jpg'))
			tl.vis.save_image(valid_hr_img, os.path.join(eval_out_path, 'hr', f'valid_hr_{i}.jpg'))

			out_bicu = scipy.misc.imresize(valid_lr_img[0], [size[0] * 4, size[1] * 4], interp='bicubic', mode=None)
			tl.vis.save_image(out_bicu, os.path.join(eval_out_path, 'bicubic', f'valid_bicu_{i}.jpg'))
			# tl.vis.save_images(np.array([valid_lr_img[0], np.array(out_bicu), out[0]]), [1,3], os.path.join(eval_out_path, 'combined', f'valid_bicu_{i}.jpg'))
		else:
			tl.vis.save_image(out[0], os.path.join(eval_out_path, 'gen', filename))
			tl.vis.save_image(valid_lr_img[0], os.path.join(eval_out_path, 'lr', filename))
			# tl.vis.save_image(valid_hr_img, os.path.join(eval_out_path, 'hr', filename))

			out_bicu = scipy.misc.imresize(valid_lr_img[0], [size[0] * 4, size[1] * 4], interp='bicubic', mode=None)
			tl.vis.save_image(out_bicu, os.path.join(eval_out_path, 'bicubic', filename))
			# tl.vis.save_images(np.array([valid_lr_img[0], np.array(out_bicu), out[0]]), [1,3], os.path.join(eval_out_path, 'combined', f'valid_bicu_{i}.jpg'))
def other():
	
	if 1:
		train_ds, test_ds, sample_ds = get_train_data()
		test_ds = test_ds.unbatch()
		candidate_dir = os.path.join(save_dir, 'handPickCandidates_jpg_new')
		tl.files.exists_or_mkdir(candidate_dir)
		test_ds = test_ds.take(1000)
		__evaluate(test_ds, candidate_dir)
		# for i, img in enumerate(test_ds.take(500)):
		# 	tl.vis.save_image(img.numpy(), os.path.join(candidate_dir, f'testimg{i}.jpg'))


	
	if 0: # size experiment
		eval_out_path = os.path.join(save_dir, 'test_video')
		videoPaths = np.array(glob2.glob(virat.ground.video.dir + '/*.mp4'))
		generator = videodataset.FrameGenerator(videoPaths, iteration_size)
		
		ds_highres = tf.data.Dataset.from_generator(generator.call, output_types=(tf.float32))
		def _downsample(img):
			# img = tf.image.decode_jpeg(img)
			# orig_size = [1072, 1920]
			# aspect_ratio = orig_size[1] / orig_size[0]
			# lr_patch = tf.image.resize(img, size=[384*aspect_ratio, 384]) #64, 48, 36
			return img
		ds_lowres = ds_highres.map(_downsample, num_parallel_calls=AUTOTUNE)
		for img in ds_lowres.take(1):
			# print(img.numpy())
			imsave(os.path.join(outdir,"resizetest.jpg"), img.numpy())
		return

def evaluate():
	use_test_folder = True
	if use_test_folder:
		eval_out_path = os.path.join(save_dir, 'test_folder')
		filenames = np.array(glob2.glob("Test_images/*.png") + glob2.glob("Test_images/*.jpg"))
		path_ds = tf.data.Dataset.from_tensor_slices(filenames)
		ds_lowres = path_ds.map(_map_fn_path2img, num_parallel_calls=AUTOTUNE)
		ds_lowres = ds_lowres.map(_map_fn_preprocess, num_parallel_calls=AUTOTUNE)
		ds_highres = None
		filenames = [os.path.basename(name) for name in filenames]
	else:
		eval_out_path = os.path.join(save_dir, 'test_video')
		videoPaths = np.array(glob2.glob(virat.ground.video.dir + '/*.mp4'))
		generator = videodataset.FrameGenerator(videoPaths, iteration_size, isTest=True)
		ds_highres = tf.data.Dataset.from_generator(generator.call, output_types=(tf.float32))
		ds_highres = ds_highres.map(_map_fn_preprocess, num_parallel_calls=AUTOTUNE).take(10)
		

		# downsample to get the lowres dataset
		ds_lowres = ds_highres.map(_map_fn_downsample, num_parallel_calls=AUTOTUNE)
		ds_lowres = ds_lowres.map(lambda l, h: l, num_parallel_calls=AUTOTUNE)
	__evaluate(ds_lowres, eval_out_path, filenames)

if __name__ == '__main__':
	tl.global_flag['mode'] = args.mode

	if tl.global_flag['mode'] == 'srgan':
		train()
	elif tl.global_flag['mode'] == 'evaluate':
		evaluate()
	elif tl.global_flag['mode'] == 'other':
		other()
	else:
		raise Exception("Unknow --mode")