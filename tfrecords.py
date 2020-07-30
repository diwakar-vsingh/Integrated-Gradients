#!/usr/bin/env python3

import os
import tensorflow as tf
from PIL import Image
import glob

output_dir = "Images/"


def convert_to_tfexample(img_path):
	try:
		with open(img_path, 'rb') as f:
			content = f.read()
		with Image.open(img_path) as im:
			im.load()
			if im.format != 'JPEG':
				print('Wrong image format, path {}, format {}'.format(img_path, im.format))
			assert (im.format == 'JPEG')
			filename = os.path.basename(img_path)
			example = tf.train.Example(
				features=tf.train.Features(
					feature={
						'image/encoded': tf.train.Feature(bytes_list=tf.train.BytesList(value=[content])),
						'image/format': tf.train.Feature(bytes_list=tf.train.BytesList(value=['JPEG'.encode()])),
						'image/width': tf.train.Feature(int64_list=tf.train.Int64List(value=[im.width])),
						'image/height': tf.train.Feature(int64_list=tf.train.Int64List(value=[im.height])),
						'image/filename': tf.train.Feature(bytes_list=tf.train.BytesList(value=[filename.encode()]))
					}))
		return example
	except Exception as e:
		print(e)
		return None


def main():
	test_images = sorted(glob.glob(os.path.join(output_dir, '*.jpg')))
	writer = tf.io.TFRecordWriter('test.tfrecord')
	
	for file in test_images:
		example = convert_to_tfexample(file)
		writer.write(example.SerializeToString())
	print('Finished converting TFRecords for Images')


if __name__ == '__main__':
	main()