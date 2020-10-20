# Script for visualization of augmentation parameters
from matplotlib import pyplot
from PIL import Image, ImageStat
import numpy as np
from tensorflow.keras.preprocessing.image import random_brightness, random_shift, random_rotation, random_zoom

def brightness(im):
   im = im.convert('L')
   stat = ImageStat.Stat(im)
   return stat.median[0], stat.mean[0]

# dark1 = Image.open('/home/toertei/PycharmProjects/traffic-signs/traffic_signs_train/belgium/7/01811_00001.ppm')
# bright1 = Image.open('/home/toertei/PycharmProjects/traffic-signs/traffic_signs_train/belgium/7/02281_00000.ppm')
# normal1 = Image.open('/home/toertei/PycharmProjects/traffic-signs/traffic_signs_train/belgium/7/01992_00000.ppm')
# print(f"{brightness(dark1)} {brightness(bright1)} {brightness(normal1)}")
#
# dark2 = Image.open('/home/toertei/PycharmProjects/traffic-signs/traffic_signs_train/belgium/20/01537_00000.ppm')
# bright2 = Image.open('/home/toertei/PycharmProjects/traffic-signs/traffic_signs_train/belgium/20/02347_00000.ppm')
# normal2 = Image.open('/home/toertei/PycharmProjects/traffic-signs/traffic_signs_train/belgium/20/01700_00000.ppm')
# print(f"{brightness(dark2)} {brightness(bright2)} {brightness(normal2)}")
#
# dark3 = Image.open('/home/toertei/PycharmProjects/traffic-signs/traffic_signs_train/belgium/18/01799_00000.ppm')
# bright3 = Image.open('/home/toertei/PycharmProjects/traffic-signs/traffic_signs_train/belgium/20/02347_00000.ppm')
# normal3 = Image.open('/home/toertei/PycharmProjects/traffic-signs/traffic_signs_train/belgium/18/01160_00000.ppm')
# print(f"{brightness(dark3)} {brightness(bright3)} {brightness(normal3)}")
#
bright4 = Image.open('/home/toertei/PycharmProjects/traffic-signs/traffic_signs_train/germany/4/00009_00029.ppm')
print(f"{brightness(bright4)} ")

num_channels = 1
brightness = lambda im : random_brightness(im, (0.2, 3))
shift = lambda im : random_shift(im, wrg=0.2, hrg=0.2, row_axis=0, col_axis=1, channel_axis=2)
rotation = lambda im : random_rotation(im, 20, row_axis=0, col_axis=1, channel_axis=2)
zoom_in = lambda im : random_zoom(im, (0.85, 0.85), row_axis=0, col_axis=1, channel_axis=2)
zoom_out = lambda im : random_zoom(im, (1.15, 1.15), row_axis=0, col_axis=1, channel_axis=2)

aug_tx_arr = [brightness, shift, rotation, zoom_in, zoom_out]

# im = Image.open('/home/toertei/PycharmProjects/traffic-signs/traffic_signs_train/germany/1/00060_00029.ppm')
im = Image.open('/home/toertei/PycharmProjects/traffic-signs/traffic_signs_train/germany/4/00006_00029.ppm')
dim = (70, 70)
im = im.resize(dim)
im = im.convert(mode='L')
# im.show()
im = np.array(im)
im = np.expand_dims(im, axis=-1)
print(im.shape)
for i in range(9):
    # tmp = random_brightness(im, (0.2, 3))
    if brightness in aug_tx_arr:
        tmp = aug_tx_arr[0](im)
    tmp1 = aug_tx_arr[1](im)
    tmp2 = aug_tx_arr[2](im)
    tmp3 = aug_tx_arr[3](im)
    tmp4 = aug_tx_arr[4](im)

    tmp = tmp.astype('uint8')
    tmp1 = tmp1.astype('uint8')
    tmp2 = tmp2.astype('uint8')
    tmp3 = tmp3.astype('uint8')
    tmp4 = tmp4.astype('uint8')
    # plot raw pixel data
    pyplot.subplot(9, 5, 1 + i*5)
    pyplot.imshow(tmp)
    pyplot.subplot(9, 5, 2 + i*5)
    pyplot.imshow(tmp1)
    pyplot.subplot(9, 5, 3 + i*5)
    pyplot.imshow(tmp2)
    pyplot.subplot(9, 5, 4 + i*5)
    pyplot.imshow(tmp3)
    pyplot.subplot(9, 5, 5 + i*5)
    pyplot.imshow(tmp4)
# show the figure
pyplot.show()


