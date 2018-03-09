VERSION = '0.0.1'


color_channels = 3
# how many pixels large should the comparison images be
#NOTE must be odd
scan_size = 63
nn_outputs = 1


input_shape = (-1, scan_size, scan_size, color_channels)
input_size = color_channels*(scan_size**2)
#
# CNN_filter_sizes = (7, 5, 3, 3, 3)
# CNN_features = (9, 16, 25, 32, 32)
# CNN_strides = (2, 2, 1, 1, 1)
#
# FCL_sizes = (5000, 1024, 512, 128, nn_outputs)

CNN_filter_sizes = (5, 3, 3)
CNN_features = (9, 16, 25)
CNN_strides = (2, 2, 2)

FCL_sizes = (3136, 128, nn_outputs)
