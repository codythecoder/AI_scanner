VERSION = '0.0.1'


color_channels = 3
# how many pixels large should the comparison images be
#NOTE must be odd
scan_size = 63
nn_outputs = 1


input_shape = (-1, scan_size, scan_size, color_channels)
input_size = color_channels*(scan_size**2)
