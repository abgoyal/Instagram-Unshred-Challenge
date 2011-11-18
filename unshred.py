from PIL import Image
import numpy
from fractions import gcd
from functools import reduce
import random
import sys

# file meta data
__version__ = "1.0"
__author__ = "abgoyal"
__license__ = "apl"


## some utilities

#sort a 2d list by a given column #
rev_sortbycolumn = lambda l,c: sorted(l, key = lambda x: x[c], reverse=True)
sortbycolumn = lambda l,c: sorted(l, key = lambda x: x[c])


# vector gcd
vgcd = lambda v: reduce(gcd,v)


## algo implementations


# distance metric - this it the key
def distance(v1, v2): # fractional difference in pixel values
	return numpy.sum(((v1-v2)/(v1+v2+1))**2)


# simple cluster detection
def cluster1d(values):
	z = sortbycolumn(values, 1)
	part = [ z.pop()[0]]
	while vgcd(part)>1:
		part.append(z.pop()[0])

	return vgcd(part[:-1])

# estimate the strip width
def stripwidth_algo(im):

	#extract image pixel data as a numpy array
	image_data = numpy.asarray(im.convert("L").getdata(), 'float')
	image_width = im.size[0]
	image_height = im.size[1]

	image_column = lambda i: (image_data[i::image_width])

	# calculate correlation coefs between each pair of adjacent columns in the image
	dists = [ [i+1, distance(image_column(i), image_column(i+1))]  for i in xrange(0,image_width-1) ]

	return cluster1d(dists)


# unshred an image given strip width
def unshred_algo(im,strip_width):
	im_width= im.size[0]
	im_height = im.size[1]
	n = im_width/strip_width # number of strips

	# extract grayscale pixel data from image into a numpy.array
	image_data = numpy.asarray(im.convert("L").getdata(), 'float')

	# extracts all pixels from a specific column
	image_column = lambda i: (image_data[i::im_width])

	# extract the left and right edges of each strip as columns
	left_edges = []
	right_edges = []
	for i in xrange(0,n):
		left_edges.append(image_column(i*strip_width))
		right_edges.append(image_column(i*strip_width + strip_width -1))



	# calculate the distance metric between each left-edge<>right-edge pair
	dists = []
	for left_edge_index,left_edge in enumerate(left_edges):
		for right_edge_index,right_edge in enumerate(right_edges):
			if not left_edge_index == right_edge_index: # a strip cannot be next to itself
				dists.append([left_edge_index, right_edge_index, distance(left_edge,right_edge) ])



	sorted_dists = rev_sortbycolumn(dists,2)


	# select the pairs with the least distance between (strip_left_edge, strip_right_edge) pixels
	assigned_left_edges  = []
	assigned_right_edges = {}
	while len(assigned_right_edges)<n:
		left_edge_index, right_edge_index, dist = sorted_dists.pop()
		if (right_edge_index not in assigned_right_edges) and (left_edge_index not in assigned_left_edges):
			assigned_right_edges[right_edge_index] = left_edge_index
			assigned_left_edges.append(left_edge_index)


	# last strip assigned is most probably the left edge
	s = assigned_left_edges[-1]
	order = [ s ]
	while s in assigned_right_edges and len(order)<n:
		order.append(assigned_right_edges[s])
		s = assigned_right_edges[s]

	return order

def strip_copy(im_original, im_copy, w, a, b):
	h = im_original.size[1]
	a_tuple= tuple([a*w, 0, (a+1)*w, h])
	b_tuple= tuple([b*w, 0, (b+1)*w, h])

	im_copy.paste( im_original.crop(a_tuple) , box=b_tuple )

def unshred(im,strip_width=None):
	im2 = im.copy()

	# if strip width not given, try to calculate it
	if not strip_width:
		strip_width = stripwidth_algo(im)

	order = unshred_algo(im,strip_width)
	for i,j in enumerate(order):
		strip_copy(im, im2, strip_width, j, i)

	return im2

def shred(im,m):
	im_width= im.size[0]
	im_height = im.size[1]
	im2 = im.copy()
	n = im_width/m
	strips = range(0,n)
	random.shuffle(strips)
	for f,t in enumerate(strips):
		strip_copy(im, im2, m, f, t)

	return im2


def demo():
	im_shreded = Image.open("TokyoPanoramaShredded.png")
	im_fixed1   = unshred(im_shreded)
	im_fixed1.show()
	im_fixed2   = unshred(im_shreded,32)
	im_fixed2.show()

	im_reshredded = shred(im_fixed2, 16)
	im_fixed1   = unshred(im_shreded)
	im_fixed1.show()
	im_fixed2   = unshred(im_shreded,16)
	im_fixed2.show()


def process_command(args):
	cmd = args[0]
	if cmd == "shred":
		w = int(args[1])
		ifname = args[2]
		ofname = args[3]
		im = Image.open(ifname)
		im_shredded = shred(im, w)
		im_shredded.save(ofname)
		return
	if cmd == "unshred":
		w = int(args[1])
		ifname = args[2]
		ofname = args[3]
		im = Image.open(ifname)

		if w==0: 
			im_fixed = unshred(im)
		else:
			im_fixed = unshred(im,w)

		im_fixed.save(ofname)
		return

	print "Unknown command"
	return


if __name__ == "__main__":
	if len(sys.argv) > 1:
		process_command(sys.argv[1:])
	else:
		demo()

