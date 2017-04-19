#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import absolute_import
import numpy as np

np.random.seed( 1337 )

import data_utils
from nid import Neural_information_density

import sys

if __name__ == '__main__':

	if len( sys.argv ) != 9:
		print( "\nUsage: ", sys.argv[ 0 ], "<context> <target> <vocabulary> <embedding size> <dropout> <batch size> <epochs> <output model>\n" )
		exit()

	context, target, vocab, embedding, dropout, batch, epoch, out_model = sys.argv[ 1: ]

	embedding = np.int( embedding )
	dropout = np.float( dropout )
	batch = np.int( batch )
	epoch = np.int( epoch )

	print( "Loading vocabulary" )
	vocab, max_features = data_utils.load_vocab( vocab )
	print( "Loading contexts" )
	context = data_utils.load_context( context )
	print( "Loading targets" )
	target = data_utils.load_target( target ) #, max_features )

	max_length = context.shape[ 1 ]
	validation_size = 0.25

	print( "Data loaded" )
	nid = Neural_information_density( context, target, max_features, max_length, batch, validation_size )
	print( "Data prepared" )
	print( "Training" )
	nid.train( embedding, dropout, epoch, out_model )
