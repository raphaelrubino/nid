#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
import numpy as np
np.random.seed(1337)

def load_context( infile ):
	context = []
	with open( infile, mode = "r" ) as f:
		for line in f:
			tmp_line = [ np.int( item ) for item in line.strip().split( " " ) ]
			tmp_line = np.asarray( tmp_line )
			context.append( tmp_line )
	context = np.asarray( context )
	return context

def load_target( infile ): #, max_features ):
	target = []
	with open( infile, mode = "r" ) as f:
		for line in f:
			tmp_target = np.int( line )
			target.append( tmp_target )
	target = np.asarray( target )
	#target = np_utils.to_categorical( target, num_classes = max_features )
	return target

def load_vocab( infile ):
	vocab = {}
	max_features = -1
	with open( infile, mode = "r" ) as f:
		for line in f:
			tmp_line = line.strip().split( " " )
			vocab[ tmp_line[ 0 ] ] = np.int( tmp_line[ 1 ] )
			if np.int( tmp_line[ 1 ] ) > max_features:
				max_features = np.int( tmp_line[ 1 ] )
	max_features = max_features + 1
	return vocab, max_features
