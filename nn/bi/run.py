#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import absolute_import
import numpy as np

np.random.seed( 1337 )

import data_utils
from binnid import BilingualNNID

import sys

if __name__ == '__main__':

	if len( sys.argv ) != 12:
		print( "\nUsage: ", sys.argv[ 0 ], "<src sentence> <src vocabulary> <src embedding size> <tgr context> <trg target> <trg vocabulary> <trg embedding size> <dropout> <batch size> <epochs> <output model>\n" )
		exit()

	src_sentences, src_vocab, src_embedding, trg_context, trg_target, trg_vocab, trg_embedding, dropout, batch, epoch, out_model = sys.argv[ 1: ]

	src_embedding = np.int( src_embedding )
	trg_embedding = np.int( trg_embedding )
	dropout = np.float( dropout )
	batch = np.int( batch )
	epoch = np.int( epoch )

	print( "Loading vocabulary" )
	src_vocab, src_max_features = data_utils.load_vocab( src_vocab )
	trg_vocab, trg_max_features = data_utils.load_vocab( trg_vocab )
	print( "Loading source sentences" )
	src_sentences, src_max_length = data_utils.load_corpus( src_sentences )
	print( "Loading contexts" )
	trg_context = data_utils.load_context( trg_context )
	print( "Loading targets" )
	trg_target = data_utils.load_target( trg_target )

	trg_max_length = trg_context.shape[ 1 ]
	validation_size = 0.25

	print( "Data loaded" )
	ft = My_fast_text( src_sentences, src_max_features, src_max_features, src_max_length, trg_context, trg_target, trg_max_features, trg_max_length, batch, validation_size )
	print( "Data prepared" )
	print( "Training" )
	ft.train( src_embedding, trg_embedding, dropout, epoch, out_model )
