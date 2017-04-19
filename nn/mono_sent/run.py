#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import absolute_import
import numpy as np

np.random.seed( 1337 )

import data_utils
from nid_sent import Neural_information_density_sentence

import sys

if __name__ == '__main__':

	if len( sys.argv ) != 11:
		print( "\nUsage: ", sys.argv[ 0 ], "<trg sentence> <trg sentence embedding size> <trg context> <trg target> <trg vocabulary> <trg context embedding size> <dropout> <batch size> <epochs> <output model>\n" )
		exit()

	trg_sentences, sentence_embedding, trg_context, trg_target, trg_vocab, context_embedding, dropout, batch, epoch, out_model = sys.argv[ 1: ]

	sentence_embedding = np.int( sentence_embedding )
	context_embedding = np.int( context_embedding )
	dropout = np.float( dropout )
	batch = np.int( batch )
	epoch = np.int( epoch )

	print( "Loading vocabulary" )
	trg_vocab, trg_max_features = data_utils.load_vocab( trg_vocab )
	print( "Loading sentences" )
	trg_sentences, sentence_max_length = data_utils.load_corpus( trg_sentences )
	print( "Loading contexts" )
	trg_context = data_utils.load_context( trg_context )
	print( "Loading targets" )
	trg_target = data_utils.load_target( trg_target )

	context_max_length = trg_context.shape[ 1 ]
	validation_size = 0.25

	print( "Data loaded" )
	nid_sent = Neural_information_density_sentence( trg_sentences, sentence_max_length, trg_context, trg_target, trg_max_features, context_max_length, batch, validation_size )
	print( "Data prepared" )
	print( "Training" )
	nid_sent.train( sentence_embedding, context_embedding, dropout, epoch, out_model )
