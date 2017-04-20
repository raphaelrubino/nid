#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import absolute_import
import numpy as np

np.random.seed( 1337 )

import data_utils
from qe_binid_sent import Bilingual_neural_information_density_sentence

import sys

if __name__ == '__main__':

	if len( sys.argv ) != 14:
		print( "\nUsage: ", sys.argv[ 0 ], "<src sentence> <src sentence embedding size> <src vocabulary> <trg sentence> <trg sentence embedding size> <trg ngram> <trg label> <trg vocabulary> <trg context embedding size> <dropout> <batch size> <epochs> <output model>\n" )
		exit()

	src_sentences, src_sentence_embedding, src_vocab, trg_sentences, sentence_embedding, trg_context, trg_target, trg_vocab, context_embedding, dropout, batch, epoch, out_model = sys.argv[ 1: ]

	src_sentence_embedding = np.int( src_sentence_embedding )
	trg_sentence_embedding = np.int( sentence_embedding )
	context_embedding = np.int( context_embedding )
	dropout = np.float( dropout )
	batch = np.int( batch )
	epoch = np.int( epoch )

	print( "Loading vocabulary" )
	trg_vocab, trg_max_features = data_utils.load_vocab( trg_vocab )
	src_vocab, src_max_features = data_utils.load_vocab( src_vocab )
	print( "Loading sentences" )
	src_sentences, src_sentence_max_length = data_utils.load_corpus( src_sentences )
	trg_sentences, trg_sentence_max_length = data_utils.load_corpus( trg_sentences )
	print( "Loading contexts" )
	trg_context = data_utils.load_context( trg_context )
	print( "Loading targets" )
	trg_target = data_utils.load_target( trg_target )

	context_max_length = trg_context.shape[ 1 ]
	validation_size = 0.25

	print( "Data loaded" )
	qe_binid_sent = Bilingual_neural_information_density_sentence( src_sentences, src_sentence_max_length, src_max_features, trg_sentences, trg_sentence_max_length, trg_context, trg_target, trg_max_features, context_max_length, batch, validation_size )
	print( "Data prepared" )
	print( "Training" )
	qe_binid_sent.train( src_sentence_embedding, trg_sentence_embedding, context_embedding, dropout, epoch, out_model )
