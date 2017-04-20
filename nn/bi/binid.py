#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import absolute_import
import numpy as np
np.random.seed(1337)

import theano

from keras.models import Model
from keras.layers import Input, Embedding, Dropout, Dense, GlobalAveragePooling1D, concatenate, Flatten
from keras.preprocessing import sequence
from keras.utils import np_utils

from sklearn.utils import shuffle

import time
import sys


class Bilingual_neural_information_density():

	def __init__( self, src_corpus, src_max_features, src_max_length, trg_context, trg_target, trg_max_features, trg_max_length, batch_size, valid_size ):
		self.src_corpus = src_corpus
		self.src_max_features = src_max_features
		self.src_max_length = src_max_length
		self.trg_context = trg_context
		self.trg_target = trg_target
		self.batch = batch_size
		self.valid_size = valid_size
		trg_nb_instances = len( self.trg_context )
		print( "Instances: {0}".format( trg_nb_instances ), flush = True )
		self.valid_size = np.int( valid_size * trg_nb_instances )
		self.train_size = np.int( trg_nb_instances - self.valid_size )
		self.trg_max_features = trg_max_features
		self.trg_max_length = trg_max_length
		self.model = -1
		self.build_train_valid()
		self.build_batched_data()

	def build_train_valid( self ):
		self.src_corpus, self.trg_context, self.trg_target = shuffle( self.src_corpus, self.trg_context, self.trg_target )
		self.src_corpus_train = self.src_corpus[ :self.train_size ]
		self.src_corpus_valid = self.src_corpus[ self.train_size: ]
		self.trg_context_train = self.trg_context[ :self.train_size ]
		self.trg_target_train = self.trg_target[ :self.train_size ]
		self.trg_context_valid = self.trg_context[ self.train_size: ]
		self.trg_target_valid = self.trg_target[ self.train_size: ]

	def build_batched_data( self ):
		self.batch_src_corpus_train = np.asarray( [ np.asarray( self.src_corpus_train[ x : x + self.batch ] ) for x in range( 0, len( self.src_corpus_train ), self.batch ) ] )
		self.batch_src_corpus_valid = np.asarray( [ np.asarray( self.src_corpus_valid[ x : x + self.batch ] ) for x in range( 0, len( self.src_corpus_valid ), self.batch ) ] )
		self.batch_trg_context_train = np.asarray( [ np.asarray( self.trg_context_train[ x : x + self.batch ] ) for x in range( 0, len( self.trg_context_train ), self.batch ) ] )
		self.batch_trg_target_train = np.asarray( [ np.asarray( self.trg_target_train[ x : x + self.batch ] ) for x in range( 0, len( self.trg_target_train ), self.batch ) ] )
		self.batch_trg_context_valid = np.asarray( [ np.asarray( self.trg_context_valid[ x : x + self.batch ] ) for x in range( 0, len( self.trg_context_valid ), self.batch ) ] )
		self.batch_trg_target_valid = np.asarray( [ np.asarray( self.trg_target_valid[ x : x + self.batch ] ) for x in range( 0, len( self.trg_target_valid ), self.batch ) ] )

	def get_model( self ):
		return self.model

	def save_architecture( self, filename ):
		with open( filename + '.architecture.json', "w" ) as f:
			f.write( self.model.to_json() )

	def save_weights( self, filename ):
		self.model.save_weights( filename + '.weights.h5', overwrite = True )

	def get_default_model( self, src_embedding, trg_embedding, dropout ):
		input_src_sentence = Input( shape = ( self.src_max_length, ), dtype = 'int32', name = 'input_src_sentence' )
		emb_src_sentence = Embedding( input_dim = self.src_max_features, output_dim = src_embedding, input_length = self.src_max_length )( input_src_sentence )
		pool_src_sentence = GlobalAveragePooling1D()( emb_src_sentence )
		drop_src_sentence = Dropout( dropout )( pool_src_sentence )

		input_trg_context = Input( shape = ( self.trg_max_length, ), dtype = 'int32', name = 'input_trg_context' )
		emb_trg_context = Embedding( input_dim = self.trg_max_features, output_dim = trg_embedding, input_length = self.trg_max_length )( input_trg_context )
		flat_trg_context = Flatten()( emb_trg_context )
		drop_trg_context = Dropout( dropout )( flat_trg_context )

		concat = concatenate( [ drop_src_sentence, drop_trg_context ] )
		output = Dense( self.trg_max_features, activation = 'softmax', name = 'output' )( concat )
		
		model = Model( inputs = [ input_src_sentence, input_trg_context ], outputs = output )
		return model

def train_model( self ):
	train_loss = 0.0
	train_acc = 0.0
	for j in range( self.batch_trg_target_train.shape[ 0 ] ):
		loss, metrics = self.model.train_on_batch( \
			{ 'input_src_sentence': sequence.pad_sequences( self.batch_src_corpus_train[ j ], \
				maxlen = self.src_max_length, dtype = 'int32', padding = 'post', value = 2 ) , \
			'input_trg_context': self.batch_trg_context_train[ j ] }, \
			{ 'output': np_utils.to_categorical( self.batch_trg_target_train[ j ], \
				num_classes = self.trg_max_features ) } )
		train_loss += loss
		train_acc += metrics
	train_loss /= j
	train_acc /= j
	return train_loss, train_acc

	def valid_model( self ):
		valid_loss = 0.0
		valid_acc = 0.0
		for k in range( self.batch_trg_target_valid.shape[ 0 ] ):
			loss, metrics = self.model.test_on_batch( \
				{ 'input_src_sentence': sequence.pad_sequences( self.batch_src_corpus_valid[ k ], \
					maxlen = self.src_max_length, dtype = 'int32', padding = 'post', value = 2 ), \
				'input_trg_context': self.batch_trg_context_valid[ k ] }, \
				{ 'output': np_utils.to_categorical( self.batch_trg_target_valid[ k ], \
					num_classes = self.trg_max_features ) } )
			valid_loss += loss
			valid_acc += metrics
		valid_loss /= k
		valid_acc /= k
		return valid_loss, valid_acc

	def train( self, src_embedding_size, trg_embedding_size, dropout, nb_epochs, out_model ):
		self.model = self.get_default_model( src_embedding_size, trg_embedding_size, dropout )
		self.model.compile( optimizer = 'RMSprop', loss = 'categorical_crossentropy', metrics = [ 'accuracy' ] )
		best_acc = np.float( 0.0 )
		best_loss = np.float( 999.9 )
		for i in range( nb_epochs ):
			time_start = time.time()
			print( "Epoch {0}".format( i + 1 ), flush = True )
			train_loss, train_acc = train_model()
			valid_loss, valid_acc = valid_model()
			if best_acc < valid_acc:
				best_acc = valid_acc
				self.save_weights( "{0}.acc_{1}".format( out_model, np.round( best_acc, 3 ) ) )
				self.save_architecture( "{0}.acc_{1}".format( out_model, np.round( best_acc, 3 ) ) )
			print( "train loss {0} -- acc: {1} ---- valid loss: {2} -- acc: {3}".format( train_loss, train_acc, valid_loss, valid_acc ), flush = True )
			time_elapsed = time.time() - time_start
			print( "{0} seconds".format( time_elapsed ), flush = True )
