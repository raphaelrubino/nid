#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import absolute_import
import numpy as np
np.random.seed(1337)

from keras.models import Sequential
from keras.layers import Embedding, Dropout, Dense, Flatten
from keras.utils import np_utils

from sklearn.utils import shuffle

import time
import sys


class Neural_information_density():

	def __init__( self, context, target, max_features, max_length, batch_size, valid_size ):
		self.context = context
		self.target = target
		self.batch = batch_size
		self.valid_size = valid_size
		nb_instances = len( self.context )
		print( "Instances: {0}".format( nb_instances ), flush = True )
		self.valid_size = np.int( valid_size * nb_instances )
		self.train_size = np.int( nb_instances - self.valid_size )
		self.max_features = max_features
		self.max_length = max_length
		self.model = -1
		self.build_train_valid()
		self.build_batched_data()

	def build_train_valid( self ):
		self.context, self.target = shuffle( self.context, self.target )
		self.context_train = self.context[ :self.train_size ]
		self.target_train = self.target[ :self.train_size ]
		self.context_valid = self.context[ self.train_size: ]
		self.target_valid = self.target[ self.train_size: ]

	def build_batched_data( self ):
		self.batch_context_train = np.asarray( [ np.asarray( self.context_train[ x : x + self.batch ] ) for x in range( 0, len( self.context_train ), self.batch ) ] )
		self.batch_target_train = np.asarray( [ np.asarray( self.target_train[ x : x + self.batch ] ) for x in range( 0, len( self.target_train ), self.batch ) ] )
		self.batch_context_valid = np.asarray( [ np.asarray( self.context_valid[ x : x + self.batch ] ) for x in range( 0, len( self.context_valid ), self.batch ) ] )
		self.batch_target_valid = np.asarray( [ np.asarray( self.target_valid[ x : x + self.batch ] ) for x in range( 0, len( self.target_valid ), self.batch ) ] )

	def get_model( self ):
		return self.model

	def save_architecture( self, filename ):
		with open( filename + '.architecture.json', "w" ) as f:
			f.write( self.model.to_json() )

	def save_weights( self, filename ):
		self.model.save_weights( filename + '.weights.h5', overwrite = True )

	def get_default_model( self, embedding, dropout ):
		model = Sequential()
		model.add( Embedding( self.max_features, embedding, input_length = self.max_length ) )
		model.add( Flatten() )
		model.add( Dropout( dropout ) )
		model.add( Dense( self.max_features, activation = 'softmax' ) )
		return model

	def train_model( self ):
		train_loss = 0.0
		train_acc = 0.0
		for j in range( self.batch_target_train.shape[ 0 ] ):
			loss, metrics = self.model.train_on_batch( self.batch_context_train[ j ], \
				np_utils.to_categorical( self.batch_target_train[ j ], num_classes = self.max_features ) )
			train_loss += loss
			train_acc += metrics
		train_loss /= j
		train_acc /= j
		return train_loss, train_acc

	def valid_model( self ):
		valid_loss = 0.0
		valid_acc = 0.0
		for k in range( self.batch_target_valid.shape[ 0 ] ):
			loss, metrics = self.model.test_on_batch( self.batch_context_valid[ k ], \
				np_utils.to_categorical( self.batch_target_valid[ k ], num_classes = self.max_features ) )
			valid_loss += loss
			valid_acc += metrics
		valid_loss /= k
		valid_acc /= k
		return valid_loss, valid_acc

	def train( self, embedding_size, dropout, nb_epochs, out_model ):
		print( "Building model", flush = True )
		self.model = self.get_default_model( embedding_size, dropout )
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
