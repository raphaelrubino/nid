#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import absolute_import
import numpy as np
np.random.seed(1337)

import theano

from keras.models import Sequential
from keras.layers import Embedding, Dropout, Dense, GlobalAveragePooling1D, LSTM, Flatten
from keras.callbacks import ModelCheckpoint, EarlyStopping
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
		print( "Instances: {0}".format( nb_instances ) )
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
		#model.add( GlobalAveragePooling1D() ) 
		#model.add( LSTM( int( embedding / 2 ), activation = "sigmoid" ) )
		#model.add( Dropout( dropout ) )
		model.add( Dense( self.max_features, activation = 'softmax' ) )
		return model

	def train( self, embedding_size, dropout, nb_epochs, out_model ):
		model = self.get_default_model( embedding_size, dropout )
		model.compile( optimizer = 'RMSprop', loss = 'categorical_crossentropy', metrics = [ 'accuracy' ] )
		best_acc = np.float( 0.0 )
		best_loss = np.float( 999.9 )
		nb_batch_train = self.batch_target_train.shape[ 0 ]
		nb_batch_valid = self.batch_target_valid.shape[ 0 ]
		batch_context_train = self.batch_context_train
		batch_target_train = self.batch_target_train
		batch_context_valid = self.batch_context_valid
		batch_target_valid = self.batch_target_valid
		for i in range( nb_epochs ):
			time_start = time.time()
			print( "Epoch {0}".format( i + 1 ) )
			train_loss = 0.0
			train_acc = 0.0
			for j in range( nb_batch_train ):
				loss, metrics = model.train_on_batch( batch_context_train[ j ], \
					np_utils.to_categorical( batch_target_train[ j ], num_classes = self.max_features ) )
				train_loss += loss
				train_acc += metrics
			train_loss /= j
			train_acc /= j
			avg_loss = 0
			avg_acc = 0
			for k in range( nb_batch_valid ):
				valid_loss, valid_metrics = model.test_on_batch( batch_context_valid[ k ], \
					np_utils.to_categorical( batch_target_valid[ k ], num_classes = self.max_features ) )
				avg_loss += valid_loss
				avg_acc += valid_metrics
			avg_loss /= nb_batch_valid
			avg_acc /= nb_batch_valid
			if best_acc < avg_acc:
				best_acc = avg_acc
				self.model = model
				self.save_weights( "{0}.best_acc".format( out_model ) )
				self.save_architecture( "{0}.best_acc".format( out_model ) )
			#predict = self.model.predict_on_batch( np.asarray( [ [ 24, 100, 71, 30, 102, 103, 104, 34 ] ] ) )
			#predict = np.where( predict == np.max( predict[ 0 ] ) )
			#print( predict )
			#predict = self.model.predict_on_batch( np.asarray( [ [ 107, 9097, 2030, 5513, 244, 24, 5164, 6471 ] ] ) )
			#predict = np.where( predict == np.max( predict[ 0 ] ) )
			#print( predict )

			print( "train loss {0} -- acc: {1} ---- valid loss: {2} -- acc: {3}".format( train_loss, train_acc, avg_loss, avg_acc ) )
			time_elapsed = time.time() - time_start
			print( "{0} seconds".format( time_elapsed ) )
#		checkpointer = ModelCheckpoint( filepath = out_model + ".hdf5", monitor = "val_acc", verbose = 0, save_best_only = True, mode = "max" )
#		earlystopping = EarlyStopping( monitor = 'val_acc', patience = 20, verbose = 0, mode = "max" )
#		history = model.fit( self.context, self.target,
#			batch_size = batch_size,
#			epochs = nb_epochs,
#			shuffle = True,
#			validation_split = 0.25,
#			callbacks = [ checkpointer, earlystopping ],
#			verbose = 1
#		)
#		model.load_weights( out_model + ".hdf5" )
