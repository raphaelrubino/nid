#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
from nltk.util import ngrams
import numpy as np

def extract_vocabulary_and_ngrams( in_corpus, n_order ):
	corpus_contexts = []
	corpus_targets = []
	vocab = {}
	vocab[ "unk" ] = 0
	vocab[ "<s>" ] = 1
	vocab[ "</s>" ] = 2
	count = len( vocab )
	with open( in_corpus, mode = "r" ) as corpus:
		for line in corpus:
			line = line.strip()
			line = "<s> " * ( n_order - 1 ) + line + " </s>" * ( n_order - 1 )
			tokens = line.split( " " )
			numberized_tokens = []
			for token in tokens:
				if token not in vocab:
					vocab[ token ] = count
					count += 1
				numberized_tokens.append( vocab[ token ] )
			line_ngrams = ngrams( numberized_tokens, n_order * 2 - 1 )
			for ngram in line_ngrams:
				left_context = " ".join( str( item ) for item in ngram[ 0 : n_order - 1 ] )
				right_context = " ".join( str( item ) for item in ngram[ n_order : len( ngram ) ] )
				target = str( ngram[ n_order - 1 ] )
				corpus_contexts.append( left_context + " " + right_context )
				corpus_targets.append( target )
	vocab = [ "{0} {1}".format( item, vocab[ item ] ) for item in vocab ]
	return vocab, corpus_contexts, corpus_targets
	
def write_file( towrite, out_file ):
	with open( out_file, mode = "w" ) as out:
		out.write( "\n".join( towrite ) )

if __name__ == '__main__':

	if len( sys.argv ) != 4:
		print( "\nUsage: ", sys.argv[ 0 ], "<input corpus> <ngram order> <output prefix>\n" )
		exit()

	in_corpus, n_order, out_prefix = sys.argv[ 1: ]

	n_order = np.int( n_order )

	out_context = "{0}.context".format( out_prefix )
	out_target = "{0}.target".format( out_prefix )
	out_vocab = "{0}.vocab".format( out_prefix )

	vocab, contexts, targets = extract_vocabulary_and_ngrams( in_corpus, n_order )
	write_file( contexts, out_context )
	write_file( targets, out_target )
	write_file( vocab, out_vocab )
