#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
from nltk.util import ngrams
import numpy as np

def format_line( line, padding = 1 ):
	line = line.strip()
	line = "<s> " * padding + line + " </s>" * padding
	line = line.split( " " )
	return line

def init_vocab():
	vocab = {
		"unk": 0,
		"<s>": 1,
		"</s>": 2,
	}
	return vocab

def extract_vocabulary_and_sentences( in_corpus ):
	corpus = []
	vocab = init_vocab()
	count = len( vocab )
	with open( in_corpus, mode = "r" ) as c:
		for line in c:
			line = format_line( line, 1 )
			numberized_tokens = []
			for token in line:
				if token not in vocab: 
					vocab[ token ] = count
					count += 1
				numberized_tokens.append( str( vocab[ token ] ) )
			corpus.append( " ".join( numberized_tokens ) )
	vocab = [ "{0} {1}".format( item, vocab[ item ] ) for item in vocab ]
	return vocab, corpus

def extract_vocabulary_and_ngrams( in_corpus, n_order ):
	corpus_contexts = []
	corpus_targets = []
	vocab = init_vocab()
	count = len( vocab )
	sentence_idx = []
	count_sentence = -1
	with open( in_corpus, mode = "r" ) as corpus:
		for line in corpus:
			count_sentence += 1
			tokens = format_line( line, ( n_order - 1 ) )
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
				sentence_idx.append( count_sentence )
	vocab = [ "{0} {1}".format( item, vocab[ item ] ) for item in vocab ]
	return vocab, corpus_contexts, corpus_targets, sentence_idx
	
def multiply_sentences( sentences, idx ):
	multi_sentences = []
	for i in idx:
		multi_sentences.append( sentences[ i ] )
	return multi_sentences

def write_file( towrite, out_file ):
	with open( out_file, mode = "w" ) as out:
		out.write( "\n".join( towrite ) )

if __name__ == '__main__':

	if len( sys.argv ) != 6:
		print( "\nUsage: ", sys.argv[ 0 ], "<input src corpus> <input trg corpus> <trg ngram order> <src output prefix> <trg output prefix>\n" )
		exit()

	in_src_corpus, in_trg_corpus, trg_n_order, out_src_prefix, out_trg_prefix = sys.argv[ 1: ]

	out_src_sentences = "{0}.sentences".format( out_src_prefix )
	out_src_vocab = "{0}.vocab".format( out_src_prefix )
	src_vocab, src_sentences = extract_vocabulary_and_sentences( in_src_corpus )

	trg_n_order = np.int( trg_n_order )
	out_trg_context = "{0}.context".format( out_trg_prefix )
	out_trg_target = "{0}.target".format( out_trg_prefix )
	out_trg_vocab = "{0}.vocab".format( out_trg_prefix )
	trg_vocab, trg_contexts, trg_targets, trg_sentence_idx = extract_vocabulary_and_ngrams( in_trg_corpus, trg_n_order )

	src_sentences = multiply_sentences( src_sentences, trg_sentence_idx )
	
	write_file( src_sentences, out_src_sentences )
	write_file( src_vocab, out_src_vocab )
	write_file( trg_contexts, out_trg_context )
	write_file( trg_targets, out_trg_target )
	write_file( trg_vocab, out_trg_vocab )
