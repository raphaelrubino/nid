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
	return vocab, corpus

def extract_vocabulary_ngrams_and_sentences( in_corpus, n_order ):
	corpus = []
	corpus_ngrams = []
	sentence_idx = []
	vocab = init_vocab()
	count = len( vocab )
	count_sentence = -1
	with open( in_corpus, mode = "r" ) as c:
		for line in c:
			count_sentence += 1
			sentline = format_line( line, 1 )
			ngramline = format_line( line, n_order - 1 )
			numberized_tokens = []
			numberized_ngram_tokens = []
			for token in sentline:
				if token not in vocab:
					vocab[ token ] = count
					count += 1
				numberized_tokens.append( str( vocab[ token ] ) )
			for token in ngramline:
				numberized_ngram_tokens.append( str( vocab[ token ] ) )
			line_ngrams = ngrams( numberized_ngram_tokens, n_order * 2 - 1 )
			for ngram in line_ngrams:
				sentence_idx.append( count_sentence )
				corpus_ngrams.append( " ".join( str( item ) for item in ngram ) )
			corpus.append( " ".join( numberized_tokens ) )
	return vocab, corpus, corpus_ngrams, sentence_idx

def multiply_sentences( sentences, idx ):
	multi_sentences = []
	for i in idx:
		multi_sentences.append( sentences[ i ] )
	return multi_sentences

def format_labels( labels ):
	vocab = {}
	count = 0
	flat_labels = []
	with open( labels, mode = "r" ) as l:
		for line in l:
			line = line.strip().split( " " )
			for token in line:
				if token not in vocab:
					vocab[ token ] = count
					count += 1
				flat_labels.append( str( vocab[ token ] ) )
	return vocab, flat_labels

def write_file( towrite, out_file ):
	with open( out_file, mode = "w" ) as out:
		out.write( "\n".join( towrite ) )

if __name__ == '__main__':

	if len( sys.argv ) != 7:
		print( "\nUsage: ", sys.argv[ 0 ], "<input src corpus> <input trg corpus> <trg ngram order> <trg labels> <src output prefix> <trg output prefix>\n" )
		exit()

	in_src_corpus, in_trg_corpus, trg_n_order, trg_labels, out_src_prefix, out_trg_prefix = sys.argv[ 1: ]

	out_src_sentences = "{0}.sentences".format( out_src_prefix )
	out_src_vocab = "{0}.vocab".format( out_src_prefix )
	src_vocab, src_sentences = extract_vocabulary_and_sentences( in_src_corpus )

	trg_n_order = np.int( trg_n_order )
	out_trg_sentences = "{0}.sentences".format( out_trg_prefix )
	out_trg_context = "{0}.context".format( out_trg_prefix )
	out_trg_labels = "{0}.labels".format( out_trg_prefix )
	out_trg_list_labels = "{0}.list_labels".format( out_trg_prefix )
	out_trg_vocab = "{0}.vocab".format( out_trg_prefix )
	trg_vocab, trg_sentences, trg_context, trg_sentence_idx = extract_vocabulary_ngrams_and_sentences( in_trg_corpus, trg_n_order )

	src_sentences = multiply_sentences( src_sentences, trg_sentence_idx )
	trg_sentences = multiply_sentences( trg_sentences, trg_sentence_idx )
	
	list_labels, trg_labels = format_labels( trg_labels )

	src_vocab = [ "{0} {1}".format( item, src_vocab[ item ] ) for item in src_vocab ]
	trg_vocab = [ "{0} {1}".format( item, trg_vocab[ item ] ) for item in trg_vocab ]
	list_labels = [ "{0} {1}".format( item, list_labels[ item ] ) for item in list_labels ]

	write_file( src_sentences, out_src_sentences )
	write_file( src_vocab, out_src_vocab )
	write_file( trg_sentences, out_trg_sentences )
	write_file( trg_context, out_trg_context )
	write_file( trg_labels, out_trg_labels )
	write_file( list_labels, out_trg_list_labels )
	write_file( trg_vocab, out_trg_vocab )
