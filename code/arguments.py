import dynet as dy
import argparse
import logging
import os
import random
from collections import Counter, defaultdict
#import matplotlib.pyplot as plt

def get_arguments():
	# Training settings
	parser = argparse.ArgumentParser(description='Char BiLSTM Baseline')


	# Dataset parameters 
	parser.add_argument('--train-data', type=str, default='data/dataset.train', 
						help='path to the train data file')
	parser.add_argument('--dev-data', type=str, default='data/dataset.dev', 
						help='path to the dev data file')
	parser.add_argument('--test-data', type=str, default='data/dataset.test', 
						help='path to the test data file')

	# training hyper parameters
	parser.add_argument('--word-dim', type=int, default=50, 
						help='char embedding dimension (default: 50)')
	parser.add_argument('--sent-dim', type=int, default=100, 
						help='input dimension (default: 100)')
	parser.add_argument('--word-hidden-dim', type=int, default=100, 
						help='hidden state dim (default: 100)')
	parser.add_argument('--sent-hidden-dim', type=int, default=100, 
						help='char ltm hidden state dim (default: 50)')
	parser.add_argument('--word-num-of-layers', type=int, default=1, 
						help='char lstm num of layers (default: 1)')
	parser.add_argument('--sent-num-of-layers', type=int, default=1, 
						help='number of layers in word lstm (default: 1)')
	parser.add_argument('--sent-pos', type=int, default=10, 
						help='max number of sentences (default: 10)')
	parser.add_argument('--sent-buckets', type=int, default=2, 
						help='number of buckets sentence (default: 2')
	parser.add_argument('--pos-dim', type=int, default=50, 
						help='pos embedding dimension (default: 50)')
	parser.add_argument('--doc-dim', type=int, default=50, 
						help='document embedding dimension (default: 50)')
	parser.add_argument('--base-dim', type=int, default=1, 
						help='base probability score dimension (default: 1)')
	parser.add_argument('--threshold', type=float, default=0.5, 
						help='threshold for choosing sentences (default: 0.5)')

	# training additional parameters
	parser.add_argument('--batch-size', type=int, default=64, 
						help='input batch size for training (default: 64)')
	parser.add_argument('--vocab-size', type=int, default=5000, 
						help='vocab size (default: 5000)')
	parser.add_argument('--epochs', type=int, default=10, 
						help='number of epochs to train (default: 100)')
	parser.add_argument('--dropout', type=float, default=0.25, 
						help='dropout rate (default: 0.25)')
	parser.add_argument('--lr', type=float, default=0.001, 
						help='learning rate (default: 0.01)')
	parser.add_argument('--optimizer', type=str, default='adam', choices = ['adam','sgd','adadelta','adagrad'],
						help='optimizer: default adam')

	# training overfitting/regularization parameters
	parser.add_argument('--use-regularization', action='store_true', 
						default=True, help='uses l2 regularization during training')
	parser.add_argument('--use-l1', action='store_true', 
						default=False, help='uses l1 regularization during training')
	parser.add_argument('--l1-reg-factor', type=float, default=1000000, 
						help='l1 reg factor (default: 1000000)')
	parser.add_argument('--l2-reg-factor', type=float, default=100, 
						help='l2 reg factor (default: 1000)')

	# additinal preprocessing dataset
	parser.add_argument('--dont-preprocess-data', action='store_true', 
						default=False, help='dont preprocess data')
	parser.add_argument('--dont-lowercase-words', action='store_true', 
						default=False, help='dont lowercase all words')

	
	# additional model parameters
	parser.add_argument('--use-relu', action='store_true', 
						default=False, help='uses relu non-linearity everywhere for FCs')
	parser.add_argument('--use-pretrained-embed', action='store_true', 
						default=True, help='use pretrained word embedding intact in addition to learning')
	parser.add_argument('--learn-word-embed', action='store_true', 
						default=False, help='learn word embedding')
	parser.add_argument('--just-pad-sents', action='store_true', 
						default=False, help='use previous hidden state in predicting current tag')
	parser.add_argument('--plots', action='store_true', 
						default=False, help='use bilstm-crf for pos tagging')
	parser.add_argument('--word-embeds', type=str, default='glove', choices = [None,'word2vec','glove','polyglot'],
						help='word embeddings to be used')
	parser.add_argument('--word-embeds-file', type=str, default=None,
						help='word embeddings to be used')
	parser.add_argument('--word-embed-dim', type=int, default=100,
						help='word embeddings sizeto be used')

	# model choices, saving and loading, logging
	parser.add_argument('--use-GRU', action='store_true', 
						default=False, help='uses GRU instead of LSTMs')
	parser.add_argument('--model', type=str, default='exact_summarunner', choices = ['exact_summarunner'],
						help='model is a bi-lstm on the words')
	parser.add_argument('--load-prev-model', type=str, default=None,
						help='load previous model: filepath')
	parser.add_argument('--save-model', type=str, default=None,
						help='save model: filepath')
	parser.add_argument('--log-file', type=str, default=None,
						help='log file: filepath')

	parser.add_argument('--dynet-mem',type=int, default=1024, 
						help='dynet-mem (default: 1024)')
	parser.add_argument('--dynet-gpus',type=int, default=1, 
						help='dynet-gpus (default: 1)')
	parser.add_argument('--dynet-devices',type=str, default='--dynet-devices', 
						help='dynet-devices (default: 0)')
	parser.add_argument('--dynet-gpu', action='store_true', 
						default=False, help='uses GPU instead of cpu')
	parser.add_argument('--dynet-profiling',type=int, default=0, 
						help='dynet-profiling (default: 0)')
	parser.add_argument('--dynet-autobatch',type=int, default=1, 
						help='--dynet-autobatch (default: 1)')
	

	args = parser.parse_args()

	return args








