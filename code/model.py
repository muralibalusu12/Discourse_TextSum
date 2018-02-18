import dynet as dy
import argparse
import logging
import os
import random
from collections import Counter, defaultdict
import matplotlib.pyplot as plt
import numpy as np



class summarunner():
	
	def __init__(self, args, word_to_ix, label_to_ix, word_vecs = None):
		self.args = args
		self.model = dy.Model()
		self.word_vecs = word_vecs
		self.word_to_ix = word_to_ix
		self.label_to_ix = label_to_ix

		logging.info('finalized vocab size: '+str(args.vocab_size))

		self.ix_to_label = {}
		for k,v in self.label_to_ix.items():
			self.ix_to_label[v] = k

		self.word_embeddings = self.model.add_lookup_parameters((len(self.word_to_ix),args.word_dim))

		self.abspos_ix = {i:i for i in range(args.sent_pos)}
		self.relpos_ix = {i: int(i*(args.sent_buckets)/args.sent_pos) for i in range(args.sent_pos)}
		self.abspos_embeddings = self.model.add_lookup_parameters((args.sent_pos,args.pos_dim))
		self.relpos_embeddings = self.model.add_lookup_parameters((args.sent_buckets,args.pos_dim))

		logging.info(self.abspos_ix)
		logging.info(self.relpos_ix)

		if self.word_vecs:
			self.word_embeddings.init_from_array(np.array(word_vecs))

		if args.use_GRU:
			self.word_rnn = dy.BiRNNBuilder(args.word_num_of_layers, args.word_dim, args.word_hidden_dim, self.model, dy.GRUBuilder)
		else:
			self.word_rnn= dy.BiRNNBuilder(args.word_num_of_layers, args.word_dim, args.word_hidden_dim, self.model, dy.VanillaLSTMBuilder)
		
		if args.use_GRU:
			self.sent_rnn = dy.BiRNNBuilder(args.sent_num_of_layers, args.sent_dim, args.sent_hidden_dim, self.model, dy.GRUBuilder)
		else:
			self.sent_rnn= dy.BiRNNBuilder(args.sent_num_of_layers, args.sent_dim, args.sent_hidden_dim, self.model, dy.VanillaLSTMBuilder)

		self.doc_output_w = self.model.add_parameters((args.doc_dim, args.sent_hidden_dim))
		self.doc_output_b = self.model.add_parameters((args.doc_dim))

		self.sent_content_output = self.model.add_parameters((args.base_dim, args.sent_hidden_dim))
		self.sent_salience_output = self.model.add_parameters((args.sent_hidden_dim, args.doc_dim))
		self.sent_novelty_output = self.model.add_parameters((args.sent_hidden_dim, args.sent_hidden_dim))
		self.sent_abspos_output = self.model.add_parameters((args.base_dim, args.pos_dim))
		self.sent_relpos_output = self.model.add_parameters((args.base_dim, args.pos_dim))
		self.sent_bias = self.model.add_parameters((args.base_dim))

		if self.args.load_prev_model is not None:
			self.model.populate(args.load_prev_model)

	
	def regularize_weights(self):
		pass

	# preprocessing function for all inputs
	def _preprocess_input(self, sentence, to_ix):
		if 'unk' in to_ix: # for words in sentence
			return [to_ix[word] if word in to_ix else to_ix['unk'] for word in sentence]
		else: # for labels of the sentences
			return [to_ix[word] for word in sentence]

	# embed the sentence with embeddings(look up parameters) for each word
	def _embed_sentence(self, sentence):
		#return [self.embeddings[word] for word in sentence]
		return [dy.lookup(self.word_embeddings, word) for word in sentence]

	# run rnn on the specified input with the corresponding init state
	def _run_rnn(self, init_state, input_vecs):
		s = init_state
		states = s.transduce(input_vecs)
		rnn_outputs = [s for s in states]
		return rnn_outputs

	# get probabilities of the label for that particular sentence
	def _get_probs(self, rnn_output, doc_output, sum_output, abspos_embed, relpos_embed):

		sent_content_output = dy.parameter(self.sent_content_output)
		sent_salience_output = dy.parameter(self.sent_salience_output)
		sent_novelty_output = dy.parameter(self.sent_novelty_output)
		sent_abspos_output = dy.parameter(self.sent_abspos_output)
		sent_relpos_output = dy.parameter(self.sent_relpos_output)
		sent_bias = dy.parameter(self.sent_bias)

		final_output = sent_content_output*rnn_output +\
						dy.transpose(rnn_output)*sent_salience_output*doc_output -\
						dy.transpose(rnn_output)*sent_novelty_output*dy.tanh(sum_output) +\
						sent_abspos_output*abspos_embed +\
						sent_relpos_output*relpos_embed +\
						sent_bias
		return dy.logistic(final_output)

	# predicting the label: greedy decoding here
	def _predict(self, probs):
		probs = probs.value()
		if probs>self.args.threshold:
			value=1
		else:
			value=0
		idx = value
		return self.ix_to_label[idx]
	
	# run the forward and backward lstm, then obtains probs over tags, calculates the loss and returns it
	def get_loss(self, input_sentences, labels):

		self.word_rnn.set_dropout(self.args.dropout)
		self.sent_rnn.set_dropout(self.args.dropout)

		embed_sents=[]

		for input_sentence in input_sentences:
			input_sentence = self._preprocess_input(input_sentence, self.word_to_ix)
			#input_sentence = [self.word_to_ix['<start>']] + input_sentence + [self.word_to_ix['<end>']]

			embed_words = self._embed_sentence(input_sentence)
			word_rnn_outputs = self._run_rnn(self.word_rnn, embed_words)
			sent_embed = dy.average(word_rnn_outputs)
			embed_sents.append(sent_embed)

		rnn_outputs = self._run_rnn(self.sent_rnn, embed_sents)

		doc_output_w = dy.parameter(self.doc_output_w)
		doc_output_b = dy.parameter(self.doc_output_b)
		doc_output = dy.tanh(doc_output_w*dy.average(rnn_outputs) + doc_output_b)

		probs=[]
		sum_output = dy.zeros(self.args.sent_hidden_dim)
		pred_labels = []
		correct = 0
		total = 0
		loss = dy.zeros(1)
		for i,rnn_output in enumerate(rnn_outputs):

			abspos_embed = dy.lookup(self.abspos_embeddings, self.abspos_ix[i])
			relpos_embed = dy.lookup(self.relpos_embeddings, self.relpos_ix[i])

			prob = self._get_probs(rnn_output, doc_output, sum_output, abspos_embed, relpos_embed)
			sum_output += dy.cmult(prob,rnn_output)
			pred_label = self._predict(prob)
			pred_labels.append(pred_label)

			if pred_label == labels[i]:
				correct+=1
			total+=1

			if labels[i]==1:
				loss -= dy.log(prob)
			else:
				loss -= dy.log(dy.scalarInput(1) - prob)

		return loss, pred_labels, correct, total


	def evaluate(self, input_sentences, labels):

		dy.renew_cg()

		self.word_rnn.disable_dropout()
		self.sent_rnn.disable_dropout()

		embed_sents=[]

		for input_sentence in input_sentences:
			input_sentence = self._preprocess_input(input_sentence, self.word_to_ix)
			#input_sentence = [self.word_to_ix['<start>']] + input_sentence + [self.word_to_ix['<end>']]

			embed_words = self._embed_sentence(input_sentence)
			word_rnn_outputs = self._run_rnn(self.word_rnn, embed_words)
			sent_embed = dy.average(word_rnn_outputs)
			embed_sents.append(sent_embed)

		rnn_outputs = self._run_rnn(self.sent_rnn, embed_sents)

		doc_output_w = dy.parameter(self.doc_output_w)
		doc_output_b = dy.parameter(self.doc_output_b)
		doc_output = dy.tanh(doc_output_w*dy.average(rnn_outputs) + doc_output_b)

		probs=[]
		sum_output = dy.zeros(self.args.sent_hidden_dim)
		pred_labels = []
		correct = 0
		total = 0
		loss = dy.zeros(1)
		for i,rnn_output in enumerate(rnn_outputs):

			abspos_embed = dy.lookup(self.abspos_embeddings, self.abspos_ix[i])
			relpos_embed = dy.lookup(self.relpos_embeddings, self.relpos_ix[i])

			prob = self._get_probs(rnn_output, doc_output, sum_output, abspos_embed, relpos_embed)
			sum_output += dy.cmult(prob,rnn_output)
			pred_label = self._predict(prob)
			pred_labels.append(pred_label)

			if pred_label == labels[i]:
				correct+=1
			total+=1

			if labels[i]==1:
				loss -= dy.log(prob)
			else:
				loss -= dy.log(dy.scalarInput(1) - prob)

		return loss.value(), pred_labels, correct, total













