import dynet as dy
import argparse
import logging
import os
import random
from collections import Counter, defaultdict
import matplotlib.pyplot as plt
import numpy as np

import arguments as arguments
import model as model



def parse_file(filename, dont_preprocess_data, dont_lowercase_words, just_pad_sents):

	if not dont_lowercase_words:
		logging.info('lowercasing words in the dataset')
	if not dont_preprocess_data:
		logging.info('preprocessing dataset')
	
	data = []
	count = 0			# maintaining a count if needed
	with open(filename) as f:
		sentences=[]
		labels=[]
		for i,line in enumerate(f):
			if line.strip()=='':
				if len(sentences)>0:
					assert(len(sentences) == len(labels))
					if just_pad_sents:
						sentences = ['<-start->'] + sentences + ['<-end->']
					data.append((sentences,labels))
					sentences=[]
					labels=[]
			else:
				content = line.strip().split()
				if dont_lowercase_words:
					content = [word.lower() for word in content[:-1]]
				sentences.append(content[:-1])
				labels.append(content[-1])
	return data


def get_word_label_ix(train_file, vocab_size):
	char_to_ix = {'unk':0}
	word_dict=Counter()
	label_to_ix={}

	for sentences,labels in train_file:
		for sentence in sentences:
			for word in sentence:
				word_dict[word] += 1
				for char in word:
					if char not in char_to_ix:
						char_to_ix[char] = len(char_to_ix)
		for label in labels:
			if label not in label_to_ix:
				label_to_ix[label] = len(label_to_ix)

	logging.info('total no of words: %d',len(word_dict))
	top_words = word_dict.most_common(vocab_size-1)

	logging.info('reducing no of words to vocab_size: %d',vocab_size)

	word_to_ix={'unk':0} # '<start>':1,'<end>':2}
	for e in top_words:
		val,count = e
		if val not in word_to_ix:
			word_to_ix[val]=len(word_to_ix)

	logging.info('reduced no of words to vocab_size: %d',len(word_to_ix))
	
	return char_to_ix, word_to_ix, label_to_ix


def extract_embeds(file_name, embed_dim, word_to_ix):
	word_vecs = {}

	with open(filename,'r') as f:
		for i,line in enumerate(f):
			if len(line.split())>embed_dim:
				word = line.split()[0]
				vec = [float(v) for v in line.split()[1:]]
				if word in word_to_ix:
					word_vecs[word] = vec
	

	logging.info('no of actual glove embeddings used: '+str(len(word_vecs)))
	
	if 'unk' not in word_vecs:
		word_vecs['unk'] = [0]*embed_dim # adding 'unk' token to this if not there

	return word_vecs

def train(network, train_data, dev_data, test_data, args):

	def get_val_set_acc(network, dev_data):
		evals = [network.evaluate(input_sentences, labels) for i, (input_sentences,labels) in enumerate(dev_data)]

		dy.renew_cg()
		loss = [l for l,p,c,t in evals]
		correct = [l for l,p,c,t in evals]
		total = [l for l,p,c,t in evals]
		return 100.0*sum(correct)/sum(total), sum(loss)/len(dev_data)

	if args.optimizer=='adadelta':
		trainer = dy.AdadeltaTrainer(network.model)
		trainer.set_clip_threshold(5)
	elif args.optimizer=='adam':
		trainer = dy.AdamTrainer(network.model, alpha = args.lr)
		trainer.set_clip_threshold(5)
	elif args.optimizer=='sgd-momentum':
		trainer = dy.MomentumSGDTrainer(network.model, learning_rate = args.lr)
	else:
		logging.critical('This Optimizer is not valid or not allowed')

	losses = []
	iterations = []

	batch_loss_vec=[]
	dy.renew_cg()

	is_best = 0
	best_val = 0
	count = 0
	count_train=-1
	for epoch in range(args.epochs):

		num_train = int(len(train_data)/args.batch_size + 1)*args.batch_size

		for ii in range(num_train):
			count_train+=1
			if count_train==len(train_data):
				count_train=0

			count+=1
			inputs, outputs = train_data[count_train]

			loss, pred_labels, correct, total = network.get_loss(inputs, outputs)
			batch_loss_vec.append(loss)

			if count%args.batch_size==0:
				batch_loss = dy.esum(batch_loss_vec)/args.batch_size
				batch_loss.forward()
				batch_loss.backward()
				trainer.update()
				batch_loss_vec=[]
				dy.renew_cg()

		dev_acc, dev_loss = get_val_set_acc(network, dev_data)
		losses.append(dev_loss)
		iterations.append(epoch)

		test_acc, test_loss = get_val_set_acc(network, test_data)

		logging.info('epoch %d done, dev loss: %f, dev acc: %f, test loss: %f, test acc: %f',
						epoch, dev_loss, dev_acc, test_loss, test_acc)

		m = network.model
		if epoch==0:
			best_val = dev_loss
			if args.save_model:
				m.save(args.save_model)
				logging.info('saving best model')
		else:
			if dev_loss < best_val:
				best_val = dev_loss
				if args.save_model:
					m.save(args.save_model)
					logging.info('re-saving best model')
	
	if count%args.batch_size!=0:
		batch_loss = dy.esum(batch_loss_vec)/len(batch_loss_vec)
		batch_loss.forward()
		batch_loss.backward()
		trainer.update()
		batch_loss_vec=[]
		dy.renew_cg()
	

	if args.plots:
		fig = plt.figure()
		plt.plot(iterations, losses)
		axes = plt.gca()
		axes.set_xlim([0,epochs])
		axes.set_ylim([0,10000])

		fig.savefig('figs/loss_plot.png')




def main():
	# Get all the command-line arguments
	args = arguments.get_arguments()

	if args.log_file!=None:
		logging.basicConfig(format='%(asctime)s %(message)s', filename=args.log_file, filemode='a', level=logging.DEBUG)
	else:
		logging.basicConfig(format='%(asctime)s %(message)s', level=logging.DEBUG)

	# Get all the command-line arguments
	logging.info('Obtained all arguments: %s', str(args))

	logging.info('parsing all data')
	if args.train_data:
		train_data = parse_file(args.train_data, args.dont_preprocess_data, args.dont_lowercase_words, args.just_pad_sents)
	if args.dev_data:
		dev_data = parse_file(args.dev_data, args.dont_preprocess_data, args.dont_lowercase_words, args.just_pad_sents)
	if args.test_data:
		test_data = parse_file(args.test_data, args.dont_preprocess_data, args.dont_lowercase_words, args.just_pad_sents)
	
	logging.info('train_length: %d', len(train_data))
	logging.info('dev_length: %d', len(dev_data))
	logging.info('test_length: %d', len(test_data))

	char_to_ix , word_to_ix, label_to_ix = get_word_label_ix(train_data,args.vocab_size)

	if len(word_to_ix)!=args.vocab_size:
		logging.info('vocab_size changed to %d',len(word_to_ix))
		args.vocab_size = len(word_to_ix)

	if args.word_embeds_file!=None and args.word_embeds_file!='None':
		logging.info('obtaining %s embeddings for words',args.word_embeds)
		word_vectors = extract_embeds(args.word_embeds_file, args.word_embed_dim, word_to_ix)
	else:
		word_vectors = None
		
	logging.info('Obtained word_to_ix: %d and label_to_ix: %d ', len(word_to_ix), len(label_to_ix))
	logging.info(label_to_ix)

	if args.model=='exact_summarunner':
		logging.info('using a word-level bilstm and sent-level bilstm model')
		
		summarunner = model.summarunner(args, word_to_ix, label_to_ix, word_vectors)

		logging.info('Created the network')

		logging.info('Training the network')
		
		# training the network
		train(summarunner, train_data, dev_data, test_data, args)

	else:
		logging.error('no such option for the model yet')
	







if __name__ == '__main__':
	main()
