from tensorflow.keras import layers, models
import numpy as np
import logging
from TF_NER.__common_parameters__ import LOGGER_NAME, DTYPE

import tensorflow as tf

logger = logging.getLogger(LOGGER_NAME)


def log_message(message, level=logging.INFO):
    message = "{} : {}".format("Model", message)
    logger.log(level, message)


class BiLSTMCRF(object):
    def __init__(self,
                 word_embedding=np.random.random((10, 10)),
                 graph=tf.Graph(),
                 num_labels=1,
                 batch_size=1,
                 rnn_layers=[],
                 intermediate_layer={},
                 char_embedding=np.random.random((10, 10)),
                 use_crf_out=True,
                 max_seq_len = 50
                 ):
        """
        :param word_embedding: np_array
        :param graph: tensorflow graph obj
        :param num_labels: int
        :param batch_size: int
        :param rnn_layers: list, each list item is a dict
                         : dict = {"size": int, "dropout" : 0.3, "activation" : 'tanh'}
        :param intermediate_layer: dict
                                 : dict = {"size": int, "dropout" : 0.3, "activation" : 'tanh'}
        :param char_embedding: np_array
        :param use_crf_out: boolean
                          : if true use crf as output layer else use a fully connected dense layer

        --> char_embedding and word_embedding cannot both be None
        """
        self._max_seq_len = max_seq_len
        self._word_embedding = word_embedding
        self._char_embedding = char_embedding
        self._num_labels = num_labels
        self._batch_size = batch_size
        self._rnn_layers = rnn_layers
        self._intermediate_layer = intermediate_layer
        self._use_crf_out = use_crf_out
        self._graph = graph
        self._crf = {}

    def _add_place_holders(self):
        self._input_seq_lengths = tf.compat.v1.placeholder(
            dtype=DTYPE['int']['tf'],
            shape=[self._batch_size],
            name='seq_lengths'
        )
        log_message('added place holder for sentence lengths with shape = {}'.format(self._input_seq_lengths.shape))

        # shape = [batch_size, sent_len]
        self._input_tok_ids = tf.compat.v1.placeholder(dtype=DTYPE['int']['tf'],
                                             shape=[self._batch_size, self._max_seq_len],
                                             name='input_tok_ids')
        log_message('added place holder for input batch with shape = {}'.format(self._input_tok_ids.shape))

        self._input_tok_label_ids = tf.compat.v1.placeholder(
            dtype=DTYPE['int']['tf'],
            shape=[self._batch_size, self._max_seq_len],
            name='token_lables'
        )
        log_message('added placeholder for token labels with shape = {}'.format(self._input_tok_label_ids.shape))

    def _make_cell_input(self):
        with self._graph.name_scope('word_embedding_encapsulation'):
            # shape = [vocab size, word_embedding_size]
            self._tok_embedding = tf.Variable(initial_value=self._word_embedding,
                                              name='token_embeddings',
                                              dtype=DTYPE['float']['tf'])
            log_message('character embedding has shape = {}'.format(self._tok_embedding.shape))

            # shape = [batch_size, sentence_len, word_embedding_size]
            # _cell_input = tf.nn.embedding_lookup(self._word_embedding, self._input_batch, name='cell_input_main_network')
            _cell_input = tf.nn.embedding_lookup(self._tok_embedding, self._input_tok_ids,
                                                 name='final_cell_input')
            log_message('cell input has shape = {}'.format(_cell_input.shape))

            # self._word_embedding_from_character = WordEmbeddingFromCharacter(graph=self._graph,
            #                                                                  scope_name='word_embedding_encapsulation')
            # self._word_embedding_from_character._add_place_holders()
            # self._word_embedding_from_character._add_encoder_cell()

        self._final_cell_input = _cell_input
        # self._final_cell_input = tf.concat([_cell_input,
        #                                         self._word_embedding_from_character._final_embedding],
        #                                        axis=-1,
        #                                        name='final_cell_input')
        log_message('final cell input has shape = {}'.format(self._final_cell_input.shape))

    def _make_lstm_arch(self):
        make_cell = lambda size, activation, dropout, name: tf.compat.v1.nn.rnn_cell.DropoutWrapper(
                                    tf.compat.v1.nn.rnn_cell.LSTMCell(size, activation=activation, name=name),
                                             input_keep_prob = 1 - dropout)
        with self._graph.name_scope('rnn_encapsulation_main_network'):
            self._forward_cells = tf.compat.v1.nn.rnn_cell.MultiRNNCell([make_cell(item['size'],
                                                                                    item['activation'],
                                                                                    item['dropout'],
                                                                                    'forward_cell{}'.format(idx + 1)
                                                                                    ) for idx, item in
                                   enumerate(self._rnn_layers)])
            self._backward_cells = tf.compat.v1.nn.rnn_cell.MultiRNNCell([make_cell(item['size'],
                                                                                    item['activation'],
                                                                                    item['dropout'],
                                                                                    'backward_cell{}'.format(idx + 1)
                                                                                    ) for idx, item in
                                   enumerate(self._rnn_layers)])

            (output_fw, output_bw), _ = tf.compat.v1.nn.bidirectional_dynamic_rnn(self._forward_cells,
                                                                          self._backward_cells,
                                                                          inputs=self._final_cell_input,
                                                                          sequence_length=self._input_seq_lengths,
                                                                          dtype=DTYPE['float']['tf'],
                                                                          scope='rnn_encapsulation_main_network/')

            # shape = [batch_size, sent_len, 2 * num_units]
            self._contextual_representation = tf.concat([output_fw, output_bw], axis=-1,
                                                        name='contextual_representation')
            log_message(
                'generated contextual representation with shape = {}'.format(self._contextual_representation.shape))

    def _make_output_layer(self):
        """
        --> make score for each word (contextual representation)
        --> connect contextual representation to dense layer with num entities as output
        :return:
        """
        with self._graph.name_scope('rnn_encapsulation_main_network/'):
            ## shape = [batch_size * sent_len, 2 * num_units]
            context_rep_flat = tf.reshape(self._contextual_representation,
                                          [self._batch_size*self._max_seq_len, 2 * sum([item['size'] for item in self._rnn_layers])])
            log_message('generated contextual representation flat with shape = {}'.format(context_rep_flat.shape))

        ## shape = [batch_size * sent_len, num_tags]
        output_layer = tf.layers.dense(
            context_rep_flat,
            units=self._intermediate_layer['size'],
            activation=self._intermediate_layer['activation'],
            name='tag_prediction_and_scoring/output_layer'
        )
        log_message('main network - generated prediction layer with shape = {}'.format(output_layer.shape))

        with self._graph.name_scope('tag_prediction_and_scoring/'):
            ntime_steps = tf.shape(self._contextual_representation)[1]
            self._token_scores = tf.reshape(output_layer,
                                            [self._batch_size, ntime_steps, self._num_labels], name='token_scores')
            log_message('generated token_scores with shape = {}'.format(self._token_scores.shape))

    def _get_label_scores(self):
        if self._use_crf_out:
            _log_likelihood, _transition_params = tf.contrib.crf.crf_log_likelihood(
                self._token_scores,
                self._input_tok_label_ids,
                self._input_seq_lengths
            )
            self._crf = {'log_likelihood': _log_likelihood, 'transition_params': _transition_params}
            log_message('using crf as output, log likelihood has shape = {}'.format(_log_likelihood.shape))

            self._token_label_loss = tf.reduce_mean(-_log_likelihood, name='loss')
        else:
            losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self._token_scores,
                                                                    labels=self._input_tok_label_ids)
            # shape = (batch, sentence, nclasses)
            mask = tf.sequence_mask(self._input_seq_lengths)
            # apply mask
            losses = tf.boolean_mask(losses, mask)

            self._token_label_loss = tf.reduce_mean(losses)

    @staticmethod
    def get_output_seq(token_scores, token_lengths, trans_param):
        return list(
            map(
                lambda x: {
                    'sequence' : x[0],
                    'score' : x[1]
                },
                [tf.contrib.crf.viterbi_decode(x[:seq_len], trans_param) for x, seq_len in
                 zip(token_scores, token_lengths)]
            )
        )

    def get_batch_output(self, batch_data):
        feed_dict = {
            self._input_seq_lengths : batch_data['word_seq']['lengths'],
            self._input_tok_ids : batch_data['word_seq']['seq'],
            self._input_tok_label_ids : batch_data['tags']
            #self._word_embedding_from_character._word_lengths_input_batch : batch_data['char_seq']['lengths'],
            #self._word_embedding_from_character._input_batch : batch_data['char_seq']['seq']
        }

        with tf.Session(graph=self._graph) as sess:
            sess.run(tf.global_variables_initializer())
            if self._use_crf_out:
                tok_scores, transition_params = sess.run([self._token_label_loss, self._crf['transition_params']], feed_dict=feed_dict)
                return BiLSTMCRF.get_output_seq(tok_scores, batch_data['word_seq']['lengths'], transition_params)
            else:
                return sess.run([self._token_label_loss], feed_dict=feed_dict)

    def save_graph(self,
                   path):
        with tf.Session(graph=self._graph) as sess:
            self._add_place_holders()
            self._make_cell_input()
            self._make_lstm_arch()
            self._make_output_layer()
            self._get_label_scores()
            tf.summary.FileWriter(path, graph=sess.graph)

