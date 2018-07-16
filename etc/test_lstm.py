import tensorflow as tf
import pprint
import numpy as np
from tensorflow.python.ops.rnn_cell import ResidualWrapper
from tensorflow.contrib.seq2seq.python.ops import attention_wrapper

pp = pprint.PrettyPrinter(indent=4)
sess = tf.InteractiveSession()
sequence_length = 10
input_size = 2
X_data = tf.placeholder(dtype=tf.float32,shape=[None, sequence_length, input_size],
                                           name='input_placeholder')
hidden_size = 2
# helper = tf.contrib.seq2seq.TrainHelper([2,3], sequence_length = 10)
# attention = tf.contrib.seq2seq.LuongAttention(num_units = hidden_size, )
cell1 = tf.contrib.rnn.GRUCell(num_units = hidden_size)
# cell2 = tf.contrib.rnn.BasicLSTMCell(num_units = hidden_size)
#
# x_data = np.array([[[1,0,0,0]]],dtype = np.float32)
#
# outputs, _state = tf.nn.dynamic_rnn(cell1, x_data, dtype = tf.float32)
sess.run(tf.global_variables_initializer())
n_layers= 3
cells = []
for i in range(n_layers):
    cell = tf.contrib.rnn.LSTMCell(hidden_size, state_is_tuple=True)

    cell = tf.contrib.rnn.AttentionCellWrapper(
        cell, attn_length=40, state_is_tuple=True)

    cell = tf.contrib.rnn.DropoutWrapper(cell,output_keep_prob=0.5)
    # cell = tf.contrib.rnn.ResidualWrapper(cell)
    cell = ResidualWrapper(cell)
    cells.append(cell)

cell = tf.contrib.rnn.MultiRNNCell(cells, state_is_tuple=True)


def build_encoder():
    print("building encoder..")
    with tf.variable_scope('encoder'):

        outputs, last_state = tf.nn.dynamic_rnn(
            cell=cell, inputs= X_data, dtype =tf.float32)
build_encoder()
# print((outputs))
def build_decoder_cell(self):
    encoder_outputs = self.encoder_outputs
    encoder_last_state = self.encoder_last_state
    encoder_inputs_length = self.encoder_inputs_length

    # To use BeamSearchDecoder, encoder_outputs, encoder_last_state, encoder_inputs_length
    # needs to be tiled so that: [batch_size, .., ..] -> [batch_size x beam_width, .., ..]

    # Building attention mechanism: Default Bahdanau
    # 'Bahdanau' style attention: https://arxiv.org/abs/1409.0473
    self.attention_mechanism = attention_wrapper.BahdanauAttention(
        num_units=self.hidden_units, memory=encoder_outputs,
        memory_sequence_length=encoder_inputs_length, )
    # 'Luong' style attention: https://arxiv.org/abs/1508.04025
    if self.attention_type.lower() == 'luong':
        self.attention_mechanism = attention_wrapper.LuongAttention(
            num_units=self.hidden_units, memory=encoder_outputs,
            memory_sequence_length=encoder_inputs_length, )

    # Building decoder_cell
    self.decoder_cell_list = [
        self.build_single_cell() for i in range(self.depth)]
    decoder_initial_state = encoder_last_state

    def attn_decoder_input_fn(inputs, attention):
        if not self.attn_input_feeding:
            return inputs

        # Essential when use_residual=True
        _input_layer = Dense(self.hidden_units, dtype=self.dtype,
                             name='attn_input_feeding')
        return _input_layer(array_ops.concat([inputs, attention], -1))

    # AttentionWrapper wraps RNNCell with the attention_mechanism
    # Note: We implement Attention mechanism only on the top decoder layer
    self.decoder_cell_list[-1] = attention_wrapper.AttentionWrapper(
        cell=self.decoder_cell_list[-1],
        attention_mechanism=self.attention_mechanism,
        attention_layer_size=self.hidden_units,
        cell_input_fn=attn_decoder_input_fn,
        initial_cell_state=encoder_last_state[-1],
        alignment_history=False,
        name='Attention_Wrapper')

    # To be compatible with AttentionWrapper, the encoder last state
    # of the top layer should be converted into the AttentionWrapperState form
    # We can easily do this by calling AttentionWrapper.zero_state

    # Also if beamsearch decoding is used, the batch_size argument in .zero_state
    # should be ${decoder_beam_width} times to the origianl batch_size
    batch_size = self.batch_size if not self.use_beamsearch_decode \
        else self.batch_size * self.beam_width
    initial_state = [state for state in encoder_last_state]

    initial_state[-1] = self.decoder_cell_list[-1].zero_state(
        batch_size=batch_size, dtype=self.dtype)
    decoder_initial_state = tuple(initial_state)


    return MultiRNNCell(self.decoder_cell_list), decoder_initial_state

def build_decoder():
    print("building decoder and attention..")
    with tf.variable_scope('decoder'):
        # Building decoder_cell and decoder_initial_state
        self.decoder_cell, self.decoder_initial_state = self.build_decoder_cell()

        # Initialize decoder embeddings to have variance=1.
        sqrt3 = math.sqrt(3)  # Uniform(-sqrt(3), sqrt(3)) has variance=1.
        initializer = tf.random_uniform_initializer(-sqrt3, sqrt3, dtype=self.dtype)

        self.decoder_embeddings = tf.get_variable(name='embedding',
                                                  shape=[self.num_decoder_symbols, self.embedding_size],
                                                  initializer=initializer, dtype=self.dtype)

        # Input projection layer to feed embedded inputs to the cell
        # ** Essential when use_residual=True to match input/output dims
        input_layer = Dense(self.hidden_units, dtype=self.dtype, name='input_projection')

        # Output projection layer to convert cell_outputs to logits
        output_layer = Dense(self.num_decoder_symbols, name='output_projection')

        if self.mode == 'train':
            # decoder_inputs_embedded: [batch_size, max_time_step + 1, embedding_size]
            self.decoder_inputs_embedded = tf.nn.embedding_lookup(
                params=self.decoder_embeddings, ids=self.decoder_inputs_train)

            # Embedded inputs having gone through input projection layer
            self.decoder_inputs_embedded = input_layer(self.decoder_inputs_embedded)

            # Helper to feed inputs for training: read inputs from dense ground truth vectors
            training_helper = seq2seq.TrainingHelper(inputs=self.decoder_inputs_embedded,
                                                     sequence_length=self.decoder_inputs_length_train,
                                                     time_major=False,
                                                     name='training_helper')

            training_decoder = seq2seq.BasicDecoder(cell=self.decoder_cell,
                                                    helper=training_helper,
                                                    initial_state=self.decoder_initial_state,
                                                    output_layer=output_layer)
            # output_layer=None)

            # Maximum decoder time_steps in current batch
            max_decoder_length = tf.reduce_max(self.decoder_inputs_length_train)

            # decoder_outputs_train: BasicDecoderOutput
            #                        namedtuple(rnn_outputs, sample_id)
            # decoder_outputs_train.rnn_output: [batch_size, max_time_step + 1, num_decoder_symbols] if output_time_major=False
            #                                   [max_time_step + 1, batch_size, num_decoder_symbols] if output_time_major=True
            # decoder_outputs_train.sample_id: [batch_size], tf.int32
            (self.decoder_outputs_train, self.decoder_last_state_train,
             self.decoder_outputs_length_train) = (seq2seq.dynamic_decode(
                decoder=training_decoder,
                output_time_major=False,
                impute_finished=True,
                maximum_iterations=max_decoder_length))

            # More efficient to do the projection on the batch-time-concatenated tensor
            # logits_train: [batch_size, max_time_step + 1, num_decoder_symbols]
            # self.decoder_logits_train = output_layer(self.decoder_outputs_train.rnn_output)
            self.decoder_logits_train = tf.identity(self.decoder_outputs_train.rnn_output)
            # Use argmax to extract decoder symbols to emit
            self.decoder_pred_train = tf.argmax(self.decoder_logits_train, axis=-1,
                                                name='decoder_pred_train')

            # masks: masking for valid and padded time steps, [batch_size, max_time_step + 1]
            masks = tf.sequence_mask(lengths=self.decoder_inputs_length_train,
                                     maxlen=max_decoder_length, dtype=self.dtype, name='masks')

            # Computes per word average cross-entropy over a batch
            # Internally calls 'nn_ops.sparse_softmax_cross_entropy_with_logits' by default
            self.loss = seq2seq.sequence_loss(logits=self.decoder_logits_train,
                                              targets=self.decoder_targets_train,
                                              weights=masks,
                                              average_across_timesteps=True,
                                              average_across_batch=True, )
            # Training summary for the current batch_loss
            tf.summary.scalar('loss', self.loss)

            # Contruct graphs for minimizing loss
            self.init_optimizer()

        elif self.mode == 'decode':

            # Start_tokens: [batch_size,] `int32` vector
            start_tokens = tf.ones([self.batch_size, ], tf.int32) * data_utils.start_token
            end_token = data_utils.end_token

            def embed_and_input_proj(inputs):
                return input_layer(tf.nn.embedding_lookup(self.decoder_embeddings, inputs))

            if not self.use_beamsearch_decode:
                # Helper to feed inputs for greedy decoding: uses the argmax of the output
                decoding_helper = seq2seq.GreedyEmbeddingHelper(start_tokens=start_tokens,
                                                                end_token=end_token,
                                                                embedding=embed_and_input_proj)
                # Basic decoder performs greedy decoding at each time step
                print("building greedy decoder..")
                inference_decoder = seq2seq.BasicDecoder(cell=self.decoder_cell,
                                                         helper=decoding_helper,
                                                         initial_state=self.decoder_initial_state,
                                                         output_layer=output_layer)
            else:
                # Beamsearch is used to approximately find the most likely translation
                print("building beamsearch decoder..")
                inference_decoder = beam_search_decoder.BeamSearchDecoder(cell=self.decoder_cell,
                                                                          embedding=embed_and_input_proj,
                                                                          start_tokens=start_tokens,
                                                                          end_token=end_token,
                                                                          initial_state=self.decoder_initial_state,
                                                                          beam_width=self.beam_width,
                                                                          output_layer=output_layer, )
            # For GreedyDecoder, return
            # decoder_outputs_decode: BasicDecoderOutput instance
            #                         namedtuple(rnn_outputs, sample_id)
            # decoder_outputs_decode.rnn_output: [batch_size, max_time_step, num_decoder_symbols] 	if output_time_major=False
            #                                    [max_time_step, batch_size, num_decoder_symbols] 	if output_time_major=True
            # decoder_outputs_decode.sample_id: [batch_size, max_time_step], tf.int32		if output_time_major=False
            #                                   [max_time_step, batch_size], tf.int32               if output_time_major=True

            # For BeamSearchDecoder, return
            # decoder_outputs_decode: FinalBeamSearchDecoderOutput instance
            #                         namedtuple(predicted_ids, beam_search_decoder_output)
            # decoder_outputs_decode.predicted_ids: [batch_size, max_time_step, beam_width] if output_time_major=False
            #                                       [max_time_step, batch_size, beam_width] if output_time_major=True
            # decoder_outputs_decode.beam_search_decoder_output: BeamSearchDecoderOutput instance
            #                                                    namedtuple(scores, predicted_ids, parent_ids)

            (self.decoder_outputs_decode, self.decoder_last_state_decode,
             self.decoder_outputs_length_decode) = (seq2seq.dynamic_decode(
                decoder=inference_decoder,
                output_time_major=False,
                # impute_finished=True,	# error occurs
                maximum_iterations=self.max_decode_step))

            if not self.use_beamsearch_decode:
                # decoder_outputs_decode.sample_id: [batch_size, max_time_step]
                # Or use argmax to find decoder symbols to emit:
                # self.decoder_pred_decode = tf.argmax(self.decoder_outputs_decode.rnn_output,
                #                                      axis=-1, name='decoder_pred_decode')

                # Here, we use expand_dims to be compatible with the result of the beamsearch decoder
                # decoder_pred_decode: [batch_size, max_time_step, 1] (output_major=False)
                self.decoder_pred_decode = tf.expand_dims(self.decoder_outputs_decode.sample_id, -1)

            else:
                # Use beam search to approximately find the most likely translation
                # decoder_pred_decode: [batch_size, max_time_step, beam_width] (output_major=False)
                self.decoder_pred_decode = self.decoder_outputs_decode.predicted_ids

build_decoder()