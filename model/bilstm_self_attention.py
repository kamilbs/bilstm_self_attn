import tensorflow as tf

class BiLSTMSelfAttention(object):
    
    def __init__(self, params):
        self.params = params
    
    def __call__(self, inputs, targets=None):
        sequence_length = inputs['sequence_length']
        input_embeddings = inputs['input_embeddings']
        
        with tf.variable_scope('bilstm'):
            cell_fw = tf.nn.rnn_cell.LSTMCell(self.params['num_hidden_units']) 
            cell_bw = tf.nn.rnn_cell.LSTMCell(self.params['num_hidden_units'])
            outputs, _ = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw, input_embeddings, sequence_length, dtype=tf.float32)

            # for things above sequence length the outputs will be full of 0 
            hidden = tf.concat(outputs,2) # batch_size x n x 2*num_hidden_units


        with tf.variable_scope('attention'):
            WS1 = tf.get_variable('WS1',
                                  shape=(self.params['attention_dimension'],2*self.params['num_hidden_units']),
                                  dtype=tf.float32)
            WS2 = tf.get_variable('WS2',
                                  shape=(self.params['number_attention_hop'],self.params['attention_dimension']),
                                  dtype=tf.float32)

            WS1_HT = tf.transpose(tf.tensordot(WS1,hidden,axes=[[1],[2]]),[1,0,2]) # batch_size x da x n
            RAW_A = tf.transpose(tf.tensordot(WS2, tf.nn.tanh(WS1_HT),axes=[[1],[1]]),[1,0,2]) # batch_size x r x n
            A = tf.nn.softmax(RAW_A) # batch_size x r x n

            if self.params['masking_attention']:
                sequence_mask = tf.expand_dims(tf.sequence_mask(sequence_length),1) # batch_size x 1 x n
                sequence_mask_value = tf.cast(sequence_mask,tf.float32)
                A_masked = A * sequence_mask_value # batch_size x r x n
                A_masked = A_masked / tf.reduce_sum(A_masked, axis=2, keepdims=True)
                A = A_masked

            M = tf.matmul(A,hidden) # batch_size x r x 2*num_hidden_units
            flatten_weighted_vectors = tf.layers.flatten(M) # batch_size x r*2*num_hidden_units

        with tf.variable_scope('dense'):
            out = flatten_weighted_vectors
            for i in range(self.params['num_hidden_layers']):
                out = tf.layers.dense(out, self.params['num_dense_units'], activation=tf.nn.relu)
            logits = tf.layers.dense(out, self.params['num_classes'])
        
        
        return logits, A