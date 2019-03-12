import tensorflow as tf
from utils import preprocessing_helpers

def make_dataset_from_csv(path_csv, num_epoch, batch_size=128, shuffle=True, drop_remainder=False):
    
    with tf.device('/cpu:0'):
        dataset = tf.data.experimental.CsvDataset(path_csv, record_defaults=[tf.string,tf.string],header=True)
        dataset = dataset.map(
            lambda text, label: tf.py_func(
                preprocessing_helpers.process_text_with_label, [text, label, True], [tf.string, tf.int32, tf.string]), num_parallel_calls=4)
        dataset = dataset.map(lambda text, seq_len, label: (tf.string_split([text]).values, [seq_len], [label]))

        if shuffle:
            dataset = dataset.shuffle(10000)

        dataset = dataset.repeat(num_epoch)


        padded_shapes = (tf.TensorShape([None]),  # sentence of unknown size, batch will have the max sequence length
                         [1],
                         [1])

        padded_values = ('<PAD>',999,'error') # 999 and 'error' never applied

        dataset = dataset.padded_batch(batch_size, padded_shapes=padded_shapes, padding_values=padded_values, drop_remainder=drop_remainder)
        dataset = dataset.prefetch(5) #Speedup
        return dataset
    
def make_pred_dataset_from_gen(texts_gen, batch_size=128):
    
    with tf.device('/cpu:0'):
        output_types = tf.string
        
        dataset = tf.data.Dataset.from_generator(texts_gen, output_types)
        dataset = dataset.map(lambda text: tf.py_func(preprocessing_helpers.process_text_without_label, [text, True],
                                                      [tf.string, tf.int32]), num_parallel_calls=4)
        
        dataset = dataset.map(lambda text, seq_len: (tf.string_split([text]).values, [seq_len]))
        
        padded_shapes = (tf.TensorShape([None]), [1])
        
        padded_values = ('<PAD>',999) # 999 never applied

        dataset = dataset.padded_batch(batch_size, padded_shapes=padded_shapes, padding_values=padded_values)
        dataset = dataset.prefetch(5) #Speedup
        return dataset
        
        
        