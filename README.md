# bilstm_self_attn

Implementation of "A structured self-attentive sentence embedding" with Tensorflow 1.13x. and using tf.data for data ingestion.

Contains everything that you need , by just starting from a CSV with the raw text and label.

1. (Optional) Fetch the data and create the proper CSV with text, label: dataset_notebooks/sentiment140.ipynb as an example 
2. Prepare Vocabulary and Embedding Matrix notebook
3. Classifier Attn notebook contains the model as a TF estimator

