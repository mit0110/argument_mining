"""BiLSTM models for argument mining with attention mechanism
"""

import numpy
import keras.backend as K
from keras import optimizers, layers
from keras.models import Model
from keras.preprocessing.sequence import pad_sequences

from models.arg_bilstm import ArgBiLSTM

MAX_SENTENCE_LENGHT = 400

class TimePreAttArgBiLSTM(ArgBiLSTM):
    """Bidirectional RNN with an attention mechanism.

    The attention is applied timestep wise before the BiLSTM layer."""

    def addPreAttentionLayer(self, merged_input):
        """Add attention mechanisms to the tensor merged_input.

        Args:
            merged_input: 3-dimensional Tensor, where the first
            dimension corresponds to the batch size, the second to the sequence
            timesteps and the last one to the concatenation of features.

        Retruns:
            3-dimensional Tensor of the same dimension as merged_input
        """
        feature_vector_size = K.int_shape(merged_input)[-1]
        att_layer = layers.TimeDistributed(
            layers.Dense(feature_vector_size, activation='tanh'),
            name='attention_matrix_score')(merged_input)
        # Calculate a single score for each timestep
        att_layer = layers.Lambda(lambda x: K.mean(x, axis=2),
                                  name='attention_vector_score')(att_layer)
        # Reshape to obtain the same shape as input
        att_layer = layers.Permute((2, 1))(
            layers.RepeatVector(feature_vector_size)(att_layer))
        merged_input = layers.multiply([att_layer, merged_input])
        return merged_input

    def label_and_attention(self, model, input_):
        """Classifies the sequences in input_ and returns the attention score.

        Args:
            model: a Keras model
            input_: a list of array representation of sentences.

        Returns:
            A tuple where the first element is the attention scores for each
            sentence, and the second is the model predictions.
        """
        layer = model.get_layer('attention_vector_score')
        attention_model = Model(
            inputs=model.input, outputs=[layer.output, model.output])
        # The attention output is (batch_size, timesteps, features)
        return attention_model.predict(input_)

    def model_predict(self, model, sentences, return_attention=False,
                      batch_size=32):
        """Model probability distribution over set of labels for sentences.

        Args:
            model: a Keras model.
        """
        pred_labels = []
        att_scores = []
        sentenceLengths = self.getSentenceLengths(sentences)

        for start in range(0, len(sentences), batch_size):
            end = start + batch_size
            instances = []
            for feature_name in self.params['featureNames']:
                input_data = pad_sequences(
                    [numpy.asarray(instance[feature_name])
                     for instance in sentences[start:end]],
                    self.max_sentece_length)
                instances.append(input_data)

            if not return_attention:
                predictions = model.predict(instances, verbose=False)
            else:
                attention, predictions = self.label_and_attention(
                    model, instances)
            predictions = predictions.argmax(axis=-1) #Predict classes

            # We need to "unpad" the predicted labels. We use the
            # lenght of any random feature in the sentence. (all features
            # should be valid for a sentence.
            for index, (pred, sentence) in enumerate(
                    zip(predictions, sentences[start:end])):
                sentence_len = len(sentence[feature_name])
                pred_labels.append(pred[-sentence_len:])
                if return_attention:
                    att_scores.append(attention[index, -sentence_len:])

        return numpy.asarray(pred_labels), numpy.asarray(att_scores)

    def predict(self, sentences, return_attention=False,
                translate_labels=True):
        """Distribution over labels for sentences given by all models.

        Args:
            sentences:
            return_attention: if True, return the attention scores for every
                prediction.
            translate_labels: if True, replace the numeric value of the labels
                using the dataset mappings.
        """
        # Pad characters
        if 'characters' in self.params['featureNames']:
            self.padCharacters(sentences)

        labels = {}
        attention = {}
        for model_name, model in self.models.items():
            padded_pred_labels, padded_att_scores = self.model_predict(
                model, sentences, return_attention)
            pred_labels = []
            att_scores = []
            # Skip padding tokens
            for index, (padded_pred, sentence) in enumerate(zip(
                    padded_pred_labels, sentences)):
                no_pad_tokens = numpy.where(numpy.asarray(
                    sentence['tokens']))[0]
                if no_pad_tokens.max() > padded_pred.shape[0]:
                    # The predicted sequence is shorter (it has been cut)
                    missing = no_pad_tokens.max() - padded_pred.shape[0]
                    pred_labels.append(numpy.pad(padded_pred, (0, missing),
                                                 'constant'))
                    if return_attention:
                        att_scores.append(numpy.pad(
                            padded_att_scores[index], (0, missing), 'constant'))
                else:
                    pred_labels.append(padded_pred[no_pad_tokens])
                    if return_attention:
                        att_scores.append(
                            padded_att_scores[index][no_pad_tokens])

            attention[model_name] = att_scores
            if not translate_labels:
                labels[model_name] = pred_labels
                continue
            idx2Label = self.idx2Labels[model_name]
            labels[model_name] = [[idx2Label[tag] for tag in tagSentence]
                                  for tagSentence in pred_labels]

        if return_attention:
            return labels, attention
        return labels


class FeaturePreAttArgBiLSTM(TimePreAttArgBiLSTM):

    def saveModel(self, modelName, epoch, dev_score, test_score):
        self.mappings['max_sentece_length'] = self.max_sentece_length
        super(FeaturePreAttArgBiLSTM, self).saveModel(modelName, epoch,
                                                      dev_score, test_score)

    def setMappings(self, mappings, embeddings):
        super(FeaturePreAttArgBiLSTM, self).setMappings(mappings, embeddings)
        self.max_sentece_length = self.mappings.get('max_sentece_length')

    def setDataset(self, datasets, data):
        super(FeaturePreAttArgBiLSTM, self).setDataset(datasets, data)
        dataset_name = [x for x in self.datasets.keys()][0]
        self.max_sentece_length = min(max([
            len(sentence['tokens'])
            for sentence in self.data[dataset_name]['trainMatrix']]),
            MAX_SENTENCE_LENGHT)
        print('Max sentence lenght {}'.format(self.max_sentece_length))

    def buildModel(self):
        if (not hasattr(self, 'dataset') or self.dataset is None) and (
            self.max_sentece_length is None):
            raise ValueError('Dataset must be set before build.')
        super(FeaturePreAttArgBiLSTM, self).buildModel()

    def featuresToMerge(self):
        """Adds the input layers."""
        tokens_input = layers.Input(shape=(self.max_sentece_length,),
                                    dtype='int32', name='words_input')
        tokens = layers.Embedding(
            input_dim=self.embeddings.shape[0],
            output_dim=self.embeddings.shape[1],
            weights=[self.embeddings], trainable=False,
            mask_zero=False,  # The attention should deal with this
            name='word_embeddings')(tokens_input)

        inputNodes = [tokens_input]
        mergeInputLayers = [tokens]

        for featureName in self.params['featureNames']:
            if featureName == 'tokens' or featureName == 'characters':
                continue

            feature_input = layers.Input(
                shape=(self.max_sentece_length,), dtype='int32',
                name=featureName+'_input')
            feature_embedding = layers.Embedding(
                input_dim=len(self.mappings[featureName]),
                output_dim=self.params['addFeatureDimensions'],
                mask_zero=False,  # The attention should deal with this
                name=featureName+'_emebddings')(feature_input)

            inputNodes.append(feature_input)
            mergeInputLayers.append(feature_embedding)
        return inputNodes, mergeInputLayers

    def addCharInput(self):
        return layers.Input(
            shape=(self.max_sentece_length, self.maxCharLen), dtype='int32',
            name='char_input')

    def addCharEmbeddingLayers(self, inputNodes, mergeInputLayers,
                               chars_input, charEmbeddings):
        # Use LSTM for char embeddings from Lample et al., 2016
        if self.params['charEmbeddings'].lower() == 'lstm':
            chars = layers.TimeDistributed(
                layers.Embedding(
                    input_dim=charEmbeddings.shape[0],
                    output_dim=charEmbeddings.shape[1],
                    weights=[charEmbeddings], trainable=True, mask_zero=False),
                name='char_emd')(chars_input)
            charLSTMSize = self.params['charLSTMSize']
            chars = layers.TimeDistributed(
                layers.Bidirectional(
                    layers.LSTM(charLSTMSize, return_sequences=False)),
                name="char_lstm")(chars)
        else:  # Use CNNs for character embeddings from Ma and Hovy, 2016
            chars = layers.TimeDistributed(
                layers.Embedding(  # Conv layer does not support masking
                    input_dim=charEmbeddings.shape[0],
                    output_dim=charEmbeddings.shape[1], trainable=True,
                    weights=[charEmbeddings]), name='char_emd')(chars_input)
            charFilterSize = self.params['charFilterSize']
            charFilterLength = self.params['charFilterLength']
            chars = layers.TimeDistributed(
                layers.Conv1D(charFilterSize, charFilterLength, padding='same'),
                name="char_cnn")(chars)
            chars = layers.TimeDistributed(layers.GlobalMaxPooling1D(),
                                           name="char_pooling")(chars)

        mergeInputLayers.append(chars)
        inputNodes.append(chars_input)
        self.params['featureNames'].append('characters')

    def addPreAttentionLayer(self, merged_input):
        """Add attention mechanisms to the tensor merged_input.

        Args:
            merged_input: 3-dimensional Tensor, where the first
            dimension corresponds to the batch size, the second to the sequence
            timesteps and the last one to the concatenation of features.

        Retruns:
            3-dimensional Tensor of the same dimension as merged_input
        """
        feature_vector_size = K.int_shape(merged_input)[-1]
        merged_input = layers.Permute((2, 1))(merged_input)
        att_layer = layers.TimeDistributed(
            layers.Dense(self.max_sentece_length, activation=None),
            name='attention_matrix_score')(merged_input)
        # Calculate a single score for each timestep
        att_layer = layers.Lambda(lambda x: K.mean(x, axis=1),
                                  name='attention_vector_score')(att_layer)
        # Reshape to obtain the same shape as input
        att_layer = layers.RepeatVector(feature_vector_size)(att_layer)
        merged_input = layers.multiply([att_layer, merged_input])
        merged_input = layers.Permute((2, 1))(merged_input)

        # We re add the mask layer after the attention is applied.
        # Of course we have the risk of masking elements that were zeroed
        # after the application of the attention scores.
        merged_input = layers.Masking(mask_value=0.0)(merged_input)
        return merged_input

