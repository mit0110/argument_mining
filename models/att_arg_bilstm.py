"""BiLSTM models for argument mining with attention mechanism
"""

import numpy
import keras.backend as K
from keras import optimizers, layers
from keras.models import Model

from models.arg_bilstm import ArgBiLSTM


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
            layers.Dense(feature_vector_size, activation=None),
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

    def model_predict(self, model, sentences, return_attention=False):
        """Model probability distribution over set of labels for sentences.

        Args:
            model: a Keras model.
        """
        pred_labels = [None]*len(sentences)
        att_scores = [None]*len(sentences)
        sentenceLengths = self.getSentenceLengths(sentences)

        for indices in sentenceLengths.values():
            nnInput = []
            for featureName in self.params['featureNames']:
                inputData = numpy.asarray([sentences[idx][featureName]
                                           for idx in indices])
                nnInput.append(inputData)

            if not return_attention:
                predictions = model.predict(nnInput, verbose=False)
            else:
                attention, predictions = self.label_and_attention(
                    model, nnInput)
            predictions = predictions.argmax(axis=-1) #Predict classes

            # We add the predictions in the correct place
            for prediction_index, sequence_index in enumerate(indices):
                pred_labels[sequence_index] = predictions[prediction_index]
                if return_attention:
                    att_scores[sequence_index] = attention[prediction_index, :]

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
            for index, sentence in enumerate(sentences):
                no_pad_tokens = numpy.where(numpy.asarray(
                    sentence['tokens']))[0]
                pred_labels.append(padded_pred_labels[index][no_pad_tokens])
                if return_attention:
                    att_scores.append(padded_att_scores[index][no_pad_tokens])

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


class TimePostAttArgBiLSTM(ArgBiLSTM):

    def __init__(self, params=None):
        super(TimePostAttArgBiLSTM, self).__init__(params)
        # In this class we need to know before hand the size of the sequences,
        # and they must be all equal, to apply the attention.
        # TODO change here
        self.max_sentece_length = defaultdict(None)

    @classmethod
    def loadModel(cls, modelPath):
        # TODO add one more parameter saved, which is the size of the sequence
        model = load_model(modelPath, custom_objects=create_custom_objects())
        with h5py.File(modelPath, 'r') as f:
            mappings = json.loads(f.attrs['mappings'])
            params = json.loads(f.attrs['params'])
            modelName = f.attrs['modelName']
            labelKey = f.attrs['labelKey']

        bilstm = cls(params)
        bilstm.setMappings(mappings, None)
        bilstm.models = {modelName: model}
        bilstm.labelKeys = {modelName: labelKey}
        bilstm.idx2Labels = {}
        bilstm.idx2Labels[modelName] = {
            v: k for k, v in bilstm.mappings[labelKey].items()}
        return bilstm

    def minibatch_iterate_dataset(self, batch_size=32):
        """Create mini-batches with the same number of timesteps.

        Sentences and mini-batch chunks are shuffled and used to the
        train the model."""

        for model_name in self.modelNames:
            if not model_name in self.max_sentece_length:
                # Calculate max_sentece_length
                train_data = self.data[model_name]['trainMatrix']
                self.max_sentece_length[model_name] = max([len(x['tokens'])
                                                          for x in train_data])

            # Shuffle the order of the examples
            numpy.random.shuffle(self.data[model_name]['trainMatrix'])

        # Iterate over the examples
        batches = {}
        training_examples = len(self.data[model_name]['trainMatrix'])
        for start in range(0, training_examples, batch_size):
            batches.clear()
            end = start + batch_size

            for model_name in self.modelNames:
                sentence_size = self.max_sentece_length[model_name]
                trainMatrix = self.data[model_name]['trainMatrix']
                label_name = self.labelKeys[model_name]
                n_class_labels = len(self.mappings[self.labelKeys[model_name]])
                labels = pad_sequences(
                    [example[label_name] for example in trainMatrix[start:end]],
                     sentence_size)
                batches[model_name] = [numpy.expand_dims(labels, -1)]

                for feature_name in self.params['featureNames']:
                    inputData = pad_sequences(
                        [numpy.asarray(instance[feature_name]) + 1  # remove 0s
                         for instance in trainMatrix[start:end]], sentence_size)
                    batches[model_name].append(inputData)
            yield batches