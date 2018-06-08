"""BiLSTM CNN with embeddings model derived from ukplab BiLSTM model.

It has some more flexible functions, but the core of the model is the same."""

import keras.backend as K
import logging
import numpy

from sklearn import metrics
from ukplab_nets.neuralnets.BiLSTM import BiLSTM
from ukplab_nets.neuralnets.keraslayers.ChainCRF import ChainCRF

from keras import optimizers, layers
from keras.models import Model
from collections import defaultdict


class ArgBiLSTM(BiLSTM):
    """BiLSTM model tailored for argumentation mining tasks"""

    def computeScore(self, modelName, devMatrix, testMatrix):
        return self.computeF1Scores(modelName, devMatrix, testMatrix)

    def computeF1(self, modelName, sentences):
        """Returns a traditional f1 score.

        It does not check the consistency of BIO labels."""
        labelKey = self.labelKeys[modelName]
        model = self.models[modelName]
        idx2Label = self.idx2Labels[modelName]

        correctLabels = numpy.concatenate([sentences[idx][labelKey]
                         for idx in range(len(sentences))])
        pred_labels = numpy.concatenate(self.predictLabels(model, sentences))

        pre, rec, f1, _ = metrics.precision_recall_fscore_support(
            correctLabels, pred_labels, average='weighted', warn_for=[])

        return pre, rec, f1

    def featuresToMerge(self):
        tokens_input = layers.Input(shape=(None,), dtype='int32',
                                    name='words_input')
        tokens = layers.Embedding(
            input_dim=self.embeddings.shape[0],
            output_dim=self.embeddings.shape[1],
            weights=[self.embeddings], trainable=False,
            name='word_embeddings')(tokens_input)

        inputNodes = [tokens_input]
        mergeInputLayers = [tokens]

        for featureName in self.params['featureNames']:
            if featureName == 'tokens' or featureName == 'characters':
                continue

            feature_input = layers.Input(shape=(None,), dtype='int32',
                                         name=featureName+'_input')
            feature_embedding = layers.Embedding(
                input_dim=len(self.mappings[featureName]),
                output_dim=self.params['addFeatureDimensions'],
                name=featureName+'_emebddings')(feature_input)

            inputNodes.append(feature_input)
            mergeInputLayers.append(feature_embedding)
        return inputNodes, mergeInputLayers

    def addCharEmbeddings(self):
        # :: Character Embeddings ::
        logging.info("Pad words to uniform length for characters embeddings")
        all_sentences = []
        for dataset in self.data.values():
            for data in [dataset['trainMatrix'], dataset['devMatrix'],
                         dataset['testMatrix']]:
                for sentence in data:
                    all_sentences.append(sentence)

        self.padCharacters(all_sentences)
        logging.info("Words padded to %d characters" % (self.maxCharLen))

        charset = self.mappings['characters']
        charEmbeddingsSize = self.params['charEmbeddingsSize']
        maxCharLen = self.maxCharLen
        charEmbeddings= []
        for _ in charset:
            limit = math.sqrt(3.0/charEmbeddingsSize)
            vector = numpy.random.uniform(-limit, limit, charEmbeddingsSize)
            charEmbeddings.append(vector)

        charEmbeddings[0] = numpy.zeros(charEmbeddingsSize) #Zero padding
        charEmbeddings = numpy.asarray(charEmbeddings)

        chars_input = layers.Input(shape=(None, maxCharLen), dtype='int32',
                                   name='char_input')
        chars = layers.TimeDistributed(
            layers.Embedding(
                input_dim=charEmbeddings.shape[0],
                output_dim=charEmbeddings.shape[1],
                weights=[charEmbeddings], trainable=True, mask_zero=True),
            name='char_emd')(chars_input)

        # Use LSTM for char embeddings from Lample et al., 2016
        if self.params['charEmbeddings'].lower() == 'lstm':
            charLSTMSize = self.params['charLSTMSize']
            chars = layers.TimeDistributed(
                layers.Bidirectional(
                    layers.LSTM(charLSTMSize, return_sequences=False)),
                name="char_lstm")(chars)
        else:  # Use CNNs for character embeddings from Ma and Hovy, 2016
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

    def handleTasks(self, mergeInputLayers, inputNodes):
        # :: Task Identifier ::
        if self.params['useTaskIdentifier']:
            self.addTaskIdentifier()

            taskID_input = layers.Input(shape=(None,), dtype='int32',
                                        name='task_id_input')
            taskIDMatrix = numpy.identity(len(self.modelNames), dtype='float32')
            taskID_outputlayer = layers.Embedding(
                input_dim=taskIDMatrix.shape[0],
                output_dim=taskIDMatrix.shape[1], weights=[taskIDMatrix],
                trainable=False, name='task_id_embedding')(taskID_input)

            mergeInputLayers.append(taskID_outputlayer)
            inputNodes.append(taskID_input)
            self.params['featureNames'].append('taskID')

    def addRecurrentLayers(self, merged_input):
        # Add LSTMs
        shared_layer = merged_input
        logging.info("LSTM-Size: %s" % str(self.params['LSTM-Size']))
        cnt = 1
        for size in self.params['LSTM-Size']:
            if isinstance(self.params['dropout'], (list, tuple)):
                shared_layer = layers.Bidirectional(
                    layers.LSTM(size, return_sequences=True,
                                dropout=self.params['dropout'][0],
                                recurrent_dropout=self.params['dropout'][1]),
                    name='shared_varLSTM_'+str(cnt))(shared_layer)
            else:
                # Naive dropout
                shared_layer = layers.Bidirectional(
                    layers.LSTM(size, return_sequences=True),
                    name='shared_LSTM_'+str(cnt))(shared_layer)
                if self.params['dropout'] > 0.0:
                    shared_layer = layers.TimeDistributed(
                        layers.Dropout(self.params['dropout']),
                        name='shared_dropout_'+str(
                            self.params['dropout'])+"_"+str(cnt))(shared_layer)
            cnt += 1
        return shared_layer

    def addAttentionLayer(self, merged_input):
        """Add attention mechanisms to the tensor merged_input.

        Args:
            merged_input: 3-dimensional Tensor, where the first
            dimension corresponds to the batch size, the second to the sequence
            timesteps and the last one to the concatenation of features.
        """
        return merged_input

    def addOutput(self, modelName, shared_layer):
        output = shared_layer

        if modelName in self.params['customClassifier']:
            modelClassifier = self.params['customClassifier'][modelName]
        else:
            modelClassifier = self.params['classifier']

        if not isinstance(modelClassifier, (tuple, list)):
            modelClassifier = [modelClassifier]

        cnt = 1
        for classifier in modelClassifier:
            n_class_labels = len(self.mappings[self.labelKeys[modelName]])

            if classifier == 'Softmax':
                output = layers.TimeDistributed(
                    layers.Dense(n_class_labels, activation='softmax'),
                    name=modelName+'_softmax')(output)
                lossFct = 'sparse_categorical_crossentropy'
            elif classifier == 'CRF':
                output = layers.TimeDistributed(
                    layers.Dense(n_class_labels, activation=None),
                    name=modelName + '_hidden_lin_layer')(output)
                crf = ChainCRF(name=modelName+'_crf')
                output = crf(output)
                lossFct = crf.sparse_loss
            elif (isinstance(classifier, (list, tuple)) and
                  classifier[0] == 'LSTM'):
                size = classifier[1]
                if isinstance(self.params['dropout'], (list, tuple)):
                    output = layers.Bidirectional(
                        layers.LSTM(size, return_sequences=True,
                             dropout=self.params['dropout'][0],
                             recurrent_dropout=self.params['dropout'][1]),
                        name=modelName+'_varLSTM_'+str(cnt))(output)
                else:
                    """ Naive dropout """
                    output = layers.Bidirectional(
                        layers.LSTM(size, return_sequences=True),
                        name=modelName+'_LSTM_'+str(cnt))(output)
                    if self.params['dropout'] > 0.0:
                        output = layers.TimeDistributed(
                            layers.Dropout(self.params['dropout']),
                            name=modelName+'_dropout_'+str(
                                self.params['dropout'])+"_"+str(cnt))(output)
            else:
                assert(False)  # Wrong classifier
            cnt += 1
        return output, lossFct

    def getOptimizer(self):
        # :: Parameters for the optimizer ::
        optimizerParams = {}
        if ('clipnorm' in self.params and self.params['clipnorm'] != None and
                self.params['clipnorm'] > 0):
            optimizerParams['clipnorm'] = self.params['clipnorm']
        if ('clipvalue' in self.params and self.params['clipvalue'] != None and
                self.params['clipvalue'] > 0):
            optimizerParams['clipvalue'] = self.params['clipvalue']

        if self.params['optimizer'].lower() == 'adam':
            opt = optimizers.Adam(**optimizerParams)
        elif self.params['optimizer'].lower() == 'nadam':
            opt = optimizers.Nadam(**optimizerParams)
        elif self.params['optimizer'].lower() == 'rmsprop':
            opt = optimizers.RMSprop(**optimizerParams)
        elif self.params['optimizer'].lower() == 'adadelta':
            opt = optimizers.Adadelta(**optimizerParams)
        elif self.params['optimizer'].lower() == 'adagrad':
            opt = optimizers.Adagrad(**optimizerParams)
        elif self.params['optimizer'].lower() == 'sgd':
            opt = optimizers.SGD(lr=0.1, **optimizerParams)
        return opt

    def buildModel(self):
        # Refactoring code so it's easier to add attention. Same functionalities
        # as original BiLSTM network.
        self.models = {}

        inputNodes, mergeInputLayers = self.featuresToMerge()
        if self.params['charEmbeddings'] not in [
                None, "None", "none", False, "False", "false"]:
            self.addCharEmbeddings(inputNodes, mergeInputLayers)

        self.handleTasks(inputNodes, mergeInputLayers)

        if len(mergeInputLayers) >= 2:
            merged_input = layers.concatenate(mergeInputLayers)
        else:
            merged_input = mergeInputLayers[0]

        merged_input = self.addAttentionLayer(merged_input)
        shared_layer = self.addRecurrentLayers(merged_input)

        for modelName in self.modelNames:
            output, lossFct = self.addOutput(modelName, shared_layer)
            optimizer = self.getOptimizer()

            model = Model(inputs=inputNodes, outputs=[output])
            model.compile(loss=lossFct, optimizer=optimizer)

            model.summary(line_length=200)
            #logging.info(model.get_config())
            #logging.info("Optimizer: %s - %s" % (str(type(model.optimizer)), str(model.optimizer.get_config())))

            self.models[modelName] = model


class AttArgBiLSTM(ArgBiLSTM):
    """Bidirectional RNN with an attention mechanism"""

    def addAttentionLayer(self, merged_input):
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
        merged_input = layers.merge([att_layer, merged_input],  mode='mul')
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