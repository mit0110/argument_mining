"""Script to run a bi_rnn, extention of the LasagneNLP scripts."""

from __future__ import print_function

import time
import sys
import argparse
import lasagne
import lasagne.nonlinearities as nonlinearities
import lasagne_nlp.utils.data_processor as data_processor
import numpy
import os
import theano.tensor as T
import theano

from lasagne_nlp.networks.networks import build_BiRNN
from lasagne_nlp.utils import utils

sys.path.insert(0, os.path.abspath('..'))
from utils import safe_mkdir


def read_args():
    parser = argparse.ArgumentParser(
        description='Tuning with bi-directional RNN')
    parser.add_argument('--fine_tune', action='store_true',
                        help='Fine tune the word embeddings')
    parser.add_argument('--embedding', choices=['word2vec', 'glove', 'senna'],
                        help='Embedding for words', required=True)
    parser.add_argument('--embedding_dict',
                        default=None, help='path for embedding dict')
    parser.add_argument('--batch_size', type=int, default=10,
                        help='Number of sentences in each batch')
    parser.add_argument('--num_units', type=int, default=100,
                        help='Number of hidden units in RNN')
    parser.add_argument('--learning_rate', type=float, default=0.1,
                        help='Learning rate')
    parser.add_argument('--decay_rate', type=float, default=0.1,
                        help='Decay rate of learning rate')
    parser.add_argument('--grad_clipping', type=float, default=0,
                        help='Gradient clipping')
    parser.add_argument('--gamma', type=float, default=1e-6,
                        help='weight for regularization')
    parser.add_argument('--oov', choices=['random', 'embedding'],
                        help='Embedding for oov word', required=True)
    parser.add_argument('--update', choices=['sgd', 'momentum', 'nesterov'],
                        help='update algorithm', default='sgd')
    parser.add_argument('--regular', choices=['none', 'l2'],
                        help='regularization for training', required=True)
    parser.add_argument('--dropout', action='store_true',
                        help='Apply dropout layers')
    parser.add_argument('--output_prediction', action='store_true',
                        help='Output predictions to temp files')
    parser.add_argument('--epochs', type=int,
                        help='Number of epochs to train the classifier')
    parser.add_argument('--train')
    parser.add_argument('--dev')
    parser.add_argument('--test')
    args = parser.parse_args()
    return args


def main():
    args = read_args()

    def construct_input_layer():
        if fine_tune:
            layer_input = lasagne.layers.InputLayer(
                shape=(None, max_length), input_var=input_var, name='input')
            layer_embedding = lasagne.layers.EmbeddingLayer(
                layer_input, input_size=alphabet_size, output_size=embedd_dim,
                W=embedd_table, name='embedding')
            return layer_embedding
        else:
            layer_input = lasagne.layers.InputLayer(
                shape=(None, max_length, embedd_dim), input_var=input_var,
                name='input')
            return layer_input

    logger = utils.get_logger("BiRNN")
    fine_tune = args.fine_tune
    oov = args.oov
    regular = args.regular
    embedding = args.embedding
    embedding_path = args.embedding_dict
    train_path = args.train
    dev_path = args.dev
    test_path = args.test
    update_algo = args.update
    grad_clipping = args.grad_clipping
    gamma = args.gamma
    output_predict = args.output_prediction
    dropout = args.dropout

    X_train, Y_train, mask_train, X_dev, Y_dev, mask_dev, X_test, Y_test, \
    mask_test, embedd_table, label_alphabet, _, _, _, _ = data_processor.load_dataset_sequence_labeling(
        train_path, dev_path,
        test_path, oov=oov,
        fine_tune=fine_tune,
        embedding=embedding,
        embedding_path=embedding_path)
    num_labels = label_alphabet.size() - 1

    logger.info("constructing network...")
    # create variables
    target_var = T.imatrix(name='targets')
    mask_var = T.matrix(name='masks', dtype=theano.config.floatX)
    if fine_tune:
        input_var = T.imatrix(name='inputs')
        num_data, max_length = X_train.shape
        alphabet_size, embedd_dim = embedd_table.shape
    else:
        input_var = T.tensor3(name='inputs', dtype=theano.config.floatX)
        num_data, max_length, embedd_dim = X_train.shape

    # construct input and mask layers
    layer_incoming = construct_input_layer()

    layer_mask = lasagne.layers.InputLayer(shape=(None, max_length),
                                           input_var=mask_var, name='mask')

    # construct bi-rnn
    num_units = args.num_units
    bi_rnn = build_BiRNN(layer_incoming, num_units, mask=layer_mask,
                         grad_clipping=grad_clipping, dropout=dropout)

    # reshape bi-rnn to [batch * max_length, num_units]
    bi_rnn = lasagne.layers.reshape(bi_rnn, (-1, [2]))

    # construct output layer (dense layer with softmax)
    layer_output = lasagne.layers.DenseLayer(
        bi_rnn, num_units=num_labels, nonlinearity=nonlinearities.softmax,
        name='softmax')

    # get output of bi-rnn shape=[batch * max_length, #label]
    prediction_train = lasagne.layers.get_output(layer_output)
    prediction_eval = lasagne.layers.get_output(layer_output,
                                                deterministic=True)
    final_prediction = T.argmax(prediction_eval, axis=1)

    # flat target_var to vector
    target_var_flatten = target_var.flatten()
    # flat mask_var to vector
    mask_var_flatten = mask_var.flatten()

    # compute loss
    num_loss = mask_var_flatten.sum(dtype=theano.config.floatX)
    # for training, we use mean of loss over number of labels
    loss_train = lasagne.objectives.categorical_crossentropy(prediction_train,
                                                             target_var_flatten)
    loss_train = (loss_train * mask_var_flatten).sum(
        dtype=theano.config.floatX) / num_loss
    ############################################
    # l2 regularization?
    if regular == 'l2':
        l2_penalty = lasagne.regularization.regularize_network_params(
            layer_output, lasagne.regularization.l2)
        loss_train = loss_train + gamma * l2_penalty

    loss_eval = lasagne.objectives.categorical_crossentropy(prediction_eval,
                                                            target_var_flatten)
    loss_eval = (loss_eval * mask_var_flatten).sum(
        dtype=theano.config.floatX) / num_loss

    # compute number of correct labels
    corr_train = lasagne.objectives.categorical_accuracy(prediction_train,
                                                         target_var_flatten)
    corr_train = (corr_train * mask_var_flatten).sum(dtype=theano.config.floatX)

    corr_eval = lasagne.objectives.categorical_accuracy(prediction_eval,
                                                        target_var_flatten)
    corr_eval = (corr_eval * mask_var_flatten).sum(dtype=theano.config.floatX)

    # Create update expressions for training.
    # hyper parameters to tune: learning rate, momentum, regularization.
    batch_size = args.batch_size
    learning_rate = args.learning_rate
    decay_rate = args.decay_rate
    momentum = 0.9
    params = lasagne.layers.get_all_params(layer_output, trainable=True)
    updates = utils.create_updates(loss_train, params, update_algo,
                                   learning_rate, momentum=momentum)

    # Compile a function performing a training step on a mini-batch
    train_fn = theano.function([input_var, target_var, mask_var],
                               [loss_train, corr_train, num_loss],
                               updates=updates)
    # Compile a second function evaluating the loss and accuracy of network
    eval_fn = theano.function([input_var, target_var, mask_var],
                              [loss_eval, corr_eval, num_loss,
                               final_prediction])

    # Finally, launch the training loop.
    log_start(batch_size, dropout, fine_tune, gamma, grad_clipping, logger,
              num_data, regular, update_algo)
    num_batches = num_data / batch_size
    num_epochs = args.epochs
    best_loss = 1e+12
    best_acc = 0.0
    best_epoch_loss = 0
    best_epoch_acc = 0
    best_loss_test_err = 0.
    best_loss_test_corr = 0.
    best_acc_test_err = 0.
    best_acc_test_corr = 0.
    stop_count = 0
    lr = learning_rate
    patience = 5

    safe_mkdir('tmp')
    for epoch in range(1, num_epochs + 1):
        logger.info('Epoch %d (learning rate=%.4f, decay rate=%.4f): ' % (
            epoch, lr, decay_rate))
        train_err = 0.0
        train_corr = 0.0
        train_total = 0
        start_time = time.time()
        train_batches = 0
        for batch in utils.iterate_minibatches(X_train, Y_train,
                                               masks=mask_train,
                                               batch_size=batch_size,
                                               shuffle=True):
            inputs, targets, masks, _ = batch
            err, corr, num = train_fn(inputs, targets, masks)
            train_err += err * num
            train_corr += corr
            train_total += num
            train_batches += 1
            time_ave = (time.time() - start_time) / train_batches
            time_left = (num_batches - train_batches) * time_ave


        # update training log after each epoch
        print('train: %d/%d loss: %.4f, acc: %.2f%%, time: %.2fs' % (
            min(train_batches * batch_size, num_data), num_data,
            train_err / train_total, train_corr * 100 / train_total,
            time.time() - start_time))

        # evaluate performance on dev data
        dev_err = 0.0
        dev_corr = 0.0
        dev_total = 0
        for batch in utils.iterate_minibatches(X_dev, Y_dev, masks=mask_dev,
                                               batch_size=batch_size):
            inputs, targets, masks, _ = batch
            err, corr, num, predictions = eval_fn(inputs, targets, masks)
            dev_err += err * num
            dev_corr += corr
            dev_total += num
            if output_predict:
                utils.output_predictions(predictions, targets, masks,
                                         'tmp/dev%d' % epoch, label_alphabet)

        log_loss('dev', dev_corr, dev_err, dev_total, logger)

        if best_loss < dev_err and best_acc > dev_corr / dev_total:
            stop_count += 1
        else:
            update_loss = False
            update_acc = False
            stop_count = 0
            if best_loss > dev_err:
                update_loss = True
                best_loss = dev_err
                best_epoch_loss = epoch
            if best_acc < dev_corr / dev_total:
                update_acc = True
                best_acc = dev_corr / dev_total
                best_epoch_acc = epoch

            # evaluate on test data when better performance detected
            test_err = 0.0
            test_corr = 0.0
            test_total = 0
            for batch in utils.iterate_minibatches(X_test, Y_test,
                                                   masks=mask_test,
                                                   batch_size=batch_size):
                inputs, targets, masks, _ = batch
                err, corr, num, predictions = eval_fn(inputs, targets, masks)
                test_err += err * num
                test_corr += corr
                test_total += num
                if output_predict:
                    utils.output_predictions(predictions, targets, masks,
                                             'tmp/test%d' % epoch,
                                             label_alphabet)

            log_loss('test', test_corr, test_err, test_total, logger)

            if update_loss:
                best_loss_test_err = test_err
                best_loss_test_corr = test_corr
            if update_acc:
                best_acc_test_err = test_err
                best_acc_test_corr = test_corr

        # stop if dev acc decrease 3 time straightly.
        if stop_count == patience:
            break

        # re-compile a function with new learning rate for training
        lr = learning_rate / (1.0 + epoch * decay_rate)
        updates = utils.create_updates(loss_train, params, update_algo, lr,
                                       momentum=momentum)
        train_fn = theano.function([input_var, target_var, mask_var],
                                   [loss_train, corr_train, num_loss],
                                   updates=updates)

    # print best performance on test data.
    logger.info(
        "final best loss test performance (at epoch %d)" % best_epoch_loss)
    log_loss('best loss in test', corr=best_loss_test_corr,
             error=best_loss_test_err, total=test_total, logger=logger)
    logger.info(
        "final best acc test performance (at epoch %d)" % best_epoch_acc)
    log_loss('best accuracy in test', corr=best_acc_test_corr,
             error=best_acc_test_err, total=test_total, logger=logger)

    # Log last predictions
    # Compile a third function evaluating the final predictions only
    predict_fn = theano.function([input_var, mask_var],
                              [final_prediction], allow_input_downcast=True)
    predictions = predict_fn(X_test, mask_test)[0]
    utils.output_predictions(predictions, Y_dev, mask_test,
                             'tmp/final_test', label_alphabet)


def log_loss(stage, corr, error, total, logger):
    logger.info('{} loss: {:.4f}, corr: {}, total: {}, acc: {:.2f}'.format(
        stage, error / total, corr, total, corr * 100 / total))


def log_start(batch_size, dropout, fine_tune, gamma, grad_clipping, logger,
              num_data, regular, update_algo):
    logger.info(
        ("Start training: {} with regularization: {}({}), dropout: {}, "
         "fine tune: {} (#training data: {}, batch size: {}, "
         "clip: {:.2f})...".format(
            update_algo, regular, (0.0 if regular == 'none' else gamma),
            dropout, fine_tune, num_data, batch_size, grad_clipping)))


if __name__ == '__main__':
    main()
