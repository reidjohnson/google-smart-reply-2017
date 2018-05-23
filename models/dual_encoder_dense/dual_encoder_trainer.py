import os

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.contrib.keras.api.keras.utils import Progbar
from test_tube.log import Experiment

from dataset.ubuntu_dialogue_corpus import UDCDataset
from models.dual_encoder_dense.model_dual_encoder_dense import dot_product_scoring

os.environ["KERAS_BACKEND"] = 'tensorflow'


def test_main(hparams):
    from dataset.ubuntu_dialogue_corpus import Tokenizer
    tokenizer = Tokenizer(hparams.vocab_path, hparams.max_seq_len)
    def create_prediction_data(text, labels, tokenizer, mode='encode'):
        print('creating encoding for input...')
        print('printing input: ', text)
        return (np.array([tokenizer.text_to_sequence(text)]),
                tokenizer.texts_to_sequences(labels),
                tokenizer.vocab_size())

    text = "This is a test input."

    data = pd.read_csv(hparams.dataset_train_path)
    labels = data['Utterance'].values
    text_enc, labels_enc, vocab_size = create_prediction_data(
        text, labels, tokenizer, mode='encode')
    x_len = text_enc.shape[0]
    y_len = labels_enc.shape[0]

    input_x = tf.placeholder(dtype=tf.int32, shape=[1, None], name='input_x')
    input_y = tf.placeholder(dtype=tf.int32, shape=[y_len, None], name='input_y')

    embedding = tf.get_variable('embedding',
        [vocab_size, hparams.embedding_dim])

    # Lookup the embeddings.
    embedding_x = tf.nn.embedding_lookup(embedding, input_x)
    embedding_y = tf.nn.embedding_lookup(embedding, input_y)

    # Sum the embeddings for each instance.
    x = tf.reduce_sum(embedding_x, axis=1)
    y = tf.reduce_sum(embedding_y, axis=1)

    S = dot_product_scoring(x, y, is_training=False)

    value = tf.argmax(S, axis=1, name='value')
    score = tf.reduce_max(S, axis=1, name='score')

    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))

    saver_all = tf.train.Saver(tf.global_variables(), max_to_keep=100)
    ckpt = tf.train.get_checkpoint_state(hparams.exp_name.upper() + '/epoch_0')
    saver_all.restore(sess, ckpt.model_checkpoint_path)
    sess.run(tf.global_variables_initializer())

    """Print the number of parameters."""
    total_parameters = 0
    print("Parameters:")
    for variable in tf.trainable_variables():
        shape = variable.get_shape()
        variable_parameters = 1
        for dim in shape:
            variable_parameters *= dim.value
        print("\t%d\t%s" % (variable_parameters, variable.name))
        total_parameters += variable_parameters
    print("Total number of parameters: %d." % total_parameters)

    feed_dict = {input_x: text_enc, input_y: labels_enc}
    out_value, out_score = sess.run([value, score], feed_dict=feed_dict)
    print('label: ', labels[out_value])
    print('score: ', out_score)


def train_main(hparams):
    """
    Main training routine for the dot semantic network bot
    :return:
    """

    # -----------------------
    # INIT EXPERIMENT
    # ----------------------
    exp = Experiment(name=hparams.exp_name,
                     debug=hparams.debug,
                     description=hparams.exp_desc,
                     autosave=False,
                     save_dir=hparams.test_tube_dir)

    exp.add_argparse_meta(hparams)
    exp.save()

    # -----------------------
    # LOAD DATASET
    # ----------------------
    udc_dataset = UDCDataset(vocab_path=hparams.vocab_path,
                             train_path=hparams.dataset_train_path,
                             test_path=hparams.dataset_test_path,
                             val_path=hparams.dataset_val_path,
                             max_seq_len=hparams.max_seq_len)

    # -----------------------
    # INIT TF VARS
    # ----------------------
    # input_x holds chat history
    # input_y holds our responses
    # labels holds the ground truth labels
    input_x = tf.placeholder(
        dtype=tf.int32, shape=[hparams.batch_size, None], name='input_x')
    input_y = tf.placeholder(
        dtype=tf.int32, shape=[hparams.batch_size, None], name='input_y')

    # ----------------------
    # EMBEDDING LAYER
    # ----------------------
    # you can preload your own or learn in the network
    # in this case we'll just learn it in the network
    embedding = tf.get_variable('embedding',
        [udc_dataset.vocab_size, hparams.embedding_dim])

    # ----------------------
    # RESOLVE EMBEDDINGS
    # ----------------------
    # Lookup the embeddings.
    embedding_x = tf.nn.embedding_lookup(embedding, input_x)
    embedding_y = tf.nn.embedding_lookup(embedding, input_y)

    # Generates 1 vector per training example.
    x = tf.reduce_sum(embedding_x, axis=1)
    y = tf.reduce_sum(embedding_y, axis=1)

    # ----------------------
    # OPTIMIZATION PROBLEM
    # ----------------------
    S = dot_product_scoring(x, y, is_training=True)
    K = tf.reduce_logsumexp(S, axis=1)
    loss = -tf.reduce_mean(tf.diag_part(S) - K)

    # allow optimizer to be changed through hyper params
    optimizer = get_optimizer(hparams=hparams, minimize=loss)

    # ----------------------
    # TF ADMIN (VAR INIT, SESS)
    # ----------------------
    sess = tf.Session()
    init_vars = tf.global_variables_initializer()
    sess.run(init_vars)

    # Add ops to save and restore all the variables.
    saver = tf.train.Saver()

    # ----------------------
    # TRAINING ROUTINE
    # ----------------------
    # admin vars
    nb_batches_served = 0
    eval_every_n_batches = hparams.eval_every_n_batches

    train_err = 1000
    prec_at_1 = 0
    prec_at_2 = 0

    # iter for the needed epochs
    print('\n\n', '-'*100,'\n  {} TRAINING\n'.format(hparams.exp_name.upper()), '-'*100, '\n\n')
    for epoch in range(hparams.nb_epochs):
        print('training epoch:', epoch + 1)
        progbar = Progbar(target=udc_dataset.nb_tng, width=50)
        train_gen = udc_dataset.train_generator(batch_size=hparams.batch_size, max_epochs=1)

        # mini batches
        for batch_context, batch_utterance in train_gen:
            feed_dict = {
                input_x: batch_context,
                input_y: batch_utterance
            }

            # OPT: run one step of optimization
            optimizer.run(session=sess, feed_dict=feed_dict)
            # update loss metrics
            if nb_batches_served % eval_every_n_batches == 0:

                # calculate test error
                train_err = loss.eval(session=sess, feed_dict=feed_dict)
                prec_at_1 = test_precision_at_k(S, feed_dict, k=1, sess=sess)
                prec_at_2 = test_precision_at_k(S, feed_dict, k=2, sess=sess)

                # update prog bar
                exp.add_metric_row({'tng loss': train_err, 'P@1': prec_at_1, 'P@2': prec_at_2})

            nb_batches_served += 1

            progbar.add(n=len(batch_context), values=[('train_err', train_err),
                                                      ('P@1', prec_at_1),
                                                      ('P@2', prec_at_2)])

        # ----------------------
        # END OF EPOCH PROCESSING
        # ----------------------
        # calculate the val loss
        print('\nepoch complete...\n')
        check_val_stats(loss, S, udc_dataset, hparams, input_x, input_y, exp, sess, epoch)

        # save model
        save_model(saver=saver, hparams=hparams, sess=sess, epoch=epoch)

        # save exp data
        exp.save()

    tf.reset_default_graph()


def evaluate_recall(y, y_test, k=1):
    num_examples = float(len(y))
    num_correct = 0
    for predictions, label in zip(y, y_test):
        if label in predictions[:k]:
            num_correct += 1
    print('recall', num_correct/num_examples, k)
    return num_correct/num_examples


def test_precision_at_k(pred_opt, feed_dict, k, sess):
    sims = pred_opt.eval(session=sess, feed_dict=feed_dict)
    #print('sims', sims[0], sims.shape)
    labels = range(0, len(sims))
    for i, pred_vector in enumerate(sims):
        sims[i] = [i[0] for i in sorted(enumerate(pred_vector), key=lambda x: x[1])][::-1]
    #print('sims2', sims[0], sims.shape)

    recall_score = evaluate_recall(sims, labels, k)
    return recall_score


def get_optimizer(hparams, minimize):
    opt = None
    name = hparams.optimizer_name
    if name == 'sgd':
        opt = tf.train.GradientDescentOptimizer(learning_rate=hparams.lr_1).minimize(minimize)
    if name == 'adam':
        opt = tf.train.AdamOptimizer(learning_rate=hparams.lr_1).minimize(minimize)
    if name == 'rmsprop':
        opt = tf.train.RMSPropOptimizer(learning_rate=hparams.lr_1).minimize(minimize)

    return opt


def save_model(saver, hparams, sess, epoch):
    print('saving model...')

    # create path if not around
    model_save_path = hparams.model_save_dir + '/{}/epoch_{}'.format(hparams.exp_name, epoch)
    if not os.path.isdir(model_save_path):
        os.makedirs(model_save_path)

    model_name = '{}/model.cpkt'.format(model_save_path)

    save_path = saver.save(sess, model_name)
    print('model saved at', save_path, '\n\n')


def check_val_stats(model, pred_opt, data, hparams, X_ph, Y_ph, exp, sess, epoch):
    """
    Runs through validation data to check the overall mean loss
    :param model:
    :param data:
    :param hparams:
    :param X_ph:
    :param Y_ph:
    :param exp:
    :param sess:
    :param epoch:
    :return:
    """
    print('checking val loss...')
    max_val_batches = 100
    val_gen = data.val_generator(batch_size=hparams.batch_size, max_epochs=1)

    overall_err = []
    overall_p_1 = []
    overall_p_2 = []
    progbar = Progbar(target=max_val_batches, width=50)
    for batch_nb in range(max_val_batches):
        batch_X, batch_Y = next(val_gen)
        if len(batch_X) == 0:
            continue

        # aggregate data
        feed_dict = {
            X_ph: batch_X,
            Y_ph: batch_Y
        }

        # calculate metrics
        val_err = model.eval(session=sess, feed_dict=feed_dict)
        precission_at_1 = test_precision_at_k(pred_opt, feed_dict, k=1, sess=sess)
        precission_at_2 = test_precision_at_k(pred_opt, feed_dict, k=2, sess=sess)

        # track metrics for means
        overall_err.append(val_err)
        overall_p_1.append(precission_at_1)
        overall_p_2.append(precission_at_2)

        # update exp and progbar
        exp.add_metric_row({'val loss': val_err, 'val P@1': precission_at_1, 'val P@2': precission_at_2})
        progbar.add(n=1)

    # log and save val metrics
    overall_val_mean_err = np.asarray(overall_err).mean()
    overall_p_1_mean = np.asarray(overall_p_1).mean()
    overall_p_2_mean = np.asarray(overall_p_2).mean()
    exp.add_metric_row({'epoch_mean_err': overall_val_mean_err,
                        'epoch_P@1_mean': overall_p_1_mean,
                        'epoch_P@2_mean': overall_p_2_mean,
                        'epoch': epoch + 1})

    print('\nval loss: ', overall_val_mean_err,
          'epoch_P@1_mean: ', overall_p_1_mean,
          'epoch_P@2_mean: ', overall_p_2_mean)
    print('-'*100)
