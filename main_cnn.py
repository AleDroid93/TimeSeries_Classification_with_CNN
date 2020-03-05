import numpy as np
import tensorflow as tf
import os
from constants import N_TIMESTAMPS, N_FOLDS, N_CLASSES, EPOCHS, BATCH_SIZE, N_BANDS, EXTENSION, BASE_FILE_NAMES, BASE_DIR_FOLDS
from model import MyCnn
from utils import reshapeToTensor
from sklearn.metrics import accuracy_score, f1_score, cohen_kappa_score
from train import train


for i in range(N_FOLDS):
    current_fold = str(i + 1)
    train_fn = BASE_DIR_FOLDS + current_fold + "/" + BASE_FILE_NAMES[
        0] + current_fold + EXTENSION
    validation_fn = BASE_DIR_FOLDS + current_fold + "/" + BASE_FILE_NAMES[
        1] + current_fold + EXTENSION
    test_fn = BASE_DIR_FOLDS + current_fold + "/" + BASE_FILE_NAMES[
        2] + current_fold + EXTENSION
    target_train_fn = BASE_DIR_FOLDS + current_fold + "/" + BASE_FILE_NAMES[
        3] + current_fold + EXTENSION
    target_validation_fn = BASE_DIR_FOLDS + current_fold + "/" + BASE_FILE_NAMES[
        4] + current_fold + EXTENSION
    target_test_fn = BASE_DIR_FOLDS + current_fold + "/" + BASE_FILE_NAMES[
        5] + current_fold + EXTENSION

    # loading the data already splitted
    x_train = np.load(train_fn)
    x_validation = np.load(validation_fn)
    x_test = np.load(test_fn)

    x_train = reshapeToTensor(x_train)
    x_validation = reshapeToTensor(x_validation)
    x_test = reshapeToTensor(x_test)

    print(x_train.shape)

    y_train = np.load(target_train_fn)
    y_validation = np.load(target_validation_fn)
    y_test = np.load(target_test_fn)
    model = MyCnn(128, N_CLASSES, 3, dropout_rate=0.5)
    print(x_train.shape)
    print("Fold %s metrics:\n" % current_fold)
    # TRAINING
    outputFolder = './MLP_output'
    if not os.path.exists(outputFolder):
        os.makedirs(outputFolder)
    """ defining loss function and the optimizer to use in the training phase """
    loss_object = tf.keras.losses.CategoricalCrossentropy()
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0005)

    ckpt = tf.train.Checkpoint(step=tf.Variable(1), optimizer=optimizer, model=model)
    manager = tf.train.CheckpointManager(ckpt, outputFolder + '/MLP_ckpts_fold ' + current_fold, max_to_keep=1)
    train(model, x_train, y_train, x_validation, y_validation, loss_object, optimizer, ckpt, manager, n_epochs=EPOCHS)

    # TESTING
    pred = model.predict(x_test)
    print("Accuracy score on test set: ", accuracy_score(np.argmax(y_test, axis=1), np.argmax(pred, axis=1)))
    print("F-score on test set: ", f1_score(np.argmax(y_test, axis=1), np.argmax(pred, axis=1), average='macro'))
    print("K-score on test set: ", cohen_kappa_score(np.argmax(y_test, axis=1),  np.argmax(pred, axis=1)))