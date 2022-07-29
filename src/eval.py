import os
import csv
import tensorflow as tf
import tensorflow.keras as keras
from sklearn.metrics import classification_report
import numpy as np


class CAMRI_Evaluate(keras.callbacks.Callback):
    def __init__(self, conf, D):
        super(CAMRI_Evaluate, self).__init__()
        self.output = [[0., 0., 0., 0., 0., 0., 0.] for _ in range(conf.epochs)]
        self.header = ['epoch', 'train_loss', 'train_recall', 'train_accuracy', 'test_loss',
                       'test_recall', 'test_accuracy']
        self.log_file = os.path.join(conf.log_path,
                                     '%s_m%d-%d_scale%d_seed%d.csv' % (
                                         conf.fname_base, conf.m_deno, conf.m_num, conf.scale,
                                         conf.seed))
        self.X, self.y, self.X_test, self.y_test = D
        self.conf = conf

    def on_epoch_end(self, epoch, logs=None):
        cce = tf.keras.losses.CategoricalCrossentropy()
        model_take_z = tf.keras.Model(inputs=self.model.input, outputs=self.model.get_layer('bf_embedding').output)
        W = tf.nn.l2_normalize(self.model.get_layer('latent_feature').W, axis=0)

        # For test data
        z = tf.nn.l2_normalize(model_take_z.predict([self.X_test, np.zeros_like(self.y_test)]), axis=1)
        logits = z @ W
        pred = tf.nn.softmax(logits)
        predict_classes = np.argmax(pred, axis=1)
        loss_test = cce(self.y_test, pred).numpy()
        test_eval_dict = classification_report(np.argmax(self.y_test, axis=1), predict_classes, output_dict=True)

        # For train data
        z = tf.nn.l2_normalize(model_take_z.predict([self.X, np.zeros_like(self.y)]), axis=1)
        logits = z @ W
        pred = tf.nn.softmax(logits)
        predict_classes = np.argmax(pred, axis=1)
        train_eval_dict = classification_report(np.argmax(self.y, axis=1), predict_classes, output_dict=True)
        loss_train = cce(self.y, pred).numpy()

        self.output[epoch] = [epoch, loss_train, train_eval_dict[str(self.conf.k)]['recall'],
                              train_eval_dict['accuracy'],
                              loss_test, test_eval_dict[str(self.conf.k)]['recall'], test_eval_dict['accuracy']]

    def on_train_end(self, logs=None):
        with open(self.log_file, 'w') as f:
            writer = csv.writer(f)
            writer.writerow(self.header)
            writer.writerows(self.output)
