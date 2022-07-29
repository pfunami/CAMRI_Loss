import os
import sys
import errno
import random
import json
import numpy as np
import configparser
from dotmap import DotMap
import tensorflow as tf
from tensorflow.keras.metrics import Recall
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.optimizers import Adam
from keras.utils.np_utils import to_categorical
import tensorflow.keras.backend as K
import archs
from eval import CAMRI_Evaluate


def set_seed(seed):
    tf.random.set_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


def load_config(config_ini_path, datasets, important_class):
    config_ini = configparser.ConfigParser()
    if not os.path.exists(config_ini_path):
        raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), config_ini_path)
    config_ini.read(config_ini_path, encoding='utf-8')
    d = dict(config_ini.items(datasets))
    d.update(dict(config_ini.items(important_class)))

    # int
    for attr in json.loads(config_ini.get("TYPE", "int")):
        d[attr] = int(d[attr])
    # float
    for attr in json.loads(config_ini.get("TYPE", "float")):
        d[attr] = float(d[attr])
    # list
    for attr in json.loads(config_ini.get("TYPE", "list")):
        d[attr] = json.loads(d[attr])

    # define margin as radian
    if 'm_num' in d and 'm_deno' in d:
        d['m'] = d['m_deno'] * np.pi / d['m_num']
    print('------------------------------')
    print(d)
    print('------------------------------')

    return DotMap(d)


def load_format():
    (x_train_raw, t_train_raw), (x_test_raw, t_test_raw) = cifar10.load_data()  # using TEST data
    t_train = to_categorical(t_train_raw)
    t_test = to_categorical(t_test_raw)
    x_train = (x_train_raw / 255)
    x_test = (x_test_raw / 255)
    return (x_train, t_train), (x_test, t_test)


def main(conf):
    model = archs.cnn_camri(conf) if conf.arch == 'CAMRI' else archs.cnn(conf)
    optimizer = Adam(lr=conf.lr) if conf.optimizer == 'Adam' else sys.exit('Optimizer is not set.')
    callbacks = [CAMRI_Evaluate(conf=conf, D=(X, y, X_test, y_test))]
    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizer,
                  metrics=['accuracy', Recall(class_id=conf.k)],
                  run_eagerly=True)
    model.summary()

    model.fit([X, y], y, validation_data=([X_test, y_test], y_test),
              batch_size=conf.batch_size,
              epochs=conf.epochs,
              callbacks=callbacks,
              verbose=1)
    model.save_weights(model_file)
    K.clear_session()


if __name__ == '__main__':
    config_ini_path = '../config.ini'
    conf = load_config(
        config_ini_path=config_ini_path,
        datasets='CIFAR10',
        important_class='CAT'  # default example
    )

    os.makedirs(conf.model_path, exist_ok=True)
    os.makedirs(conf.log_path, exist_ok=True)
    model_file = os.path.join(conf.model_path,
                              '%s_m%d-%d_scale%d_seed%d.h5' % (
                                  conf.fname_base, conf.m_deno, conf.m_num, conf.scale,
                                  conf.seed))
    set_seed(conf.seed)

    (X, y), (X_test, y_test) = load_format()
    main(conf)
