from tensorflow.keras.models import Model
from tensorflow.keras.layers import \
    Dense, Conv2D, BatchNormalization, GlobalAveragePooling2D, Input, MaxPooling2D, Dropout, Activation
from camri import CAMRI


def convs(conf, inputs):
    x = Conv2D(64, (3, 3), padding="SAME", activation="relu")(inputs)
    x = Conv2D(64, (3, 3), padding="SAME", activation="relu")(x)
    x = BatchNormalization()(x)
    x = Conv2D(64, (3, 3), padding="SAME", activation="relu")(x)
    x = MaxPooling2D()(x)
    x = Dropout(0.25)(x)
    x = Conv2D(128, (3, 3), padding="SAME", activation="relu")(x)
    x = Conv2D(128, (3, 3), padding="SAME", activation="relu")(x)
    x = BatchNormalization()(x)
    x = Conv2D(128, (3, 3), padding="SAME", activation="relu")(x)
    x = MaxPooling2D()(x)
    x = Dropout(0.25)(x)
    x = Conv2D(256, (3, 3), padding="SAME", activation="relu")(x)
    x = Conv2D(256, (3, 3), padding="SAME", activation="relu")(x)
    x = BatchNormalization()(x)
    x = Conv2D(256, (3, 3), padding="SAME", activation="relu")(x)
    x = Conv2D(256, (3, 3), padding="SAME", activation="relu")(x)
    x = Conv2D(256, (3, 3), padding="SAME", activation="relu")(x)
    x = BatchNormalization()(x)
    x = Conv2D(512, (3, 3), padding="SAME", activation="relu")(x)
    x = Conv2D(512, (3, 3), padding="SAME", activation="relu")(x)
    x = GlobalAveragePooling2D()(x)

    x = Dense(conf.z_dim, kernel_initializer='he_normal')(x)
    z = BatchNormalization(name='bf_embedding')(x)

    return z


def cnn_camri(conf):
    y = Input(shape=(conf.num_class,))
    inputs = Input(shape=tuple(conf.input_shape))

    z = convs(conf, inputs=inputs)
    logits = CAMRI(
        important_class=conf.k,
        m=conf.m,
        s=conf.scale,
        n_classes=conf.num_class,
        name='latent_feature'
    )([z, y])
    output = Activation('softmax')(logits)

    return Model([inputs, y], output)


# Reference implementation for you to understand CAMRI method by comparison with normal CNN architecture.
def cnn(conf):
    inputs = Input(shape=tuple(conf.input_shape))

    z = convs(conf, inputs=inputs)
    logits = Dense(conf.num_class, activation='linear', name="latent_feature", use_bias=True)(z)
    output = Activation('softmax')(logits)

    return Model(inputs, output)
