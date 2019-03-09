
from keras.layers import Input, Dense, MaxPooling2D, Flatten, Activation, Embedding, Lambda
from keras.layers import Conv2D
from keras.models import Model
from keras.optimizers import SGD
from keras.layers import Layer
from keras import backend as K
from keras import backend
from keras import losses
import numpy as np
import itertools
backend.set_image_dim_ordering('th')



label_size = 7

class CenterLossLayer(Layer):
    def __init__(self, alpha=0.5, lambda_2=0.5, **kwargs):
        super().__init__(**kwargs)
        self.alpha = alpha
        self.lambda_2 = lambda_2

    def build(self, input_shape):
        self.centers = self.add_weight(name='centers', shape=(7,1024), initializer='uniform', trainable=False)
        super().build(input_shape)

    def pairwise_distance(self, pair):
        fea_k = self.centers[pair[0], :]
        fea_j = self.centers[pair[1], :]
        # 1x1
        return K.dot(fea_k, K.transpose(fea_j)) + 1

    # x[0] is N*1024 , X[1] is N*7 one hot,
    def call(self, x, mask=None):

        delta_centers = K.dot(K.transpose(x[1]), K.dot(x[1], self.centers) - x[0])
        center_counts = K.sum(K.transpose(x[1]), axis=1, keepdims=True) + 1
        delta_centers /= center_counts
        new_centers = self.centers - self.alpha * delta_centers
        self.add_update((self.centers, new_centers), x)

        # Center Loss calculate
        self.result = x[0] - K.dot(x[1], self.centers)
        self.result = 0.5 * K.sum(self.result**2, axis=1, keepdims=True)


        # N x 1
        return self.result

    def compute_output_shape(self, input_shape):
        return K.int_shape(self.result)


def baseModel(alpha, img, labels):


    # Block 1
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='conv1_1')(img)
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='conv1_2')(x)

    x = MaxPooling2D((2, 2), strides=(2, 2), name='pool1')(x)

    # Block 2
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='conv2_1')(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='conv2_2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='pool2')(x)

    # Block 3
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='conv3_1')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='conv3_2')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='conv3_3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='pool3')(x)

    # Block 4
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='conv4_1')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='conv4_2')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='conv4_3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='pool4')(x)

    # Block 5
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='conv5_1')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='conv5_2')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='conv5_3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='pool5')(x)


    model = Model(input=img, output=x, name='vggface_vgg16')  # load weights
    weights_path = './data/rcmalli_vggface_tf_notop_vgg16.h5'
    model.load_weights(weights_path)

    for layer in model.layers[:-4]:
        layer.trainable = False

    base_model_output = model.output
    x = Flatten(name='flatten_final_model')(base_model_output)
    x = Dense(1024, name='fc_final_1')(x)
    x = Activation('relu', name='ss')(x)
    x = Dense(1024, name='fc_final_2')(x)
    x = Activation('relu', name='side_out')(x)

    main = Dense(7, activation='softmax', name='main_out')(x)
    # x is n x 1024 , labels n x 7 one hot
    side = CenterLossLayer(alpha=alpha, name='centerlosslayer')([x, labels])

    return main,side





def generateModel2(initial_learning_rate = 1e-3, lambda_c = 0.003, alpha = 0.5):
    main_input = Input(shape=(3, 224, 224))  # 网络输入
    aux_input = Input(shape=(7,))

    Final_output, Side_output = baseModel(alpha, main_input, aux_input)
    model = Model(inputs=[main_input, aux_input], outputs=[Final_output, Side_output])
    model.compile(optimizer=SGD(initial_learning_rate, momentum=0.9),
                  loss=[losses.categorical_crossentropy, lambda y_true, y_pred: y_pred],
                  loss_weights=[1, lambda_c],
                  metrics=['accuracy'])


    return model


