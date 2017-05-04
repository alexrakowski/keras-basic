from keras.engine import Model
from keras.models import Sequential


def simple(no_classes, input_shape):
    model = Sequential()
    model.add(Dense(1024, activation='relu', input_shape=input_shape))

    if no_classes > 2:
        activation = 'softmax'
        loss = 'categorical_crossentropy'
        last_layer_outputs = no_classes
    else:
        activation = 'sigmoid'
        loss = 'binary_crossentropy'
        last_layer_outputs = 1

    model.add(Dense(last_layer_outputs, activation=activation))
    model.compile(loss=loss, optimizer='adam', metrics=['accuracy'])
    return model
