from keras.layers import Dense
from keras.models import Sequential


def simple(no_classes, input_shape):
    model = Sequential()
    model.add(Dense(128, activation='relu', input_shape=input_shape))
    model.add(Dense(128, activation='relu', ))
    model.add(Dense(128, activation='relu', ))

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
    model.__setattr__('name', 'simple')
    return model
