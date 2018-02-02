
# coding: utf-8

# In[ ]:


import keras
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model, Sequential
from keras.layers import Input, Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
import os
from keras.callbacks import CSVLogger, TensorBoard
from keras.layers.noise import AlphaDropout
import sys
from random import randint

batch_size = 64
num_classes = 10
epochs = 1
data_augmentation = False
num_predictions = 20
experiments = 3
save_dir = os.path.join(os.getcwd(), 'saved_models')

# The data, shuffled and split between train and test sets:
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# Convert class vectors to binary class matrices.
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)


activations_all = ['sigmoid', 'tanh', 'relu', 'linear', 'elu', 'selu']
activations_sigmoid = ['sigmoid', 'tanh']
activations_linear = ['relu', 'linear', 'elu', 'selu']


mode = sys.argv[1]

if mode == 'all':
    activations = activations_all
elif mode == 'sigmoid':
    activations = activations_sigmoid
elif mode == 'linear':
    activations = activations_linear

experiment = int(sys.argv[2])

model_name = 'cifar10_'+ mode +'_all_' + str(experiment)

print('-'*30)
print('Randomized ' + mode + ' activations on all layers')
model_name = 'cifar10_rand_all_all_' + str(experiment)
dense_units = 512

print('Experiment', experiment+1)
    
inputs = Input(shape=(32,32,3))

#     x = Conv2D(32, (3, 3), padding='same', input_shape=x_train.shape[1:])(inputs)
layer = []
for i in range(32):
    activation = activations[randint(0, len(activations)-1)]
    layer.append(Conv2D(1, (3, 3), padding='same', activation=activation, input_shape=x_train.shape[1:])(inputs))

x = keras.layers.concatenate(layer)

#     x = Conv2D(32, (3, 3))(x)
layer = []
for i in range(32):
    activation = activations[randint(0, len(activations)-1)]
    layer.append(Conv2D(1, (3, 3), activation=activation)(x))

x = keras.layers.concatenate(layer)

x = MaxPooling2D(pool_size=(2, 2))(x)
x = Dropout(0.25)(x)

#     x = Conv2D(64, (3, 3), padding='same')(x)
layer = []
for i in range(64):
    activation = activations[randint(0, len(activations)-1)]
    layer.append(Conv2D(1, (3, 3), activation=activation)(x))

x = keras.layers.concatenate(layer)

#     x = Conv2D(64, (3, 3))(x)
layer = []
for i in range(64):
    activation = activations[randint(0, len(activations)-1)]
    layer.append(Conv2D(1, (3, 3), activation=activation)(x))

x = keras.layers.concatenate(layer)

x = MaxPooling2D(pool_size=(2, 2))(x)
x = Dropout(0.25)(x)
x = Flatten()(x)

layer = []
for i in range(dense_units):
    activation = activations[randint(0, len(activations)-1)]
    layer.append(Dense(1, activation=activation)(x))

x = keras.layers.concatenate(layer)  

x = Dropout(0.5)(x)
predictions = Dense(num_classes, activation='softmax')(x)

model = Model(inputs=inputs, outputs=predictions)

# initiate RMSprop optimizer
opt = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)

# Let's train the model using RMSprop
model.compile(loss='categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

csv_logger = CSVLogger('../logs/cifar_10/' + model_name + '.csv', append=True, separator=';')
tb = TensorBoard(log_dir='./tb_logs/' + model_name , histogram_freq=0, batch_size=batch_size, write_graph=True, write_grads=False, write_images=False, embeddings_freq=0, embeddings_layer_names=None, embeddings_metadata=None)
model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          validation_data=(x_test, y_test),
          shuffle=True,
         verbose=1, callbacks=[csv_logger, tb])


# Save model and weights
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)
model_path = os.path.join(save_dir, model_name + ".h5")
model.save(model_path)
print('Saved trained model at %s ' % model_path)
scores = model.evaluate(x_test, y_test, verbose=2)
print('Test loss:', scores[0])
print('Test accuracy:', scores[1])