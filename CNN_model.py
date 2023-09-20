import numpy
import os
import matplotlib.pyplot as plt
from keras import applications
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.models import Sequential, Model
from keras.layers import Dropout, Flatten, Dense, GlobalAveragePooling2D
from keras import backend as k
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, TensorBoard, EarlyStopping

files_train = 0
files_test = 0

cwd = os.getcwd()
folder = 'train_data/train'

for sub_folder in os.listdir(folder):
  if sub_folder == '.DS_Store':
    continue
  path, dirs, files = next(os.walk(os.path.join(folder, sub_folder)))
  files_train += len(files)

folder = 'train_data/test'  
for sub_folder in os.listdir(folder):
  if sub_folder == '.DS_Store':
    continue
  path, dirs, files = next(os.walk(os.path.join(folder, sub_folder)))
  files_test += len(files) 

print(files_train, files_test) 

img_width, img_height = 48, 48
train_data_dir = "train_data/train"
test_data_dir = "train_data/test"
num_train_samples = files_train
num_test_samples = files_test
batch_size = 32
epochs = 15
num_classes = 2

#Check ImageDataGenerator and what it does 
#Check this section for a better understanding of Very Deep Neural Networks VGG, and Convolutional Neural Net
model = applications.VGG16(weights = "imagenet", include_top=False, input_shape=(img_width, img_height, 3))
for layer in model.layers[:10]:
  layer.trainable = False

x = model.output
x = Flatten()(x)
predictions = Dense(num_classes, activation="softmax")(x)

model_final = Model(inputs = model.input, outputs = predictions)

model_final.compile(loss = "categorical_crossentropy", optimizer = optimizers.legacy.SGD(learning_rate = 0.0001, momentum = 0.9), metrics = ["accuracy"])

train_datagen = ImageDataGenerator(
  rescale = 1./255,
  horizontal_flip = True,
  fill_mode = "nearest",
  zoom_range = 0.1,
  width_shift_range = 0.1,
  height_shift_range = 0.1,
  rotation_range = 5
)

test_datagen = ImageDataGenerator(
  rescale = 1./255,
  horizontal_flip = True,
  fill_mode = "nearest",
  zoom_range = 0.1,
  width_shift_range = 0.1,
  height_shift_range=0.1,
  rotation_range=5
)

train_generator = train_datagen.flow_from_directory(
  train_data_dir,
  target_size = (img_height, img_width),
  batch_size = batch_size,
  class_mode = "categorical"
)

test_generator = test_datagen.flow_from_directory(
  test_data_dir,
  target_size = (img_height, img_width),
  class_mode = "categorical"
)

# Save the model according to the conditions
checkpoint = ModelCheckpoint("car1.h5",monitor="accuracy", verbose=1, save_best_only=True, save_weights_only=False, mode='auto', save_freq=1)
early = EarlyStopping(monitor="accuracy", min_delta=0, patience=10, verbose=1, mode='auto')

history_object = model_final.fit(
  train_generator,
  steps_per_epoch = num_train_samples // batch_size,
  epochs = epochs,
  validation_data = test_generator,
  validation_batch_size = num_test_samples,
  callbacks = [checkpoint, early]
)

print(history_object.history.keys())
plt.plot(history_object.history['accuracy'])
plt.plot(history_object.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('acc')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
