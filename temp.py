# Importing required packages
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

import tensorflow as tf

import matplotlib.pyplot as plt

# Paths to images
train_melanoma_dir = os.path.join('/Users/gustavchristensen/Desktop/TechLabs/skin-lesions/train/melanoma/')
train_benign_dir = os.path.join('/Users/gustavchristensen/Desktop/TechLabs/skin-lesions/train/benign/')

valid_melanoma_dir = os.path.join('/Users/gustavchristensen/Desktop/TechLabs/skin-lesions/valid/melanoma/')
valid_benign_dir = os.path.join('/Users/gustavchristensen/Desktop/TechLabs/skin-lesions/valid/benign/')

# Number of images
print('total training melanoma images:', len(os.listdir(train_melanoma_dir)))
print('total training benign images:', len(os.listdir(train_benign_dir)))
print('total valid melanoma images:', len(os.listdir(valid_melanoma_dir)))
print('total valid benign images:', len(os.listdir(valid_benign_dir)))

# Data preprocessing
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Training augmentation configuration
train_datagen = ImageDataGenerator(rescale=1./255,
                                   rotation_range=90,
                                   zoom_range=0.2,
                                   shear_range=0.2,
                                   horizontal_flip=True)

# Validation augmentation - only rescaling by 1./255
validation_datagen = ImageDataGenerator(rescale=1./255)

# Flow training images in batches of 100 using train_datagen generator
train_generator = train_datagen.flow_from_directory(
        '/Users/gustavchristensen/Desktop/TechLabs/skin-lesions/train/',  # This is the source directory for training images
        classes = ['benign', 'melanoma'],
        target_size=(200, 200),  # All images will be resized to 200x200
        batch_size=100,
        # Use binary labels
        class_mode='binary')

# Flow validation images in batches of 15 using test_datagen generator
validation_generator = validation_datagen.flow_from_directory(
        '/Users/gustavchristensen/Desktop/TechLabs/skin-lesions/valid/',  # This is the source directory for testing images
        classes = ['benign', 'melanoma'],
        target_size=(200, 200),  # All images will be resized to 200x200
        batch_size=15,
        # Use binary labels
        class_mode='binary',
        shuffle=False)

# Modelling
model = tf.keras.models.Sequential([
    # First conv layer
    tf.keras.layers.Conv2D(16, (3,3), activation='relu', input_shape=(200, 200, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    
    # Second conv layer
    tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    
    # Third conv layer
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    
    # Fourth conv layer
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    
    # Fifth conv layer
    #tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    #tf.keras.layers.MaxPooling2D(2,2),
    
    # Flatten conv layer
    tf.keras.layers.Flatten(),
    
    # 1024 neuron hidden layer
    tf.keras.layers.Dense(1024, activation='relu'),
    
    # Output layer
    tf.keras.layers.Dense(1, activation='sigmoid')])

model.compile(loss='binary_crossentropy',
              optimizer=tf.keras.optimizers.RMSprop(lr=0.001),
              metrics=['accuracy'])

# Model summary
model.summary()

# Early stopping callback
earlystopper = tf.keras.callbacks.EarlyStopping(
    monitor='val_acc',
    min_delta=0.001,
    patience=5,
    mode='max',
    restore_best_weights=True)

# Fit model
history = model.fit_generator(train_generator,
      steps_per_epoch=8,  
      epochs=15,
      verbose=1,
      validation_data=validation_generator,
      validation_steps=8,
      callbacks=[earlystopper])

# Model accuracy
loss, accuracy = model.evaluate(train_generator)
print("\nModel's Evaluation Metrics: ")
print("---------------------------")
print("Accuracy: {} \nLoss: {}".format(accuracy, loss))

# Plotting the results
acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(acc) + 1)

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()

plt.figure()

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()