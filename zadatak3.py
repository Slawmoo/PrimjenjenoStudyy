import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
import seaborn as sns
import tensorflow as tf
from keras.utils import image_dataset_from_directory
 
# ucitavanje podataka iz odredenog direktorija 
train_ds = image_dataset_from_directory( 
    directory='C:/Users/student/Desktop/Karavla8/Train', 
    labels='inferred', 
    label_mode='categorical', 
    batch_size=32, 
    image_size=(48, 48)) 

test_ds = image_dataset_from_directory( 
    directory='C:/Users/student/Desktop/Karavla8/Test', 
    labels='inferred', 
    label_mode='categorical', 
    batch_size=32, 
    image_size=(48, 48)) 

# TODO: kreiraj model pomocu keras.Sequential(); prikazi njegovu strukturu

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Conv2D(32, (3, 3), activation='relu',  input_shape=(48, 48, 3)))
model.add(tf.keras.layers.Conv2D(32, (3, 3), activation='relu'))
model.add(tf.keras.layers.MaxPooling2D((2, 2)))
model.add(tf.keras.layers.Dropout(0.2))
model.add(tf.keras.layers.Conv2D(32, (3, 3), activation='relu'))
model.add(tf.keras.layers.Conv2D(32, (3, 3), activation='relu'))
model.add(tf.keras.layers.MaxPooling2D((2, 2)))
model.add(tf.keras.layers.Dropout(0.2))
model.add(tf.keras.layers.Conv2D(32, (3, 3), activation='relu'))
model.add(tf.keras.layers.Conv2D(32, (3, 3), activation='relu'))
model.add(tf.keras.layers.MaxPooling2D((2, 2)))
model.add(tf.keras.layers.Dropout(0.2))
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(512, activation='relu'))
model.add(tf.keras.layers.Dropout(0.2))
model.add(tf.keras.layers.Dense(43, activation='softmax'))

# TODO: definiraj karakteristike procesa ucenja pomocu .compile()

model.compile(loss='categorical_crossentropy', 
              optimizer='sgd', 
              metrics=['accuracy'])

# TODO: provedi ucenje mreze
model.fit(train_ds, epochs=1, batch_size=128)

# TODO: Prikazi test accuracy i matricu zabune
loss_and_metrics = model.evaluate(test_ds, batch_size=128) 

y_pred = model.predict(test_ds, batch_size=128)
y_pred = np.argmax(y_pred)
y_test = np.argmax(test_ds)

cm = confusion_matrix(y_test, y_pred)

num_classes = len(np.unique(y_test))

classifiers = range(43)
# Plot confusion matrix as heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(cm, annot=True, cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')

model.summary() 

plt.xticks(np.arange(num_classes) + 0.5, classifiers, rotation='vertical')

# Add classifiers as y-tick labels
plt.yticks(np.arange(num_classes) + 0.5, classifiers, rotation='horizontal')

plt.show()

# TODO: spremi model
model.save("model.h5")