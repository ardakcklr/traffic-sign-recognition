# Gerekli modüllerin aktarılması
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import os
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from keras.layers import Conv2D, MaxPool2D, Dense, Flatten, Dropout

# Görüntü verileri ve görüntü etiketlerinin depolanacağı listelerin oluşturulması
# Sınıf sayısının belirlenmesi
data = []
labels = []
classes = 43
cur_path = os.getcwd()
# Görüntülerin ve görüntülere atanmış etiketlerin alınması
for i in range(classes):
    path = os.path.join(cur_path,'train',str(i))
    images = os.listdir(path)
    for a in images:
        try:
            image = Image.open(path + '\\'+ a)
            image = image.resize((30,30))
            image = np.array(image)
            #sim = Image.fromarray(image)
            data.append(image)
            labels.append(i)
        except:
            print("Error loading image")
# Yukarıda alınmış görüntü verisi ve etiket listelerinin numpy arraylere dönüştürülmesi
data = np.array(data)
labels = np.array(labels)
print(data.shape, labels.shape)

# Eğitim ve test veri setlerinin birbirinden ayrılması
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)
print("Shape of x_train:", X_train.shape, " and y_train:",y_train.shape)
print("Shape of x_test: ", X_test.shape, " and y_test:",y_test.shape)

# Etiketlerin one hot encoding e çevirilmesi
y_train = to_categorical(y_train, 43)
y_test = to_categorical(y_test, 43)

# Çeşitli görüntü işleme işlemleri aracılığı ile modelin oluşturulması
model = Sequential()
model.add(Conv2D(filters=32, kernel_size=(5,5), activation='relu', input_shape=X_train.shape[1:]))
model.add(Conv2D(filters=32, kernel_size=(5,5), activation='relu'))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(rate=0.25))
model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(rate=0.25))
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(rate=0.5))
model.add(Dense(43, activation='softmax'))

# Modelin derlenmesi
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
epochs = 15
history = model.fit(X_train, y_train, batch_size=32, epochs=epochs, validation_data=(X_test, y_test))
model.save("traffic_classifier.h5")

# Doğruluğu gösteren yönelim grafikleri
plt.figure(0)
plt.plot(history.history['accuracy'], label='training accuracy')
plt.plot(history.history['val_accuracy'], label='val accuracy')
plt.title('Accuracy')
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.legend()
plt.show()
plt.figure(1)
plt.plot(history.history['loss'], label='training loss')
plt.plot(history.history['val_loss'], label='val loss')
plt.title('Loss')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.legend()
plt.show()

# Doğruluğun test veri seti üzerinde test edilmesi

from sklearn.metrics import accuracy_score
import pandas as pd
y_test = pd.read_csv('Test.csv')
labels = y_test["ClassId"].values
imgs = y_test["Path"].values
test_data=[]
for img in imgs:
    image = Image.open(img)
    image = image.resize((30,30))
    test_data.append(np.array(image))

# Tahminlemelerin ve bu tahminlemelere uygun olan sınıflandırmaların aktarımı
test_data = np.array(test_data)
predictions_x = model.predict(test_data)
classes_x = np.argmax(predictions_x,axis=1)

# Test veri setinin doğruluk değerinin hesaplanması

from sklearn.metrics import accuracy_score
print("\n" +"Accuracy score: ")
print(accuracy_score(labels, classes_x))
