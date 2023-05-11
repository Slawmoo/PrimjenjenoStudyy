from skimage.transform import resize
from skimage import color
import matplotlib.image as mpimg
import numpy as np
import joblib
import matplotlib.pyplot as plt
from PIL import ImageOps


images=('test.png','TestMy.png','TestMy1.png','TestMy2.png','TestMy3.png')
# ucitaj sliku i prikazi ju
for image in images:
    filename = image

    img = ImageOps.Image.open(filename)
    img = img.resize((28, 28))  # resize image to 28x28
    gray = np.mean(img, axis=2).astype(np.uint8)  # convert to grayscale
    vector = gray.reshape(784)

    plt.figure()
    plt.imshow(gray, cmap=plt.get_cmap('gray'))
    plt.show()

    # prebacite sliku u vektor odgovarajuce velicine
    img_vector = vector.reshape(1, -1)

    # ucitavanje modela
    filename = "NN_model.sav"
    mlp_mnist = joblib.load(filename)

    # napravi predikciju i spremi u varijablu label kao string
    label = str(mlp_mnist.predict(img_vector)[0])

    print("------------------------")
    print("Slika sadrzi znamenku:", label)
