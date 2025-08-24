import os
import cv2

img_size = (580, 500)

off_photos = "testing_data/training (copy)/n0"
on_photos = "testing_data/training (copy)/n0"

off = []
on = []

for filename in os.listdir(off_photos):
    image_path = os.path.join(off_photos, filename)
    img = cv2.resize(cv2.imread(image_path), img_size).flatten() # loads the image, then immediately resizes it

    off.append(img)

print("Off images loaded")
for filename in os.listdir(on_photos):
    image_path = os.path.join(on_photos, filename)
    img = cv2.resize(cv2.imread(image_path), img_size).flatten() # loads the image, then immediately resizes it

    on.append(img)

print("Data prep done")



