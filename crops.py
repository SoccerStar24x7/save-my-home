import cv2

def crop(image_path):

    img = cv2.imread(image_path)
    cropped_image = img[220:325, 490:790] # Slici
    cv2.imwrite(image_path, cropped_image)
