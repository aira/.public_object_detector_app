import io
import os
from google.cloud import vision
from google.cloud.vision import types
from skimage.data import coffee
from PIL import Image

# setting the JSON file to be the environment variable required by GCV
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = 'apikey.json'

# initiate coffee numpy array
img = coffee()

# convert to an image using PIL and then saving the file
pil_img = Image.fromarray(img, 'RGB')
pil_img.save('image.jpg', 'JPEG')

# Instantiates a client
client = vision.ImageAnnotatorClient()

# name of image file to annotate
image_file_name = 'image.jpg'

with io.open(image_file_name, 'rb') as image_file:
    content = image_file.read()

image = types.Image(content=content)

# performs label detection
response = client.label_detection(image=image)
labels = response.label_annotations

for label in labels:
    print(label)
