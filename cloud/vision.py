import io
import os
from google.cloud import vision
from google.cloud.vision import types
from skimage.data import coffee
from PIL import Image


def GCV(numpimage):
    """ Takes an image from the stream and runs it through google cloud vision

    Args:image (3D np.array): (rows, columns, channels)
        rows (int): width of image
        columns (int): height of image
        channels (int): number of channels, if the image is in color
    Returns: list[]
        The outputs of descriptions that Google Cloud Vison provides

    >>> from skimage.data import coffee
    >>> testimage = coffee()
    >>> GCV(testimage)
    ['espresso', 'cup', 'coffee', 'coffee cup', 'cup', 'coffee milk', 'tea', 'ristretto', 'cuban espresso', 'serveware']

    """
    # setting the JSON file to be the environment variable required by GCV
    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = 'apikey.json'
    # setting variable img equal to passed in image
    img = numpimage
    # convert the 3D numpy array to an image using PIL.Image
    pil_img = Image.fromarray(img, None)
    # producing bytes objects from the image and saving that in memory
    ramfile = io.BytesIO()
    pil_img.save(ramfile, format='JPEG')
    contents = ramfile.getvalue()
    ramfile.close()
    # Instantiates a client
    client = vision.ImageAnnotatorClient()
    # setting the contents of the image which was saved in memory as an image type that GCV can recognize
    image = types.Image(content=contents)
    # performs label detection
    response = client.label_detection(image=image)
    labels = response.label_annotations

    description = []

    for label in labels:
        description.append(label.description)

    return description


if __name__ == '__main__':
    import doctest
    doctest.testmod()
