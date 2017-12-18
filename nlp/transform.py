from skimage.data import coffee

def _normalize_x_and_y(image, box):
    """ 
    Takes in an image which will be provided and then computes the normalized bouding box information.
    Args:
        xmin - this is the left most point of the bounding box
        xmax - this is the right most point of the bounding box
        ymin - this is the lowest point of the bounding box
        ymax - this is the highest point of the bouding box
    From image, try to get image width and height such that we can scale it appropriately

        (xmin,ymax).    (xmax,ymax)
        ---------------
        |             |
        |             |
        |             |
        |             |     
        |             |
        ________________
        (xmin,ymin)     (xmax,ymin)



    The output will be the following 6 parameters
        x - the left most point and is scaled between -1 and 1. to scale, can do (xmin-image_width/2)/(image_width/2)
        y - the bottom most point and is scaled between -1 and 1. to scale, can do (ymin-image_height/2)/(image_height/2)
        width - this is defined as xmax-xmin. to scale, compute (xmax-xmin)/(image_width/2)
        height - this is defined as ymax-ymin. to scale, compute (ymax-ymin)/(image_height/2)
        z - set to 0
        depth - set to 0

        (x,y+height).   (x+width, y+height)
        ---------------
        |             |
        |             |
        |             |
        |             |     
        |             |
        ________________
        (x,y)           (x+width,y)

    these test cases are for a 100*100 image but generic code is built to run any height and width
    >>> from skimage.data import coffee
    >>> img = coffee()
    >>> _normalize_x_and_y(img,(100,100,50,50))
    (-0.5, -0.8333333333333334, 0.0, 0.0, 0.0, 0.0)
    >>> _normalize_x_and_y(img,(10,90,10,90))
    (-0.95, -0.9666666666666667, 0.4, 0.26666666666666666, 0.0, 0.0)
    >>> _normalize_x_and_y(img,(0,400,0,600))
    (-1.0, -1.0, 2.0, 2.0, 0.0, 0.0)
    >>> _normalize_x_and_y(img,(100,50,0,600))
    Traceback (most recent call last):
    ...
    AssertionError: xmin is greater than xmax
    >>> _normalize_x_and_y(img,(100,600,0,600))
    Traceback (most recent call last):
    ...
    AssertionError: xmax is greater than image width
    >>> _normalize_x_and_y(img,(100,400,200,100))
    Traceback (most recent call last):
    ...
    AssertionError: ymin is greater than ymax
    >>> _normalize_x_and_y(img,(-100,400,100,100))
    Traceback (most recent call last):
    ...
    AssertionError: xmin < 0
    """
    xmin, xmax, ymin, ymax = box
    im_width, im_height = image.shape[:2]

    assert xmin <= xmax, 'xmin is greater than xmax'
    assert ymin <= ymax, 'ymin is greater than ymax'
    assert xmin <= im_width, 'xmin is greater than image width'
    assert xmax <= im_width, 'xmax is greater than image width'
    assert ymin <= im_height, 'ymin is greater than image height'
    assert ymax <= im_height, 'ymax is greater than image height'
    assert xmin >= 0, 'xmin < 0'
    assert ymin >= 0, 'ymin < 0'
    
    x_center = im_width / 2
    y_center = im_height / 2
    x = (xmin - (x_center)) / x_center
    y = (ymin - (y_center)) / y_center
    width = (xmax - xmin) / x_center
    height = (ymax - ymin) / y_center
    z = 0.0
    depth = 0.0
    return (x, y, width, height, z, depth)

def position(image,box):
    """ 
    this function takes in the image as its argument.
    the function then calls the normalizing function to get the values of x,y, width and height from
    normalize function. 
    Once the values of x,y,width and height are received, the function returns what it believes to be the
    position of the object.
    The position of the object has been defined in the follwing manner:
        if the x position is less than half of the image widht, it is in the left side but to check if it is 
        centered or not, I check the value of x+width of box. If the value of x+width is greater than 0 and the
        value of x is less than 0, i say that the object is centered. In the case that x < 0 and x+width < 0, 
        I say the object is to the left. In any case, since the width cannot be negative, if x >= 0, I say that it
        is to the right as the position. 

    Also have accounted for error messages. 
    >>> from skimage.data import coffee
    >>> img = coffee()
    >>> position(img,(0,400,0,600))
    center
    >>> position(img,(0,100,0,200))
    left
    >>> position(img,(200,400,300,400))
    right
    >>> position(img,(-100,200,300,400))
    Traceback (most recent call last):
    ...
    AssertionError: xmin < 0
    >>> position(img,(0,400,0,800))
    Traceback (most recent call last):
    ...
    AssertionError: ymax is greater than image height

    """
    
    x,y,width,height,z,depth = _normalize_x_and_y(image,box)
    position = ""
    if (x <= (0)):
        if (x+width <= 0):
            position = "left"
        else:
            position = "center"
    if (x >= (0)):
        position = "right"
    
    print (position)

if __name__ == '__main__':
    import doctest
    doctest.testmod()