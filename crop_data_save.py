import os
import cv2
from matplotlib import pyplot as plt
from matplotlib import pyplot
from matplotlib.patches import Rectangle
from mtcnn.mtcnn import MTCNN

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

real_directory = './data/real_and_fake_face/training_real/'
fake_directory = './data/real_and_fake_face/training_fake/'

real_crop_directory = './crop_data/real_and_fake_face/training_real/'
fake_crop_directory = './crop_data/real_and_fake_face/training_fake/'

real = os.listdir(real_directory)
fake = os.listdir(fake_directory)

def draw_image_with_boxes(filename, result_list, new_filename):
    global y
    global height
    global x
    global width
    # load the image
    #data = pyplot.imread(filename)
    # plot the image
    #pyplot.imshow(data)
    # get the context for drawing boxes
    #ax = pyplot.gca()
    # plot each box
    for result in result_list:
        # get coordinates
        x, y, width, height = result['box']
        # create the shape
        #rect = Rectangle((x, y), width-x, height-y, fill=False, color='red')
        # draw the box
        #ax.add_patch(rect)
    # show the plot
    #pyplot.show()

    img2 = cv2.imread(filename)
    cropped_image = img2[y:y+height, x:x+width]
    cv2.imwrite(new_filename, cropped_image)

    return(x,y,x+width,y+height)


for i in real:
    print(i)
    filename = os.path.join(real_directory,i)

    new_filename = os.path.join(real_crop_directory,i)

    detector = MTCNN()
    # detect faces in the image
    img = cv2.cvtColor(cv2.imread(filename), cv2.COLOR_BGR2RGB)

    faces = detector.detect_faces(img)
    # display faces on the original image
    draw_image_with_boxes(filename, faces,new_filename)


for i in fake:
    print(i)
    filename = os.path.join(fake_directory, i)

    new_filename = os.path.join(fake_crop_directory, i)

    detector = MTCNN()
    # detect faces in the image
    img = cv2.cvtColor(cv2.imread(filename), cv2.COLOR_BGR2RGB)

    faces = detector.detect_faces(img)
    # display faces on the original image
    draw_image_with_boxes(filename, faces, new_filename)
