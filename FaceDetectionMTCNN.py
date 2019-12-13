from keras.models import load_model
import cv2
from mtcnn.mtcnn import MTCNN
import os

base_dir = os.path.dirname(__file__)

detector = MTCNN()

count = 0

if not os.path.exists('faces'):
    print("New directory created")
    os.makedirs('faces')

count = 0

for file in os.listdir(base_dir + '/image'):
    image = cv2.imread(base_dir + '/image/' + file)
    result = detector.detect_faces(image)

    for face in result:
        box = face['box']
        img = image[box[1]:(box[1] + box[3]), box[0]:(box[0] + box[2])]
        try:
            cv2.imwrite(base_dir + '/faces/' + str(count) + '.jpg', img)
            print(count)
        except:
            print('Error Saving Image')
        count += 1

# for file in os.listdir(base_dir + '/video'):
#     video = cv2.VideoCapture("{}/video/{}".format(base_dir,file))
#     print(file)
    
#     if(not video.isOpened()): print('Error Opning Video File')

#     while(video.isOpened()):
#         ret, image = video.read()
#         result = detector.detect_faces(image)

#         if ret :
#             for face in result:
#                 print(count)
#                 box = face['box']
#                 img = image[box[1]:(box[1] + box[3]), box[0]:(box[0] + box[2])]
#                 count += 1
#                 try:
#                     cv2.imwrite(base_dir + '/faces/' + str(count) + '.jpg', img)
#                 except:
#                     print('Error Saving Image')

#         else: break