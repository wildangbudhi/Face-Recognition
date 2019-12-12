import os
import numpy as np
import cv2
from keras.models import load_model
from mtcnn.mtcnn import MTCNN
import tensorflow as tf

base_dir = os.path.dirname(__file__)

class FaceRecognition:
    def __init__(self):
        self.detector = MTCNN()
        self.model = load_model(base_dir + '/facenet/model/facenet_keras.h5')
    
    def detectFaces(self, frame :np.ndarray) -> np.ndarray:
        faces = self.detector.detect_faces(frame)
        # x1 x2 y1 y2
        return np.array([ [ face['box'][0], ( face['box'][0] + face['box'][2] ), face['box'][1], ( face['box'][1] + face['box'][3] ) ] for face in faces ], dtype=float)
    
    def makeReactangleFaces(self, frame :np.ndarray, rect :np.ndarray):
        cv2.rectangle(frame, ( rect[0], rect[1] ), ( rect[2], rect[3] ), ( 0, 155, 255 ), 2)
    
    def saveFaceFromFrame(self, path :str, frame :np.ndarray):
        cv2.imwrite(path, frame)
    
    def loadDatasetAndCompress(self, path :str, filename :str):
        print('Load Dataset and Compressing')

        X, Y = [], []

        for label in os.listdir(path):
            for file in os.listdir(path + label + '/'):
                X.append( cv2.imread(path + label + '/' + file) )
                Y.append( label )
        
        if not os.path.exists('CompressedImages'): os.makedirs('CompressedImages')
        
        print('Compressed File saved as {}'.format(base_dir + '/CompressedImages/' + filename))
        np.savez_compressed(base_dir + '/CompressedImages/' + filename, X=np.asarray(X), Y=np.asarray(Y))


with tf.device('/CPU:0'):
    a = FaceRecognition()
    a.loadDatasetAndCompress(base_dir + '/images/Labeled/', 'dataset.npz')

    # Kurang Train Model, Deteksi, dll
