import os
import numpy as np
import cv2
from keras.models import load_model
from mtcnn.mtcnn import MTCNN
import tensorflow as tf
from numpy import load
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import Normalizer
from sklearn.svm import SVC

base_dir = os.path.dirname(__file__)

class FaceRecognition:
    def __init__(self):
        self.detector = MTCNN()
        self.model = load_model(base_dir + '/facenet/model/facenet_keras.h5')
        print('Detector and Model Loaded')
    
    def detectFaces(self, frame :np.ndarray) -> np.ndarray:
        faces = self.detector.detect_faces(frame)
        # x1 x2 y1 y2
        return np.array([ [ face['box'][0], ( face['box'][0] + face['box'][2] ), face['box'][1], ( face['box'][1] + face['box'][3] ) ] for face in faces ], dtype=int)
    
    def makeReactangleFaces(self, frame :np.ndarray, rect :np.ndarray):
        cv2.rectangle(frame, ( int(rect[0]), int(rect[2]) ), ( int(rect[1]), int(rect[3]) ), ( 0, 155, 255 ), 2)
    
    def getFaces(self, frame :np.ndarray, rect :np.ndarray) -> list:
        return [ frame[ r[2]:r[3] , r[0]:r[1] ] for r in rect ]

    def saveFaceFromFrame(self, path :str, frame :np.ndarray):
        cv2.imwrite(path, frame)
    
    def embedding(self, image :np.ndarray):
        mean, std = image.mean(), image.std()
        image = (image - mean) / std
        image = np.expand_dims(image, axis=0)
        image = self.model.predict(image)
        return image[0]
    
    def loadDatasetAndEmbedding(self, path :str, filename :str):
        print('Load Dataset, Emmbedding and Write to File')

        X, Y = [], []

        for label in os.listdir(path):
            for file in os.listdir(path + label + '/'):
                image = cv2.imread(path + label + '/' + file)
                image = cv2.resize(image, (160, 160), interpolation=cv2.INTER_AREA)
                image = self.embedding(image)

                X.append( image )
                Y.append( label )
        
        if not os.path.exists('CompressedImages'): os.makedirs('CompressedImages')
        
        print('Compressed File saved as {}'.format(base_dir + '/CompressedImages/' + filename))
        np.savez_compressed(base_dir + '/CompressedImages/' + filename, X=np.asarray(X), Y=np.asarray(Y))
    
    def embeddingDataTest(self, frame :np.ndarray):
        faces = self.getFaces(frame, self.detectFaces(frame))
        facesFeatures = []

        for f in faces:
            f = cv2.resize(f, (160, 160), interpolation=cv2.INTER_AREA)
            facesFeatures.append( self.embedding(f) )
        
        return np.asarray(facesFeatures)

    def predict(self):
        data = np.load(base_dir + '/CompressedImages/dataset.npz')
        X_Train, Y_Train = data['X'], data['Y']

        print(Y_Train)

        # TestData
        testData = cv2.imread('yoga.jpg')
        testData = self.embeddingDataTest(testData)

        for X_Test in testData:
            in_encoder = Normalizer(norm='l2')
            X_Train = in_encoder.transform(X_Train)
            X_Test = in_encoder.transform(X_Test.reshape(1, -1))

            out_encoder = LabelEncoder()
            out_encoder.fit(Y_Train)
            Y_Train = out_encoder.transform(Y_Train)

            model = SVC(kernel='linear', probability=True)
            model.fit(X_Train, Y_Train)

            Y_Pred = model.predict(X_Test)
            print(out_encoder.inverse_transform(Y_Pred))

        

def main():
    a = FaceRecognition()
    a.predict()

if __name__ == "__main__":
    main()

    # Kurang Train Model, Deteksi, dll
