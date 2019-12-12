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

    def train(self, path :str):
        data = np.load(path)
        self.X_Train, self.Y_Train = data['X'], data['Y']

        in_encoder = Normalizer(norm='l2')
        self.X_Train = in_encoder.transform(self.X_Train)
        self.X_Test = in_encoder.transform(self.X_Test.reshape(1, -1))

        self.out_encoder = LabelEncoder()
        self.out_encoder.fit(self.Y_Train)
        self.Y_Train = self.out_encoder.transform(self.Y_Train)

        self.SVCModel = SVC(kernel='linear', probability=True)
        self.SVCModel.fit(self.X_Train, self.Y_Train)

    def runCamera(self):
        cap = cv2.VideoCapture(0)

        if (cap.isOpened()== False): print("Error opening video stream or file")

        while(cap.isOpened()):
            ret, frame = cap.read()
            if(ret):
                rectFaces = self.detectFaces(frame)
                self.makeReactangleFaces(frame, rectFaces)

                cv2.imshow('Frame',frame)
                if cv2.waitKey(25) & 0xFF == ord('q'): break
            else: break

        cap.release()
        cv2.destroyAllWindows()

def main():
    a = FaceRecognition()
    a.train(base_dir + '/CompressedImages/dataset.npz')
    a.runCamera()

if __name__ == "__main__":
    main()

    # Kurang Train Model, Deteksi, dll
