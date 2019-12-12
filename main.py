import os
import numpy as np
import cv2
from keras.models import load_model
from mtcnn.mtcnn import MTCNN
import tensorflow as tf
from numpy import load
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import Normalizer
from sklearn.svm import SVC
from datetime import datetime

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
    
    def getFace(self, frame :np.ndarray, rect :np.ndarray) -> np.ndarray:
        return frame[ rect[2]:rect[3] , rect[0]:rect[1] ]

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

    def run(self, path :str):
        data = np.load(path)
        X_Train, Y_Train = data['X'], data['Y']

        in_encoder = Normalizer(norm='l2')
        X_Train = in_encoder.transform(X_Train)

        out_encoder = LabelEncoder()
        out_encoder.fit(Y_Train)
        Y_Train = out_encoder.transform(Y_Train)

        SVCModel = SVC(kernel='linear', probability=True)
        SVCModel.fit(X_Train, Y_Train)

        log_file = open('log.txt', 'a')

        cap = cv2.VideoCapture(0)

        if (cap.isOpened()== False): print("Error opening video stream or file")

        count = 0

        while(cap.isOpened()):
            now = datetime.now()
            ret, frame = cap.read()
            if(ret):
                rectFaces = self.detectFaces(frame)
                for rect in rectFaces:
                    self.makeReactangleFaces(frame, rect)
                    face = self.getFace(frame, rect)
                    face = cv2.resize(face, (160, 160), interpolation=cv2.INTER_AREA)
                    face = self.embedding(face)
                    face = in_encoder.transform(face.reshape(1, -1))
                    Y_Pred = SVCModel.predict(face)
                    Y_Pred = out_encoder.inverse_transform(Y_Pred)[0]
                    print(Y_Pred)

                    if((count % 10000) == 0): 
                        log_file.write(now.strftime("%d/%m/%Y %H:%M:%S") + ' --> ' + Y_Pred + '\n')
                        count = 0
                    
                    count += 1
                    
                    cv2.putText(frame, str(Y_Pred), (rect[0], rect[2] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, ( 0, 155, 255 ) )

                cv2.imshow('Frame',frame)
                if cv2.waitKey(25) & 0xFF == ord('q'): break
            else: break

        log_file.close()
        cap.release()
        cv2.destroyAllWindows()

def main():
    a = FaceRecognition()
    a.run(base_dir + '/CompressedImages/dataset.npz')

if __name__ == "__main__":
    main()