import cv2
import numpy as np
import os

########## KNN CODE ############
def distance(v1, v2):
    # Eucledian
    return np.sqrt(((v1-v2)**2).sum())

def knn(train, test, k=5):
    dist = []
    print(test.shape)
    for i in range(train.shape[0]):
        # Get the vector and label
        ix = train[i, :-1]
        iy = train[i, -1]
        # Compute the distance from test point
        d = distance(test, ix)
        dist.append([d, iy])
    # Sort based on distance and get top k
    dk = sorted(dist, key=lambda x: x[0])[:k]
    # Retrieve only the labels
    label = np.array(dk)[:, -1]

    # Get frequencies of each label
    output = np.unique(label, return_counts=True)
    # Find max frequency and corresponding label
    index = np.argmax(output[1])
    return output[0][index]
################################

#Init Camera
cap = cv2.VideoCapture(0)

# Face Detection
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")

skip = 0
dataset_path = './data/'

face_data = []    # X data set
labels = []       # Y data set

class_id = 0
names = {}

# data preparation
for file in os.listdir(dataset_path):
    if file.endswith('npy'):
        # create mapping of class_id and names
        names[class_id] = file[:-4]

        data_item = np.load(dataset_path+file)
        print('loaded ', file)
        face_data.append(data_item)

        # create labels for the files or Y data set
        target = class_id*np.ones((data_item.shape[0],))
        class_id += 1
        labels.append(target)

face_dataset = np.concatenate(face_data, axis=0)     # combining all npy faces files to make X_train
face_labels = np.concatenate(labels, axis=0).reshape((-1, 1))       # combining all labels to make Y_train
trainset = np.concatenate((face_dataset, face_labels), axis=1)     # Appending labels with corresponding users along the column

# testing for new face
while True:
    ret, frame = cap.read()
    if not ret:
        continue

    faces = face_cascade.detectMultiScale(frame, 1.3, 5)
    if len(faces) == 0:
        continue

    for face in faces:
        x, y, w, h = face

        # Get the face ROI
        offset = 10
        face_section = frame[y-offset:y+h+offset, x-offset:x+w+offset]
        face_section = cv2.resize(face_section, (100, 100))

        # Predicted Label (out)
        face_section = face_section.flatten()
        print(face_section.shape)
        out = knn(trainset, face_section[:10000])

        # Display on the screen the name and rectangle around it
        pred_name = names[int(out)]
        cv2.putText(frame, pred_name, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 255), 2)

    cv2.imshow("Faces", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()



