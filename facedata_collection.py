import cv2
import numpy as np

# init camera
cap = cv2.VideoCapture(0)

# face detection
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")

skip = 0
face_data = []
dataset_path = './data/'
file_name = input('Enter the name of person : ')
# we will catch frames from video until user presses 'q'
while True:
    ret, frame = cap.read()
    if not ret:
        continue
    # convert the frame to gray frame for sake of simplicity
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray_frame, 1.3, 5)
    if len(faces) == 0:
        continue
    faces = sorted(faces, key=lambda x: x[2]*x[3])
    for face in faces[-1 : ]:
        x,y,w,h = face
        cv2.rectangle(gray_frame,(x,y),(x+w,y+h),(0,255,255), 2)
        # extract the face area
        offset = 10
        face_section = gray_frame[y-offset:y+h+offset, x-offset:x+w+offset]
        face_section = cv2.resize(face_section, (100, 100))
        face_data.append(face_section)
    # cv2.imshow("i love you baby",frame)
    cv2.imshow("gray frame",gray_frame)
    key_pressed = cv2.waitKey(1) & 0xFF
    if key_pressed == ord('q'):
        break

# convert face_data to numpy array and save in File systems
face_data = np.asarray(face_data)
face_data = face_data.reshape((face_data.shape[0], -1))
np.save(dataset_path+file_name, face_data)
print(f'successfully captured your {face_data.shape[0]} images')
print(face_data.shape)
cap.release()
cv2.destroyAllWindows()

