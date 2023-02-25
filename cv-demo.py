import cv2

def detect_face(frame):
    face_model = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_model.detectMultiScale(gray, 1.1, 4)
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
    return frame


def detect_eyes(frame, type="lt"):
    if (type == "rt"):
        eye_model = cv2.CascadeClassifier('right_eye.xml')
    else:
        eye_model = cv2.CascadeClassifier('left_eye.xml')

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    eyes = eye_model.detectMultiScale(gray, 2, 4)
    for (x, y, w, h) in eyes:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
    return frame

def detect_all(frame):
    lt_eye_model = cv2.CascadeClassifier('left_eye.xml')
    rt_eye_model = cv2.CascadeClassifier('right_eye.xml')
    face_model = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_model.detectMultiScale(gray, 1.5, 4)
    lt_eye = lt_eye_model.detectMultiScale(gray, 2.5, 2)
    rt_eye = rt_eye_model.detectMultiScale(gray, 2.5, 2)
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
    for (x, y, w, h) in lt_eye:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
    for (x, y, w, h) in rt_eye:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
    cv2.putText(frame, '{} person, {} eyes'.format(len(faces), len(lt_eye)), (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA) 
    return frame

vid = cv2.VideoCapture(0)
while(1):
    ret, frame = vid.read()
    frame = cv2.flip(frame, 1)
    cv2.imshow('frame', detect_all(frame))

    if cv2.waitKey(1) & 0xff == ord('q'):
        break

vid.release()
cv2.destroyAllWindows()

