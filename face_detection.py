import cv2
import sys

def facedetection():
    # loading classifier for frontal face
    faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

    # return video from default web cam
    cap = cv2.VideoCapture(0)

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()

        # if there is video
        if ret != 0: 
            # convert to gray scale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) 

            # return rectangle with coordinates (x, y, w, h)
            faces = faceCascade.detectMultiScale(
                gray,
                scaleFactor=1.2,
                minNeighbors=5,
                minSize=(30, 30),
                flags=cv2.CASCADE_SCALE_IMAGE
            )

            # Draw rectangle around faces
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 4) # color order --> BGR

            # Display resulting frame
            cv2.imshow('Video', frame)

            # press 'q' to exit the while loop
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    # Release the capture
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    facedetection()