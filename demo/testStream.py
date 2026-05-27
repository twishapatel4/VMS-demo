import cv2
 
STREAM_URL = "http://192.168.29.97:5000/video"
 
cap = cv2.VideoCapture(STREAM_URL)
 
if not cap.isOpened():

    print("Failed to connect to stream")

    exit()
 
while True:

    ret, frame = cap.read()
 
    if not ret:

        print("Failed to receive frame")

        break
 
    cv2.imshow("Laptop 1 Webcam Stream", frame)
 
    # Press q to quit

    if cv2.waitKey(1) & 0xFF == ord('q'):

        break
 
cap.release()

cv2.destroyAllWindows()
 