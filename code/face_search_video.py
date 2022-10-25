import cv2
from face_search_image import specifyFace


cap = cv2.VideoCapture(0)

#cap = cv2.VideoCapture(r'data\test\record.mkv')
while True:
    fps = int(cap.get(5))
    frame_count = cap.get(7)
    success, frame = cap.read()

    if success:
        print('da chup')
        #set size of frame
        frame = cv2.resize(frame, (1000, 600))
        specifyFace(frame)
    else:
        break

    #check to quit
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()