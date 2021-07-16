import cv2
import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()


cap = cv2.VideoCapture(0)

while cap.isOpened():
    success,frame = cap.read()

    if not success:
        print("camera not accessable")
        break

    result = hands.process(frame)
    print(result)
    for landmark in result.multi_hand_landmarks:
        print('hand_landmarks:', landmark)


        mp_drawing.draw_landmarks(frame,landmark, mp_hands.HAND_CONNECTIONS)


    cv2.imshow("frame",frame)

    k = cv2.waitKey(10)
    if k & 0xff == ord('q'):
        break



cap.release()
cv2.destroyAllWindows()