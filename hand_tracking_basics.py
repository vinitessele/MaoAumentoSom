# pip install opencv-python
# pip install mediapipe

import cv2
import mediapipe as mp
import pyautogui
import time     # check framerate

capture = cv2.VideoCapture(0)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands()    # podemos só ignorar os argumentos, mas é bom conhece-los
mp_draw = mp.solutions.drawing_utils

# coletando o framerate
previous_time = 0
current_time = 0

''' Hands()
def __init__(self,
               static_image_mode=False,
               max_num_hands=2,
               model_complexity=1,
               min_detection_confidence=0.5,
               min_tracking_confidence=0.5):
Args:
    static_image_mode: Whether to treat the input images as a batch of static
        and possibly unrelated images, or a video stream.
    max_num_hands: Maximum number of hands to detect.

    model_complexity: Complexity of the hand landmark model: 0 or 1.
    Landmark accuracy as well as inference latency generally go up with the
    model complexity.

    min_detection_confidence: Minimum confidence value ([0.0, 1.0]) for hand
    detection to be considered successful.

    min_tracking_confidence: Minimum confidence value ([0.0, 1.0]) for the
    hand landmarks to be considered tracked successfully.
    """
'''

wait_command = 0
check_hands = True


while True:
    success, img = capture.read()

    # aqui temos que pegar a nossa imagem em rgb, pq a nossa classe só usa objetos RGB
    img_RGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    results = hands.process(img_RGB)
    # print(results) -> <class 'mediapipe.python.solution_base.SolutionsOutputs'>, ou seja, temos uma classe (que tem atributos)
    # print(results.multi_hand_landmarks)

    if results.multi_hand_landmarks:
        for hand in results.multi_hand_landmarks:

            for id, lm in enumerate(hand.landmark):
                # print(id, lm)   # nos retorna as coordenadas em casas decimais, mas queremos em px. Pra isso faremos uma conta

                height, width, channels = img.shape
                center_x, center_y = int(lm.x*width), int(lm.y*height) 

                # print(id, center_x, center_y)
                # ideia basica de alterar/analisar alguma junta
                # if id == 8:
                #     cv2.circle(img, (center_x, center_y), 10, (255, 255, 0), cv2.FILLED)

                # exemplo 1: areas de trabalho
                wait_time = 1
                able_to_procced = wait_command - time.time() < wait_time

                if able_to_procced:
                    check_hands = True
                else:
                    check_hands = False

                # if id == 8: print(check_hands)

                if id == 8 and check_hands:
                    if center_x > .9*width:
                        pyautogui.hotkey("ctrl", "win", "left")
                        wait_command = time.time() + 2*wait_time
                    elif center_x < .1*width:
                        pyautogui.hotkey("ctrl", "win", "right")
                        wait_command = time.time() + 2*wait_time


            # mp_draw.draw_landmarks(img, hand) # executar primeiro assim
            mp_draw.draw_landmarks(img, hand, mp_hands.HAND_CONNECTIONS)

    current_time = time.time()
    fps = 1/(current_time - previous_time)
    previous_time = current_time

    cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_DUPLEX, 2, (255,0,255), 3)
    # cv2.putText(material, texto, localização, fonte, fontScale, cor, thickness)

    cv2.imshow("Image", img)
    cv2.waitKey(1)