import cv2
import mediapipe as mp
from numpy import interp
from math import sqrt, pow
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
from ctypes import cast, POINTER

# Initialize Mediapipe and Pycaw
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hand = mp.solutions.hands

minvalue = 30
maxvalue = 0

handmodel = mp_hand.Hands(
    model_complexity=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)

devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(
    IAudioEndpointVolume._iid_, CLSCTX_ALL, None
)
volume = cast(interface, POINTER(IAudioEndpointVolume))
minvolume, maxvolume, _ = volume.GetVolumeRange()

# Hand detection function
def detect_hand(image):
    try:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = handmodel.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        return image, results
    except Exception as e:
        print(f"Error in detect_hand: {e}")
        return image, None

# Draw landmarks and handle volume control
def draw_landmarks(image, results):
    try:
        global maxvalue, minvalue, minvolume, maxvolume
        if results and results.multi_hand_landmarks:  # Ensure valid results
            for hand_landmarks in results.multi_hand_landmarks:
                l = hand_landmarks.landmark
                h, w, _ = image.shape

                thumb_tip = (round(l[4].x * w), round(l[4].y * h))
                indexfinger_tip = (round(l[8].x * w), round(l[8].y * h))
                thumb2 = (round(l[2].x * w), round(l[2].y * h))
                indexfinger5 = (round(l[5].x * w), round(l[5].y * h))
                
                distance = sqrt(pow(thumb_tip[0] - indexfinger_tip[0], 2) +
                                pow(thumb_tip[1] - indexfinger_tip[1], 2))
                distance2 = sqrt(pow(thumb2[0] - indexfinger5[0], 2) +
                                 pow(thumb2[1] - indexfinger5[1], 2))
                distance, distance2 = round(distance), round(distance2)

                if maxvalue == 0:
                    maxvalue = distance2 * 2
                if abs(distance - 2 * distance2) < 10:
                    maxvalue = distance2 * 2

                volumetoset = interp(distance, [minvalue, maxvalue], [minvolume, maxvolume])
                volume.SetMasterVolumeLevel(volumetoset, None)


                cv2.putText(image, str(distance), (0, 20), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0), 2)
                cv2.line(image, indexfinger_tip, thumb_tip, (255, 0, 0), 2)
                cv2.putText(image, str(distance2 * 2), (0, 60), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0), 2)
                
                mp_drawing.draw_landmarks(
                    image,
                    hand_landmarks,
                    mp_hand.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style()
                )
    except Exception as e:
        print(f"Error in draw_landmarks: {e}")

# Detect thumbs-up gesture
def detect_thumbs_up(results, h, w):
    if results and results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            l = hand_landmarks.landmark
            thumb_tip = l[4]
            thumb_mcp = l[2]
            if thumb_tip.y < thumb_mcp.y:
                folded_fingers = all(l[i].y > l[i - 2].y for i in [8, 12, 16, 20])
                if folded_fingers:
                    return True
    return False

# Detect thumbs-down gesture
def detect_thumbs_down(results, h, w):
    if results and results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            l = hand_landmarks.landmark
            thumb_tip = l[4]
            thumb_mcp = l[2]
            if thumb_tip.y > thumb_mcp.y:
                folded_fingers = all(l[i].y > l[i - 2].y for i in [8, 12, 16, 20])
                if folded_fingers:
                    return True
    return False

# Main program
webcam = cv2.VideoCapture(0)

try:
    # Process webcam feed in the background
    print("Waiting for thumbs-up gesture...")
    while True:
        ret, frame = webcam.read()
        if not ret:
            print("Failed to capture frame. Exiting...")
            break

        frame, results = detect_hand(frame)
        h, w, _ = frame.shape
        if detect_thumbs_up(results, h, w):
            print("Thumbs-up gesture detected. Starting volume control...")
            break

    # Main loop for volume control after thumbs-up is detected
    while True:
        ret, frame = webcam.read()
        if not ret:
            print("Failed to capture frame. Exiting...")
            break

        frame, results = detect_hand(frame)
        draw_landmarks(frame, results)

        h, w, _ = frame.shape
        if detect_thumbs_down(results, h, w):
            print("Thumbs-down gesture detected. Exiting...")
            break

        cv2.imshow("WEBCAM", frame)
        if cv2.waitKey(1) & 0xFF == 27:  # ESC key to exit
            break
except Exception as e:
    print(f"Error: {e}")
finally:
    webcam.release()
    cv2.destroyAllWindows()
