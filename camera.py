import time
import cv2
import mediapipe as mp
import math
# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils
i = 1
counter_run = False
counter = 5
start_time = 0
def calculate_distance(point1, point2):
    # Calculate the Euclidean distance between two points
    return math.sqrt((point2.x - point1.x) ** 2 + (point2.y - point1.y) ** 2 + (point2.z - point1.z) ** 2)

def calculate_angle(a, b, c, d, e):
    radians1 = math.atan2(e.y - d.y, e.x - d.x) - math.atan2(c.y - b.y, c.x - b.x)
    radians2 = math.atan2(c.y - b.y, c.x - b.x) - math.atan2(a.y - b.y, a.x - b.x)
    angle1 = math.fabs(radians1 * 180.0 / math.pi)
    angle2 = math.fabs(radians2 * 180.0 / math.pi)
    return angle1, angle2

class VideoCamera:
    def __init__(self):
        # Open the camera
        self.cap = cv2.VideoCapture(0)

    def __del__(self):
        # Release the camera
        self.cap.release()

    def get_frame(self):
        # Read frame from the camera
        success, frame = self.cap.read()
        if not success:
            return b''
        
        global i
        global counter_run
        global counter
        global start_time
        # Convert the image to RGB
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process the image with MediaPipe Hands
        results = hands.process(image_rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Draw the landmarks on the image
                # mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                # the index finger
                index_finger = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                # the middle finger
                middle_finger = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
                # Get the coordinates of the fingertips and palm
                thumb = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
                ring_finger = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP]
                pinky = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]
                angle1, angle2 = calculate_angle(thumb, index_finger, middle_finger, ring_finger, pinky)
                # Get the center coordinates of the hand
                center_x = int(hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP].x * frame.shape[1])
                center_y = int(hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP].y * frame.shape[0])
                is_closed = angle1 < 90 and angle2 < 300
                # if the hand is closed
                if is_closed:
                    # add circle at the center
                    cv2.circle(frame, (center_x, center_y), 10, (0, 0, 255), -1)
                    cv2.putText(frame, "closed", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                # line spec
                image_height, image_width, _ = frame.shape
                start_point = (int(index_finger.x * image_width), int(index_finger.y * image_height))
                end_point = (int(middle_finger.x * image_width), int(middle_finger.y * image_height))
                # draws the line
                cv2.line(frame, start_point, end_point, None, None)
                distance = calculate_distance(index_finger, middle_finger)
                # detect yoyo code & run the counter
            if distance > 0.16 and not counter_run:
                counter_run = True
                start_time = time.time()
                is_closed = False
                # if the counter is running
        if counter_run:
            elapsed_time = time.time() - start_time

            if elapsed_time >= 1:
                counter -= 1
                start_time = time.time()
            #capturing image after counting is done
            if counter == 0:
                print("Image saved successfully")
                #name the file!
                cv2.imwrite(f'Docs/capimg{i}.jpg', frame)
                i += 1
                counter = 5
                # counter is false
                counter_run = False
            # displaying the counter
            cv2.putText(frame, f"Capturing image in {counter}...", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        ret, jpeg = cv2.imencode('.jpg', frame)
        return jpeg.tobytes()