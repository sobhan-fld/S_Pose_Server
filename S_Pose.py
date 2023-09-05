import mediapipe as mp
import cv2
import Angles_cal as angle
import gc
import time

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

pose = mp_pose.Pose(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
    model_complexity=1,
)

# variables
count_butterfly = 0
is_count_butterfly = False

count_squat = 0
is_count_squat = False
last_squat_time = None
prev_time = None

count_curling = 0
is_count_curling = False

count_legtrain = 0
is_count_legtrain = False
is_count_legtrain_1 = False
count_legtrain_1 = 0
is_count_legtrain_2 = False
count_legtrain_2 = 0


def butterfly(frame):
    global count_butterfly, is_count_butterfly
    # Convert the BGR frame to RGB for Mediapipe processing
    imgRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame with Mediapipe
    results = pose.process(imgRGB)

    # Draw pose landmarks on the frame
    mp_drawing.draw_landmarks(
        frame,
        results.pose_landmarks,
        mp_pose.POSE_CONNECTIONS,
        mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
        mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2)
    )

    right_elbow_angle, left_elbow_angle, right_arm_angle, left_arm_angle = angle.elbow_shoulder(results, mp_pose)

    # (Your original code does not provide details on how you manage these angles, so I am just drawing them)
    if right_elbow_angle < 110:
        warn1 = "keep your right hand straight"
        cv2.putText(frame, warn1, (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    if left_elbow_angle < 110:
        warn2 = "keep your left hand straight"
        cv2.putText(frame, warn2, (10, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    # Exercise counter
    if right_arm_angle > 25 and left_arm_angle > 25 and right_elbow_angle > 110 and left_elbow_angle > 110:
        if not is_count_butterfly:
            count_butterfly += 1
            is_count_butterfly = True
    else:
        is_count_butterfly = False

    angles = {
        "right_elbow_angle": right_elbow_angle,
        "left_elbow_angle": left_elbow_angle,
        "right_arm_angle": right_arm_angle,
        "left_arm_angle": left_arm_angle,
        "count": count_butterfly
    }

    count_frames = 1
    if count_frames >= 50:
        gc.collect()
    return frame, angles

def squat(frame):
    global count_squat, is_count_squat, last_squat_time
    # Convert the BGR frame to RGB for Mediapipe processing
    imgRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame with Mediapipe
    results = pose.process(imgRGB)

    # Draw pose landmarks on the frame
    mp_drawing.draw_landmarks(
        frame,
        results.pose_landmarks,
        mp_pose.POSE_CONNECTIONS,
        mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
        mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2)
    )

    right_knee_angle, left_knee_angle = angle.knee_angle(results, mp_pose)
    waist_angle = angle.waist(results, mp_pose)

    curr_time = time.time()
    prev_time = curr_time

    # Exercise counter
    if right_knee_angle < 140 and left_knee_angle < 140 and waist_angle < 110:
        if not is_count_squat:
            count_squat += 1
            is_count_squat = True
            if last_squat_time is not None:
                time_between_squats = curr_time - last_squat_time
                # print(f"Time between squats: {time_between_squats:.2f} seconds")

            last_squat_time = curr_time  # Update the last squat time
    else:
        is_count = False


    angles = {
        "right_knee_angle": right_knee_angle,
        "left_knee_angle": left_knee_angle,
        "waist_angle": waist_angle,
        "count": count_squat
    }

    count_frames = 1
    if count_frames >= 50:
        gc.collect()
    return frame, angles


def curling(frame):
    global count_curling, is_count_curling
    # Convert the BGR frame to RGB for Mediapipe processing
    imgRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame with Mediapipe
    results = pose.process(imgRGB)

    # Draw pose landmarks on the frame
    mp_drawing.draw_landmarks(
        frame,
        results.pose_landmarks,
        mp_pose.POSE_CONNECTIONS,
        mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
        mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2)
    )

    right_elbow_angle, left_elbow_angle, right_arm_angle, left_arm_angle = angle.elbow_shoulder(results, mp_pose)

    if right_arm_angle > 20:
        warn1 = "keep your right arm lower"
        cv2.putText(frame, warn1, (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    if left_arm_angle > 20:
        warn1 = "keep your left arm lower"
        cv2.putText(frame, warn1, (10, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    # Exercise counter
    if right_elbow_angle < 35 and left_elbow_angle < 35:
        if not is_count_curling:
            count_curling += 1
            is_count_curling = True
    else:
        is_count_curling = False


    angles = {
        "right_elbow_angle": right_elbow_angle,
        "left_elbow_angle": left_elbow_angle,
        "right_arm_angle": right_arm_angle,
        "left_arm_angle": left_arm_angle,
        "count": count_squat
    }

    count_frames = 1
    if count_frames >= 50:
        gc.collect()
    return frame, angles

def legtrain(frame):
    global count_legtrain, is_count_legtrain, is_count_legtrain_1, count_legtrain_1, is_count_legtrain_2, count_legtrain_2
    # Convert the BGR frame to RGB for Mediapipe processing
    imgRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame with Mediapipe
    results = pose.process(imgRGB)

    # Draw pose landmarks on the frame
    mp_drawing.draw_landmarks(
        frame,
        results.pose_landmarks,
        mp_pose.POSE_CONNECTIONS,
        mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
        mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2)
    )

    right_knee_angle, left_knee_angle = angle.knee_angle(results, mp_pose)
    waist_angle = angle.waist(results, mp_pose)

    # Exercise counter
    if right_knee_angle < 90 and left_knee_angle > 120:
        if not is_count_legtrain:
            count_legtrain += 1
            is_count_legtrain = True
    else:
        is_count_legtrain = False

    if left_knee_angle < 90 and right_knee_angle > 120:
        if not is_count_legtrain_1:
            count_legtrain_1 += 1
            is_count_legtrain_1 = True
    else:
        is_count_legtrain_1 = False

    if is_count_legtrain_1 == is_count_legtrain_2:
        if not is_count_legtrain_2:
            count_legtrain_2 += 1
            is_count_legtrain_2 = True
    else:
        is_count_legtrain_2 = False


    angles = {
        "right_knee_angle": right_knee_angle,
        "left_knee_angle": left_knee_angle,
        "waist_angle": waist_angle,
        "count": count_legtrain_2
    }

    count_frames = 1
    if count_frames >= 50:
        gc.collect()
    return frame, angles


def hand(frame):
    mpHands = mp.solutions.hands
    hands = mpHands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5, model_complexity=1)
    mpDraw = mp.solutions.drawing_utils

    imgRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mpDraw.draw_landmarks(frame, hand_landmarks, mpHands.HAND_CONNECTIONS)

    count_frames = 1
    if count_frames >= 50:
        gc.collect()
    return frame