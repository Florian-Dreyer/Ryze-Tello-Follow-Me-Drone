from djitellopy import Tello
import cv2
import numpy as np
from Model_App import ModelApp


def initialize_tello():
    """Start and initialize the drone
    """
    my_drone = Tello()
    my_drone.connect()
    my_drone.for_back_velocity = 0
    my_drone.left_right_velocity = 0
    my_drone.up_down_velocity = 0
    my_drone.yaw_velocity = 0
    my_drone.speed = 0
    print(f"Current battery: {my_drone.get_battery()}")
    my_drone.streamoff()
    my_drone.streamon()
    return my_drone


def tello_get_frame(drone: Tello, width: int = 320, height: int = 320) -> cv2.UMat:
    """Get frame from drone, resize it and return it as cv2.UMat

    Params:
    drone -- Ryze Tello drone object
    width -- width for resizing the image
    height -- height for resizing the image

    Returns:
    resized image as cv2.UMat
    """
    frame = drone.get_frame_read()
    frame = frame.frame
    image = cv2.resize(frame, (width, height))
    return image


def find_face(image: cv2.UMat) -> tuple:
    """Detects face on image and returns the position of it.

    Params:
    image (cv2.UMat): image to find face on.

    Returns:
    image, coordinates for the middle of the detected face and area.
    """
    model = ModelApp()
    pred_box = model.predict_box(image)
    if not pred_box:
        return image, [[0, 0], 0]
    x_0, y_0, x_1, y_1 = pred_box

    face_midpoint = [(x_0 + x_1) / 2, (y_0 + y_1) / 2]
    face_area = [abs(x_0-x_1) * abs(y_0, y_1)]

    return image, [face_midpoint, face_area]


def track_face(drone: Tello, info, width: int, pid, p_error) -> int:
    """Control drone to follow face using pid controller

    Params:
    drone -- Ryze Tello drone object
    info -- list with coordinates for the middle point and area of face
    width -- width of image
    pid -- information for pid controller
    p_error -- previous error

    Returns:
    the current error, use it in next iteration as p_error
    """
    error = info[0][0] - width // 2
    speed = pid[0] * error + pid[1] * (error - p_error)
    speed = int(np.clip(speed, -100, 100))

    print(f"Current speed: {speed}")
    if info[0][0] != 0:
        drone.yaw_velocity = speed
    else:
        drone.for_back_velocity = 0
        drone.left_right_velocity = 0
        drone.up_down_velocity = 0
        drone.yaw_velocity = 0
        error = 0
    if drone.send_rc_control:
        drone.send_rc_control(drone.left_right_velocity, drone.for_back_velocity, drone.up_down_velocity, drone.yaw_velocity)
    return error
