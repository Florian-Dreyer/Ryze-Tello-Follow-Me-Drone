from djitellopy import Tello
import cv2
import numpy as np


def initialize_tello():
    """
    Start and initialize the drone
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


def tello_get_frame(drone: Tello, width: int = 360, height: int = 240) -> cv2.UMat:
    """
    Get frame from drone, resize it and return it as cv2.UMat

    :param drone: Ryze Tello drone object
    :param width: width for resizing the image
    :param height: height for resizing the image
    :return: resized image as cv2.UMat
    """
    frame = drone.get_frame_read()
    frame = frame.frame
    image = cv2.resize(frame, (width, height))
    return image


def find_face(image: cv2.UMat) -> tuple:
    """
    Detects face on image and returns the position of it

    :param image: image to find face on
    :return: image, coordinates for the middle of the detected face
    """
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(image_gray, 1.1, 6)

    my_face_list_c = []
    my_face_list_area = []

    for (x, y, width, height) in faces:
        cv2.rectangle(image, (x, y), (x + width, y + height), (0, 0, 255), 2)
        c_x = x + width // 2
        c_y = y + height // 2
        area = width * height
        my_face_list_area.append(area)
        my_face_list_c.append([c_x, c_y])

    if len(my_face_list_area) != 0:
        i = my_face_list_area.index(max(my_face_list_area))
        return image, [my_face_list_c[i], my_face_list_area[i]]
    else:
        return image, [[0, 0], 0]


def track_face(drone: Tello, info, width: int, pid, p_error) -> int:
    """
    Control drone to follow face using pid controller

    :param drone: Ryze Tello drone object
    :param info: list with coordinates for the middle point and area of face
    :param width: width of image
    :param pid: information for pid controller
    :param p_error: previous error
    :return: the current error, use it in next iteration as p_error
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
