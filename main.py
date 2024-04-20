# Imports
from utils import *
import cv2

# Constants
WIDTH, HEIGHT = 360, 240
PID = [0.4, 0.4, 0]

# Variables
p_error = 0
start_counter = 0  # 0 if not flying else 1

drone = initialize_tello()

while True:

    # start flight if not already flying
    if start_counter == 0:
        drone.takeoff()
        start_counter = 1

    # Track face and follow it
    image = tello_get_frame(drone, WIDTH, HEIGHT)
    image, info = find_face(image)
    pError = track_face(drone, info, WIDTH, PID, p_error)
    cv2.imshow('Image', image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        drone.land()
        break
