import argparse
import base64
import cv2
from datetime import datetime
import os
import shutil
import time

import numpy as np
import matplotlib.image as mpimg
import tensorflow as tf
import socketio
import eventlet
import eventlet.wsgi
from PIL import Image
from flask import Flask
from io import BytesIO

import gevent
from gevent.threadpool import ThreadPool

from tensorflow.keras.backend import set_learning_phase
from tensorflow.keras.models import load_model

import speech_recognition as sr
r = sr.Recognizer()

def load_image(data_dir, image_file):
    return mpimg.imread(os.path.join(data_dir, image_file.strip()))


def preprocess(image, height, width):
    image = image[60:-25, :, :]
    image = cv2.resize(image, (width, height), cv2.INTER_AREA)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
    image = cv2.GaussianBlur(image, (3, 3), 0)
    return image

sio = socketio.Server() #Async
app = Flask(__name__)

#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

MAX_SPEED = 30
AVG_SPEED = 20
MIN_SPEED = 10

speed_limit = MAX_SPEED

CMD_HISTORY = []
VOICE_CMD = ''

def make_coordinates(image, line_parameters):
    slope, intercept = line_parameters
    y1 = image.shape[0]
    y2 = int(y1 * (3/5))
    x1 = int((y1-intercept) / slope)
    x2 = int((y2-intercept) / slope)
    return np.array([x1, y1, x2, y2])

def average_slope_intercept(image, lines):
    left_fit = []
    right_fit = []
    for line in lines:
        x1, y1, x2, y2 = line.reshape(4)
        parameters = np.polyfit((x1, x2), (y1, y2), 1)
        slope = parameters[0]
        intercept = parameters[1]
        if slope < 0:
            left_fit.append((slope, intercept))
        else:
            right_fit.append((slope, intercept))
    left_fit_average = np.average(left_fit, axis=0)
    right_fit_average = np.average(right_fit, axis=0)
    left_line = make_coordinates(image, left_fit_average)
    right_line = make_coordinates(image, right_fit_average)
    return np.array([left_line, right_line])

def canny(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    return cv2.Canny(blur, 50, 150)

def display_lines(image, lines):
    line_image = np.zeros_like(image)
    if lines is not None:
        for x1, y1, x2, y2 in lines:
            cv2.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 10)
    return line_image

def region_of_interest(image):
    height = image.shape[0]
    
    polygons = np.array([
        [(200, height), (1100, height), (550, 250)]
        ])

    mask = np.zeros_like(image)
    cv2.fillPoly(mask, polygons, 255)
    masked_image = cv2.bitwise_and(image, mask)
    return masked_image

def policy_network(image, autonomous_mode=True):    
    return steering_angle, throttle

@sio.on('telemetry')
def telemetry(sid, data):

    # print(data.keys())

    if data:
        # The current steering angle of the car
        #steering_angle = float(data["steering_angle"])
        # The current speed of the car
        speed = float(data["speed"])
        # The current image from the center camera of the car
        image = Image.open(BytesIO(base64.b64decode(data["image"])))

        try:
            image = np.asarray(image)

            ####
            # canny_image = canny(image)
            # cropped_image = region_of_interest(canny_image)
            # lines = cv2.HoughLinesP(cropped_image, 2, np.pi/180, 100, np.array([]), minLineLength=40, maxLineGap=5)
            # averaged_lines = average_slope_intercept(image, lines)
            # line_image = display_lines(image, averaged_lines)
            # combo_image = cv2.addWeighted(image, 0.8, line_image, 1, 1)

            # image_bgr = cv2.GaussianBlur(combo_image, (3, 3), 0)
            # image_bgr = cv2.cvtColor(combo_image, cv2.COLOR_RGB2BGR)
            # cv2.imshow("Camera", image_bgr)
            # cv2.waitKey(1)
            ###

            image = preprocess(image, 64, 64)  # apply the preprocessing
            image = np.array([image])  # the model expects 4D array

            # predict the steering angle for the image
            global VOICE_CMD
            global speed_limit, image_count, base_time
            EXEC_FIN = False

            # Replace with instruction parser module
            # Call instruction parser
            # INSTR_CMD = ''
            # if VOICE_CMD != '':
            #     # Run command parser
            #     if VOICE_CMD == 'stop':
            #         INSTR_CMD = 'stop'
            #     elif VOICE_CMD == 'go':
            #         INSTR_CMD = 'go'

            if 'stop' in VOICE_CMD:
                steering_angle = float(model.predict(image, batch_size=1))
                
                if speed > 0:
                    throttle = 0
                else:
                    throttle = 0

            else:
                steering_angle = float(model.predict(image, batch_size=1))

                if speed > speed_limit:
                    speed_limit = MIN_SPEED  # slow down
                else:
                    speed_limit = MAX_SPEED

                throttle = 1.0 - (steering_angle) ** 2 - (speed / speed_limit) ** 2

            # print('Steering Angle: ', steering_angle)
            # print('Throttle: ', throttle)
            # print('Speed: ', speed)
            send_control(steering_angle, throttle, 500)

        except Exception as e:
            print(e)

    else:
        sio.emit('manual', data={}, skip_sid=True)

import time
def audio_in():

    while True:
        global VOICE_CMD

        print('Ready to interact ...')
        with sr.Microphone() as source:                
            audio = r.listen(source)           

        try:
            parsed_aud = str(r.recognize(audio))

            if parsed_aud:

                VOICE_CMD = str(r.recognize(audio))
                CMD_HISTORY.append(VOICE_CMD)

                time.sleep(10)

        except LookupError: # speech is unintelligible
            print('Audio is too bad')

        time.sleep(0.5)

@sio.on('connect')
def connect(sid, environ):
    print("connect ", sid)

    pool = ThreadPool(3)
    pool.spawn(audio_in)

    send_control(0, 0, 0)

def send_control(steering_angle, throttle, reverse):

    sio.emit(
        "steer",
        data={
            'steering_angle': steering_angle.__str__(),
            'throttle': throttle.__str__(),
            'brakes': reverse.__str__()
        },
        skip_sid=True)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Remote Driving')
    parser.add_argument(
        'model',
        type=str,
        help='Path to model h5 file. Model should be on the same path.'
    )
    args = parser.parse_args()

    #enable_xla()
    set_learning_phase(0)
    model = load_model(args.model)

    # wrap Flask application with engineio's middleware
    app = socketio.Middleware(sio, app)

    # deploy as an eventlet WSGI server
    eventlet.wsgi.server(eventlet.listen(('', 4567)), app)

