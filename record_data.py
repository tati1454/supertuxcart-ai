import argparse
import base64
import json
import sys
import os
import time
import random
from io import BytesIO

import keyboard
from PIL import ImageGrab

def take_screenshot():
    image = ImageGrab.grab().convert("L").resize((256, 144))
    stream = BytesIO()
    image.save(stream, format="JPEG")
    return base64.b64encode(stream.getvalue()).decode()

def check_keypresses():
    out = []

    if keyboard.is_pressed('left'):
        out.append('left')
    
    if keyboard.is_pressed('right'):
        out.append('right')

    if keyboard.is_pressed('down'):
        out.append('down')

    if keyboard.is_pressed('up'):
        out.append('up')

    return out

def get_sample(pressed_keys): 
    expected = [0, 0, 0, 0]

    if 'left' in pressed_keys:
        expected[0] = 1
    if 'right' in pressed_keys:
        expected[1] = 1
    if 'up' in pressed_keys:
        expected[2] = 1
    if 'down' in pressed_keys:
        expected[3] = 1

    return {"frame": take_screenshot(), "expected": expected}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Record data for training")
    parser.add_argument('dataset_folder', metavar='output', type=str, nargs=1, help='Output folder')
    parser.add_argument('--test-data-amount', type=int, nargs=1, required=True, help='Specify the porcentage of test data the dataset will have')
    args = parser.parse_args()

    data = []

    try:
        time.sleep(3)
        print("Capturing data...")
        while True:
            pressed_keys = check_keypresses()

            sample = get_sample(pressed_keys)
            if sample["expected"] == [0, 0, 0, 0]:
                continue
        
            data.append(sample)
            time.sleep(0.1)

    except KeyboardInterrupt:
        print("Saving dataset...")
        random.shuffle(data)

        testing_data = data[:round(len(data) * (args.test_data_amount[0] / 100))]
        training_data = data[round(len(data) * (args.test_data_amount[0] / 100)):]

        if not os.path.exists(args.dataset_folder[0]):
            os.mkdir(args.dataset_folder[0])

        if not os.path.exists(args.dataset_folder[0] + "/training"):
            os.mkdir(args.dataset_folder[0] + "/training")

        if not os.path.exists(args.dataset_folder[0] + "/testing"):
            os.mkdir(args.dataset_folder[0] + "/testing")

        training_capture_filename = None
        testing_capture_filename = None

        i = 1
        while os.path.exists(args.dataset_folder[0] + f"/training/capture{i}.json"):
            i += 1
        training_capture_filename = args.dataset_folder[0] + f"/training/capture{i}.json"

        i = 1
        while os.path.exists(args.dataset_folder[0] + f"/testing/capture{i}.json"):
            i += 1
        testing_capture_filename = args.dataset_folder[0] + f"/testing/capture{i}.json"

        with open(training_capture_filename, "w") as training_capture_file:
            json.dump(training_data, training_capture_file)

        with open(testing_capture_filename, "w") as testing_capture_file:
            json.dump(testing_data, testing_capture_file)
