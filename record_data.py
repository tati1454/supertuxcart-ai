import argparse
import base64
import json
import sys
import time
from io import BytesIO

import keyboard
import pyscreenshot as ImageGrab

def take_screenshot():
    image = ImageGrab.grab().convert("L").resize((256, 144))
    stream = BytesIO()
    image.save(stream, format="JPEG")
    return base64.b64encode(stream.getvalue()).decode()

def check_keypresses():
    out = []

    if keyboard.is_pressed('down'):
        out.append('down')

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
    parser.add_argument('output_file', metavar='output', type=str, nargs=1, help='Output filename')
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
        print("Quitting...")
        with open(args.output_file[0], "w") as output_file:
            output_file.write(json.dumps(data))
