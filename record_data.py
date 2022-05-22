import argparse
import json
import os
import time
import random

import keyboard

from screenshot import take_screenshot, get_window_by_name

def check_keypresses():
    out = []

    if keyboard.is_pressed('left'):
        out.append('left')
    
    if keyboard.is_pressed('right'):
        out.append('right')

    return out

def get_sample(game_window, pressed_keys):
    expected = [0, 0, 0]

    if 'left' in pressed_keys:
        expected[0] = 1

    elif 'right' in pressed_keys:
        expected[1] = 1

    else:
        expected[2] = 1

    return {"frame": take_screenshot(game_window, convert_to_base64=True), "expected": expected}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Record data for training")
    parser.add_argument('dataset_folder', metavar='output', type=str, nargs=1, help='Output folder')
    parser.add_argument('--test-data-amount', type=int, nargs=1, required=True, help='Specify the porcentage of test data the dataset will have')
    parser.add_argument('--game-window', type=str, required=False, help='Name of the window to capture, by default it will capture the entire screen')
    parser.add_argument('--only-left-right', action="store_true", default=False, help='Ignore straight samples.')
    args = parser.parse_args()

    data = []

    try:
        time.sleep(3)
        print("Capturing data...")
        while True:
            pressed_keys = check_keypresses()
            game_window = None

            if args.game_window is None:
                game_window = get_window_by_name("root")
            else:
                game_window = get_window_by_name(args.game_window)
                if game_window is None:
                    print(f"Couldn't find window {args.game_window}")
                    exit()
            sample = get_sample(game_window, pressed_keys)
            if args.only_left_right:
                if sample["expected"][2] == 1:
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
