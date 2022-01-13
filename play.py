import sys
import time

from PIL import ImageGrab
import model
import pyautogui
import torch
from torchvision.transforms import ToTensor

def take_screenshot():
    return ImageGrab.grab().convert("L").resize((64, 36))

def predict_action(model: model.TuxDriverModel, image):
    input_image = ToTensor()(image).unsqueeze(0)
    return model(input_image)

if __name__ == "__main__":
    neural_network = model.TuxDriverModel()
    neural_network.load_state_dict(torch.load(sys.argv[1]))
    
    neural_network.eval()
    pressing_left = False
    pressing_right = False
    while True:
        result = predict_action(neural_network, take_screenshot())[0]

        print("\r" + str(result.detach().numpy()) + " Pressing: ", end="")

        if float(result[0]) > 0.5 and pressing_left == False:
            print("left ", end="")
            pyautogui.keyDown("left")
        else:
            if pressing_left == True:
                pyautogui.keyUp("left")
        
        if float(result[1]) > 0.5 and pressing_right == False:
            print("right ", end="")
            pyautogui.keyDown("right")
        else:
            if pressing_right == True:
                pyautogui.keyUp("right")

        # if float(result[2]) > 0.5:
        #     print("up ", end="")
        #     pyautogui.keyDown("up")
        # else:
        #     pyautogui.keyUp("up")

        # if float(result[3]) > 0.5:
        #     print("down ", end="")
        #     pyautogui.keyDown("down")
        # else:
        #     pyautogui.keyUp("down")

        time.sleep(0.1)
