#Importing Required libraries
import cv2
import numpy as np
import getch
import RPi.GPIO as GPIO
import time
import tensorflow.lite as tflite
# import tflite_runtime.interpreter as tflite
import sections
from PIL import Image
from gtts import gTTS
import os
import pyrebase
import nltk
from nltk import word_tokenize

#Setting up pins for rasperry pi
servoPIN = 17
GPIO.setmode(GPIO.BCM)
GPIO.setup(servoPIN, GPIO.OUT)
p = GPIO.PWM(servoPIN,50) # GPIO 17 for PWM with 50Hz
firebaseConfig = {
  "apiKey": "",
  "authDomain": "",
  "databaseURL": "",
  "projectId": "",
  "storageBucket": "",
  "messagingSenderId": "",
  "appId": "",
  "measurementId": ""
}

firebase=pyrebase.initialize_app(firebaseConfig)

db=firebase.database()
i=0
lst=[]
#This imported module finds which section that the user is in the super market
sections()

#Video feed detects the objects and proceeds to the billing once objects are detected.
cam = cv2.VideoCapture(0)
while True:
    ret, image = cam.read()
    cv2.imshow('Imagetest',image)
    k = cv2.waitKey(1)
    if k != -1:
        break
cv2.imwrite('/home/pi/testimage.jpg', image)
cam.release()
cv2.destroyAllWindows()
    
#Thsese functions are used in the prediciton process using tflite
def load_labels(label_path):
    r"""Returns a list of labels"""
    with open(label_path, 'r') as f:
        return [line.strip() for line in f.readlines()]


def load_model(model_path):
    r"""Load TFLite model, returns a Interpreter instance."""
    interpreter = tflite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    return interpreter


def process_image(interpreter, image, input_index, k=3):
    r"""Process an image, Return top K result in a list of 2-Tuple(confidence_score, label)"""
    input_data = np.expand_dims(image, axis=0)  # expand to 4-dim

    # Process
    interpreter.set_tensor(input_index, input_data)
    interpreter.invoke()

    # Get outputs
    output_details = interpreter.get_output_details()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    print(output_data.shape)  # (1, 1001)
    output_data = np.squeeze(output_data)

    # Get top K result
    top_k = output_data.argsort()[-k:][::-1]  # Top_k index
    result = []
    for i in top_k:
        score = float(output_data[i] / 255.0)
        result.append((i, score))

    return result


def display_result(top_result, frame, labels):
    r"""Display top K result in top right corner"""
    font = cv2.FONT_HERSHEY_SIMPLEX
    size = 0.6
    color = (255, 0, 0)  # Blue color
    thickness = 1

    for idx, (i, score) in enumerate(top_result):
        # print('{} - {:0.4f}'.format(label, score))
        x = 12
        y = 24 * idx + 24
        cv2.putText(frame, '{} - {:0.4f}'.format(labels[i], score),
                    (x, y), font, size, color, thickness)
        print(labels[i])
        global response1
        response1=labels[i]
        break

#     cv2.imshow('Image Classification', frame)

if __name__ == "__main__":

    model_path = '/home/pi/Downloads/model#/model.tflite'
    label_path = '/home/pi/Downloads/model#/labels.txt'  
    image_path = '/home/pi/testimage.jpg'

    interpreter = load_model(model_path)
    labels = load_labels(label_path)

    input_details = interpreter.get_input_details()
    # Get Width and Height
    input_shape = input_details[0]['shape']
    height = input_shape[1]
    width = input_shape[2]

    # Get input index
    input_index = input_details[0]['index']

    frame = cv2.imread(image_path, cv2.IMREAD_COLOR)
    print(frame.shape)

    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    image = image.resize((width, height))

    top_result = process_image(interpreter, image, input_index)
    display_result(top_result, frame, labels)
    

    print(response1)
    response=response1[1:]
    print(response)
    language = 'en'
    file1 = open("myfile.txt","w")
    # L = ["Apple\n"] 
        
    # \n is placed to indicate EOL (End of Line)
    file1.write(response)
    

    # file1.writelines("\n")
    
    file1.close() #to change file access modes
    
    
    #Servo motor works to open a lid so that customers can put the objects inside
    p.start(2.5) # Initialization
    while True:
        p.ChangeDutyCycle(5)
        time.sleep(1)
        p.ChangeDutyCycle(7.5)
        time.sleep(1)
        p.ChangeDutyCycle(10)
        time.sleep(1)
        p.ChangeDutyCycle(12.5)
        time.sleep(1)
        p.ChangeDutyCycle(10)
        time.sleep(1)
        p.ChangeDutyCycle(7.5)
        time.sleep(1)
        p.ChangeDutyCycle(5)
        time.sleep(1)
        p.ChangeDutyCycle(2.5)
        time.sleep(1)
        p.stop()
        GPIO.cleanup()
        break
    
    #This object detection is used to confirm the items that were put in to the cart, and then to proceed to billing process once the shopping is done.
    cam = cv2.VideoCapture(2)
    while True:
        ret, image = cam.read()
        cv2.imshow('Imagetest',image)
        k = cv2.waitKey(1)
        if k != -1:
            break
    cv2.imwrite('/home/pi/testimage1.jpg', image)
    cam.release()
    cv2.destroyAllWindows()

    model_path = '/home/pi/Downloads/model#/model.tflite'
    label_path = '/home/pi/Downloads/model#/labels.txt'  
    image_path = '/home/pi/testimage.jpg'

    interpreter = load_model(model_path)
    labels = load_labels(label_path)

    input_details = interpreter.get_input_details()
    # Get Width and Height
    input_shape = input_details[0]['shape']
    height = input_shape[1]
    width = input_shape[2]

    # Get input index
    input_index = input_details[0]['index']

    frame = cv2.imread(image_path, cv2.IMREAD_COLOR)
    print(frame.shape)

    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    image = image.resize((width, height))

    top_result = process_image(interpreter, image, input_index)
    display_result(top_result, frame, labels)
    

    print(response1)
    response=response1[1:]
    print(response)

    file1 = open("myfile.txt","a")
    file1.writelines(response)
    file1 = open("myfile.txt","r+") 
    text=file1.readlines()
    # print(text)
    # for i in text:
    # text[0].removesuffix('\n')
    print(text)
    l=word_tokenize(text[0])
    if (l[0]==l[1]):
        print("Same")
        res="Successfully added this item"
    else:
        print("Different objects")
    lst.append(response)
    sections()
    print(lst)
    


