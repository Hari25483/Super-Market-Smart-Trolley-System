# -Super-Market-Smart-Trolley-System
#############################################################
 This project was implemented to remove thefts in stores when adding items to trolley, to provide the location of sections in store, and to generate automated bills that will  be displayed in the app when checking out.
 #     Author : Hariharan Raveenthiran

 #############################################################
 
 Required Items
 1. Rasperry pi Model 4 B
 2. 2 Web cameras/ Pi cameras
 3. Servo motor
 4. Trolley model (Madee by Cardboad or any other items)
 
###  First we are finding the sections of sueprmarket such as Soap section, Biscuit section, Cosmetics section, Exit etc. by using video processing done to video feed coming the first camera and prediciting the sections by using tflite model that was trained using the datasets of images of sections in foodcity. After finding the sections, real time location coordinates will be sent to cloud and the location will be displayed in mobile app. 
