# -Super-Market-Smart-Trolley-System

This project was implemented to remove thefts in stores when adding items to trolley, to provide the location of sections in store, and to generate automated bills that will  be displayed in the app when checking out.
 #     Authors/Creators : Hariharan Raveenthiran & Sivasthigan Santhirarajah
 #     Author or Creator of Mobile application: Thanoraj Muthulingam

 
 Required Items
 1. Rasperry pi Model 4 B
 2. 2 Web cameras/ Pi cameras
 3. Servo motor
 4. Trolley model (Made by Cardboad or any other items)
 
###  First we are finding the sections of sueprmarket such as Soap section, Biscuit section, Cosmetics section, Exit etc. by using video processing done to video feed coming the first camera and prediciting the sections by using tflite model that was trained using the datasets of images of sections in foodcity. After finding the sections, real time location coordinates will be sent to cloud and the location will be displayed in mobile app. 


#Let me explain a simple scenario that can be done using my setup.

Assume that you are at the entrance of a food city, and you are going inside. There will be many sections in the foodcity according to the goods it has. First camera in our trolley (i.e) Main camera will find the sections using image processing and tflite, and it will update the location of the customer in the app in realtime. 

Assume that you are going to add orange to your cart. First camera will predict whether it is an orange or not, and then when the customer puts that orange in to the trolley, the second camera inside the trolley will predict whether the item that is being put in to the trolley is same or different from the item that was captured by first camera. If  item is same in both cameras, then that will be added to the shopping cart. Other wise bill will not be generated, and the shop owners will be notified.

