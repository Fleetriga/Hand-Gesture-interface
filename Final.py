#imported libraries
import ctypes
import time
import numpy as np
import cv2
import os

def resource_path(relative_path):
    try:
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)

def contours(img, currentMousePosition, defect_threshold):
    #Create a copy of the frame given to draw onto, as contour calculation affects the frame it is performed on
    contouredFrame = np.copy(img)

    #Convert the single channel copy into a 3 channel colour image
    contouredFrame = cv2.cvtColor(contouredFrame, cv2.COLOR_GRAY2BGR)

    #Calculate contours for the given image
    im2, contours, hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    #Extract largest contour
    max_area = 0
    cnt = None
    for i in range(len(contours)):
            current_cnt = contours[i]
            area = cv2.contourArea(current_cnt)
            if(area > max_area):
                max_area = area
                cnt = current_cnt
    
    #If there was an object within the given frame to calculate contours for
    if cnt != None:
        #Draw the contours found onto the copy created at the start of the function
        cv2.drawContours(contouredFrame,[cnt],0,(0,255,255),2)

        #Calculate the convex hull for the object identified by the calculated contours
        hull = cv2.convexHull(cnt, returnPoints = False)
        
        #Calculate the image defects of the convex hull
        defects = cv2.convexityDefects(cnt,hull)

        #Create a list to hold the position of the defects after they have been filtered
        l = []
        if defects != None and len(defects) > 0:
            #Iterate through the defects
            for i in range(defects.shape[0]):
                #First drawn isn't part of the object, so don't include
                if i != 0: 
                    #Unpack the current defect
                    s,e,f,d = defects[i,0]

                    #Further unpack values
                    start = tuple(cnt[s][0])
                    end = tuple(cnt[e][0])
                    far = tuple(cnt[f][0])

                    #Even if the defect isnt drawn we still want the entire convex hull, so draw it
                    cv2.line(contouredFrame,start,end,[0,255,0],2) #Convex hull

                    #Filter defects by distance to the convex hull(false positives)
                    if d > defect_threshold:
                        #add the location of the defect
                        l.append(far)

                        #Draw the defect onto the frame
                        cv2.circle(contouredFrame,far,3,[0,0,255],-1) #Defects
            
        #Return the mean position of the filtered defects as well as the number of defects. (Final return is the frame this data was drawn onto)
        if(len(l) > 0):
            return l[0], len(l), contouredFrame
        else: return currentMousePosition, 0, contouredFrame
    return 0, 0, contouredFrame

def update_moving_average(currentAverage):
    global lastPosition
    global currentPosition
    if len(moving_average) < 5:
        moving_average.append(currentAverage)
    else:
        moving_average.pop(0)
        moving_average.append(currentAverage)

        #Set last position to current position
        lastPosition = currentPosition

        #And then set current position to the new mean of the moving average
        currentPosition = np.mean(moving_average, axis=0) #Gives a 2d position

def reset_moving_average():
    moving_average = []

def set_rects(event, x, y, flags, param):
    # grab references to the global variables
    global rects, drawBound
    #Check if the left mouse button was pressed.
    if event == cv2.EVENT_LBUTTONDOWN:
        #Reset it so if you start drawing a second rectangle the first disappears.
        drawBound = False

        #Set the beginning corner of the new rectangle being drawn
        rects = [x, y]
 
    # check if the left mouse button was released.
    elif event == cv2.EVENT_LBUTTONUP:
        #append the mouse x and y to the rectangle corners array rects = [x1, y1, x2, y2]
        rects.append(x), rects.append(y)

        #Convert to the format used in this software for easy unpacking of values (check method get_rectangle_in_shape(rects, img)) [[x1, y1, x2, y2]]
        rects = [rects]
        drawBound = True

def update_average(dataset):
    return np.mean(dataset, axis=0)

def colour_model_generation_sequence(cam):
    global rects, drawBound

    #Wait for the user to set bounding box of their hand and press Q to continue to next step
    while(True):
        # Capture frame-by-frame
        ret, frame = cam.read()
        
        #If the user has chosen a bound,draw it
        if(drawBound):
            frame = box(rects, frame)
        
        #Blur frame to get rid of features.
        blur = cv2.GaussianBlur(frame, (25, 25), 0)
        
        #Show frame
        cv2.imshow("Software", frame)
        
        #Wait for q key to be pressed AFTER user has chosen a bound
        if cv2.waitKey(1) & 0xFF == ord('q'):
            if(drawBound):
                break

    #Convert the last frame into the correct colour field, extract just the area bounded by the rectangle
    frame = cv2.cvtColor(blur, konkaiColour)
    rects = validate_rects(rects, frame.shape)
    justRectangle = get_string_of_values_from_rectangle(rects, frame)
    return update_average(justRectangle)

def colour_model_priority_regeneration(cam, average_channel_one, average_channel_two, average_channel_three):
    global channel_one_priority, channel_two_priority, channel_three_priority, rects, drawBound

    # create trackbars for threshold changes
    cv2.createTrackbar('SkinThresh','Software',1500,5000,update_threshold)
    cv2.createTrackbar('Y_Prio','Software',50,100,update_Y)
    cv2.createTrackbar('Cr_Prio','Software',50,100,update_Cb)
    cv2.createTrackbar('Cb_Prio','Software',50,100,update_Cr)


    ##Alow user to choose their threshold and "mousepad" then continue by pressing q key
    while True:
        ########grab the current frame
        (grabbed, frame) = cam.read()

        #Convert the colour space of the frame
        frame = cv2.cvtColor(frame, konkaiColour)

        #Blur to remove features of objects within the grabbed frame.
        frame = cv2.GaussianBlur(frame, (25,25), 0)
        
        #Calculate the distance from the mean pixel of skin's colour to each pixels colour in the grabbed frame
        distances = ((((frame[:,:,0]-average_channel_one)**2)*channel_one_priority) + ((((frame[:,:,1]-average_channel_two)**2))*channel_two_priority) + ((((frame[:,:,2]-average_channel_three)**2))*channel_three_priority))

        #Threshold pixels too far away from mean (This results in a 1D ARRAY, conversion to 3channels is required later on)
        thresholded = np.array(np.where(distances < threshold , 255, 0), np.uint8)

        #Convert to RGB so bound can be drawn onto it
        thresholded = cv2.cvtColor(thresholded, cv2.COLOR_GRAY2BGR)

        #If the user has chosen a bound,draw it
        if(drawBound):
            thresholded = box(rects, thresholded)
        
        #Display outputs
        cv2.imshow("Software", thresholded)

        #If I is pressed, redo previous step
        if cv2.waitKey(1) & 0xFF == ord('i'):
            average_channel_three, average_channel_two, average_channel_one = colour_model_generation_sequence(cam)
        
        #If q is pressed continue to next stage
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            return average_channel_one, average_channel_two, average_channel_three

def defect_threshold_regeneration(cam, average_channel_one, average_channel_two, average_channel_three, channel_one_priority, channel_two_priority, channel_three_priority, rects):
    global defect_threshold
    # create trackbars for threshold changes
    cv2.createTrackbar('DfctThresh','Software',1500,10000,update_defect_threshold)

    ##Allow user to chose their defect threshold
    while True:
        ########grab the current frame
        (grabbed, frame) = cam.read()
        contourBox = get_rectangle_in_shape(rects, frame);

        #Convert it from BGR to HSV so light invariance can be applied
        contourBox = cv2.cvtColor(contourBox, konkaiColour)

        #Blur the grabbed frame for more robust isolation
        contourBox = cv2.GaussianBlur(contourBox, (25,25), 0)
        
        #Calculate the distance from the mean pixel of skin's colour to each pixel's colour
        distances = ((((contourBox[:,:,0]-average_channel_one)**2)*channel_one_priority) + ((((contourBox[:,:,1]-average_channel_two)**2))*channel_two_priority) + ((((contourBox[:,:,2]-average_channel_three)**2))*channel_three_priority))

        #Threshold pixels too far away from mean (This results in a 1D ARRAY, conversion to 3channels is required later on)
        thresholded = np.array(np.where(distances < threshold , 255, 0), np.uint8)

        #Find defect information, and also draw informative infromation onto the frame
        defectMean, noDefects, contourBox = contours(thresholded, (0,0), defect_threshold)
        
        #Show information in a frame to be analysed by user
        frame = overrideFrame(rects, contourBox, frame)

        #Display outputs
        cv2.imshow("Software", frame)
        
        #If I is pressed, redo previous step
        if cv2.waitKey(1) & 0xFF == ord('i'):
            cv2.destroyAllWindows()
            average_channel_three, average_channel_two, average_channel_one = colour_model_priority_regeneration(cam, average_channel_three, average_channel_two, average_channel_one)
            cv2.namedWindow("Software")
            cv2.createTrackbar('DfctThresh','Software',1500,10000,update_defect_threshold)
        
        #If q is pressed continue to next stage
        if cv2.waitKey(1) & 0xFF == ord('q'):
            #Destroy windows to get rid of trackbar
            cv2.destroyAllWindows()
            return average_channel_one, average_channel_two, average_channel_three

##
# METHODS FOR RECTANGLES
# DRAW, CONVERT BETWEEN openCV IMAGE FORMAT openCV Window FORMAT, CROP IMAGE TO GIVEN RECTANGLE,  
##
def box(rects, img):
    #Unpack corners of the given rectangle
    for x1, y1, x2, y2 in rects:
        #Draw the rectangle onto the frame for the given values
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
    return img


def validate_rects(rects, max):
    #Unpack corners of the rectangle
    for x1, y1, x2, y2 in rects:
        #bound all values within window size (if coordinate is less than 0)
        if x1 < 0: 
            x1 = 0
        if y1 < 0:
            y1 = 0
        if x2 < 0:
            x2 = 0
        if y2 < 0:
            y2 = 0
        #bound all values within window size (if coordinate is greater than max x or y value)
        if x1 > max[1]:
            x1 = max[1]
        if y1 > max[0]:
            y1 = max[0]
        if x2 > max[1]:
            x2 = max[1]
        if y2 > max[0]:
            y2 = max[0]

        #Get (minX, maxX) of bounding rectangle
        if(x1 > x2):
            temp = (x2, x1)
        else:
            temp = (x1, x2)

        #Get (minY, maxY) of bounding rectangle
        if(y1 > y2):
            temp2 = (y2, y1)
        else:
            temp2 = (y1, y2)
    return [[temp[0], temp2[0], temp[1], temp2[1]]]

def convert_rects(rects):
    #Unpack corners of the rectangle
    for x1, y1, x2, y2 in rects:
        #Put bounding rectangle coordinates in order (minX, minY, maxX, maxY)
        #Put this in format this software uses for easy unboxing of values [[rectdata]]
        #Swap X and Y values around because openCV images are (Y, X) and not (X, Y) like in the callback function
        return [[y1, x1, y2, x2]]

def get_string_of_values_from_rectangle(rects, img):
    #Unpack corners of the rectangle
    for x1, y1, x2, y2 in rects:

        #Derive data needed for next step.
        xRange, yRange, x, y = x2-x1,  y2-y1, x1, y1
 
        #Create empty list to be returned at the end of the function
        justRectangle = []
 
        #Iterate through values to get a 1d array of pixels instead of one in the shape of the rectangle given (e.g. a 10x10 array of pixels)
        for y2 in range(yRange):
            for x2 in range(xRange):
                justRectangle.append((img[y2+y,x2+x]))
        return np.array(justRectangle)

def get_rectangle_in_shape(rects, img):
    #Unpack corners of the rectangle
    for x1, y1, x2, y2 in rects:
        #Return the section of the frame that covers that rectangle
        box = img[x1:x2,y1:y2]
        return box

def overrideFrame(rects, override, img):
    for x1, y1, x2, y2 in rects:
        img[x1:x2, y1:y2] = override[:,:]
        return img

##
# METHOD FOR PARSING USER INPUT AND DECIDING/PERFORMING RESULTING ACTION
#
##
def commitAction(noDefects, currentHandPos, last_input, sprites, time_at_last_input):
    #Set the default output to "no input given"
    returnImg = sprites[0]

    print last_input

    #If there's an object in shot i.e. noDefects > 0
    if noDefects > 0:
        #We always want to be updating the position so that when the mouse moves a correct movement vector can be calculated
        update_moving_average(currentHandPos);
        unholdClick()
        
        #If the imput is set for "lifting the mouse off the pad" don't move the mouse and reset the movingAverage
        if(noDefects == 2):
            returnImg = sprites[4]
            #Lifting the mouse requires no time to engage
            last_input = 4

            #restart counter for input in a way that makes the progress bar fill instantly
            time_at_last_input = time.time()
        #For #defects classed as "right clicking"
        elif(noDefects == 4):
            #Show that right click has been inputed
            returnImg = sprites[2]

            #Change the "last input" to right click
            last_input = 2

            #If right mouse click command has been held for long enough to be considered a genuine command
            proceed, time_at_last_input = same_input(last_input, 2, time_at_last_input)
            if(proceed):
                if(held_long_enough(time_at_last_input)):
                    rightClick()
                    moveMouse()

        #For defects classed as "left clicking"
        elif(noDefects == 3):
            #Show that right click has been inputed
            returnImg = sprites[1]

            #Change the "last input" to right click
            last_input = 1

            #If movement left mouse click has been held for long enough to be considered a genuine command
            proceed, time_at_last_input = same_input(last_input, 1, time_at_last_input)
            if(proceed):
                if(held_long_enough(time_at_last_input)):
                    leftClick()
                    moveMouse()

        #Else we'll only move the mouse (Note this is only when mouse is not lifted i.e number of defects == 2)  
        else:
            #Show that mouse movement has been inputted
            returnImg = sprites[3]

            #Change the "last input" to mouse movement
            last_input = 3

            #If movement command has been held for long enough to be considered a genuine command
            proceed, time_at_last_input = same_input(last_input, 3, time_at_last_input)
            if(proceed):
                if(held_long_enough(time_at_last_input)):
                    moveMouse()     
        
        #If neither the input for left or right click is down, set both corresponding flags to false
        if(noDefects != 4):
            global rightClickHeld
            rightClickHeld = False

        if(noDefects != 3):
            global leftClickHeld
            leftClickHeld = False
        return returnImg, time_at_last_input, last_input
    empty, time_at_last_input = same_input(999, 1000)
    return returnImg, time_at_last_input, last_input

##
# METHODS FOR VALIDATING USER INPUT
#
##
def get_progression_point(time_since_last):
    percentage = time.time() - time_since_last
    if(percentage >= 1):
        return 300
    elif(percentage > 0):
        return int(300*percentage)
    else:
        return 0

def held_long_enough(time_at_last_input):
    if(time.time() - time_at_last_input >= 1):
        return True
    else:
        return False

def same_input(last_input, givenInput, time_at_last_input):
    if(last_input == givenInput):
        return True, time_at_last_input
    else:
        last_input = givenInput
        return False, time.time()

def bound_mouse_position(currentMousePos):
    #Get the screensize using windows API
    screensize = ctypes.windll.user32.GetSystemMetrics(0), ctypes.windll.user32.GetSystemMetrics(1)

    #If current mouse position goes outside of that, bring it back in to the bound
    if(currentMousePos[0] < 0):
        currentMousePos[0] = 0

    if(currentMousePos[0] > screensize[0]):
        currentMousePos[0] = screensize[0]

    if(currentMousePos[1] < 0):
        currentMousePos[1] = 0

    if(currentMousePos[1] > screensize[1]):
        currentMousePos[1] = screensize[1]
    return currentMousePos

##
# METHODS FOR MOUSE MANIPULATION
#
##
def unholdClick():
    if(leftClickHeld == False):
        ctypes.windll.user32.mouse_event(4,0,0,0,0)
    if(rightClickHeld == False):
        ctypes.windll.user32.mouse_event(16,0,0,0,0)

def leftClick():
    global leftClickHeld
    if(leftClickHeld == False):
        leftClickHeld = True
        ctypes.windll.user32.mouse_event(2,0,0,0,0)

def rightClick():
    global rightClickHeld
    if(rightClickHeld == False):
        rightClickHeld = True
        ctypes.windll.user32.mouse_event(8,0,0,0,0)

def moveMouse():
    global lastPosition
    global currentPosition
    global currentMousePosition
    #Make a move vector from the difference in positions 
    movementVector = [(currentPosition[0]-lastPosition[0])*2, (currentPosition[1]-lastPosition[1])*2]
    
    #And then add it to the current mouse position
    currentMousePosition = [currentMousePosition[0] + movementVector[0], currentMousePosition[1] + movementVector[1]]

    #Bound it so that the mouse cannot go outside the screen (it'd take a while for the mouse to move if it's position is at -10,000)
    currentMousePosition = bound_mouse_position(currentMousePosition)

    ctypes.windll.user32.SetCursorPos(int(currentMousePosition[0]), int(currentMousePosition[1]))
    time.sleep(.01)

#
# METHODS FOR USER INPUT VIA WINDOW SLIDERS
#
def update_threshold(threshVal):
    global threshold
    threshold = threshVal

def update_Y(threshVal):
    global channel_one_priority
    channel_one_priority = float(threshVal)/50

def update_Cb(threshVal):
    global channel_two_priority
    channel_two_priority = float(threshVal)/50

def update_Cr(threshVal):
    global channel_three_priority
    channel_three_priority = float(threshVal)/50

def update_defect_threshold(threshVal):
    global defect_threshold
    defect_threshold = threshVal

#
# METHODS FOR ONCE THE PROGRAM HAS ENDED TO GIVE USER FULL CONTROLL OF THEIR MOUSE IN 
# SOME EXTREME CASES.
#
def end_program():
    #Destroy the window because the program has finished
    cv2.destroyAllWindows()

    #Relinquish controll of the camera
    cam.release()

    #Once again, unclick both mouse buttons to give user full control.
    ctypes.windll.user32.mouse_event(8,0,0,0,0), ctypes.windll.user32.mouse_event(16,0,0,0,0) #For right click hold and unhold
    ctypes.windll.user32.mouse_event(2,0,0,0,0), ctypes.windll.user32.mouse_event(4,0,0,0,0) #For left click hold and unhold

#A few booleans about user input
leftClickHeld = False
rightClickHeld = False

#Single variable to toggle debug elements
debug = True
konkaiColour = cv2.COLOR_BGR2YCrCb

#Get access to the camera and assign it to cam
cam = cv2.VideoCapture(0)

#Rect global variables
rects = ()
drawBound = False

#Set the variables required for mouse callback (When a mouse event happens in the "Software" window the set_rects() method is called to process the event)
cv2.namedWindow("Software")
cv2.setMouseCallback("Software", set_rects)

#Calculate the mean pixel colour of the given rectangle and unbox into three variables
average_channel_one, average_channel_two, average_channel_three = colour_model_generation_sequence(cam)

#Threshold for something to be considered skin
threshold = 1500
channel_one_priority = 1
channel_two_priority = 1
channel_three_priority = 1

#Turn off the bounding box until the user draws it again
drawBound = False

#Allow user to change colour model channel priority to get a crisper image segmentation
average_channel_one, average_channel_two, average_channel_three = colour_model_priority_regeneration(cam, average_channel_one, average_channel_two, average_channel_three)

#Recreate the window this time for defect threshold input
cv2.namedWindow('Software')

#Threshold for something to be considered skin
defect_threshold = 1500

grabbed, frame = cam.read()

#Convert rectangle coordinates from (x, y) format to (y, x) format
rects = validate_rects(rects, frame.shape)
rects = convert_rects(rects)

#Allow user to change defect model to get more robust gesture recognition later
average_channel_one, average_channel_two, average_channel_three = defect_threshold_regeneration(cam, average_channel_one, average_channel_two, average_channel_three, channel_one_priority, channel_two_priority, channel_three_priority, rects)

#Create an array to store the past 3 hand positions
moving_average = []

#Intialise some variables to allow smooth mouse movement
lastPosition = [0,0]
currentPosition = [0,0]
currentMousePosition = [0,0]

#Images of all inputs expected
noInput_s = cv2.imread(resource_path("sprites/default.png"))
leftMouse_s = cv2.imread(resource_path("sprites/LMB.png"))
rightMouse_s = cv2.imread(resource_path("sprites/RMB.png"))
movement_s = cv2.imread(resource_path("sprites/movement.png"))
lifted_s = cv2.imread(resource_path("sprites/lifted.png"))

sprites = (noInput_s, leftMouse_s, rightMouse_s, movement_s, lifted_s)

#Variables
time_at_last_input = time.time()
last_input = 0

##Main loop
while True:
    ########grab the current frame
    (grabbed, frame) = cam.read()
    contourBox = get_rectangle_in_shape(rects, frame);
    ##Blur the grabbed frame for more robust isolation
    contourBox = cv2.GaussianBlur(contourBox, (5,5), 0)

    #Convert it from BGR to HSV so light invariance can be applied
    contourBox = cv2.cvtColor(contourBox, konkaiColour)

    ##Blur the grabbed frame for more robust isolation
    contourBox = cv2.GaussianBlur(contourBox, (25,25), 0)
    
    #Calculate the distance from the mean pixel of skin's colour to each pixel's colour
    distances = ((((contourBox[:,:,0]-average_channel_one)**2)*channel_one_priority) + ((((contourBox[:,:,1]-average_channel_two)**2))*channel_two_priority) + ((((contourBox[:,:,2]-average_channel_three)**2))*channel_three_priority))

    #Threshold pixels too far away from mean (This results in a 1D ARRAY, conversion to 3channels is required later on)
    thresholded = np.array(np.where(distances < threshold , 255, 0), np.uint8)

    #Find defect information, and also draw informative infromation onto the frame
    defectMean, noDefects, contourBox = contours(thresholded, currentMousePosition, defect_threshold)

    #Decide which action to perform from defect information and then perform it.
    committedAction, time_at_last_input, last_input = commitAction(noDefects, defectMean, last_input, sprites, time_at_last_input)

    #Show information in a frame to be analysed by user
    frame = overrideFrame(rects, contourBox, frame)
    
    #A frame to show how long an input has been held
    input_progress = np.zeros(shape=(10, 300, 3))

    #Draw onto the frame, a progress bar that fills as the command is held
    cv2.line(input_progress, (0,5), (get_progression_point(time_at_last_input), 5), (0,255,0), 3)

    #Display output
    cv2.imshow("Software", frame)

    #Display action being performed as a sprite and the progress to performing it
    cv2.imshow("Progression", input_progress), cv2.imshow("Action", committedAction)
    
    #Wait for a key to be pressed to restart the program
    if cv2.waitKey(1) & 0xFF == ord('i'):
        cv2.destroyAllWindows()
        #Restart mouse callback
        drawBound = False
        cv2.namedWindow("Software")
        cv2.setMouseCallback("Software", set_rects)

        #Go through the process again
        average_channel_one, average_channel_two, average_channel_three = colour_model_generation_sequence(cam)
        average_channel_one, average_channel_two, average_channel_three = colour_model_priority_regeneration(cam, average_channel_one, average_channel_two, average_channel_three)

        #Convert rectangle coordinates from (x, y) format to (y, x) format
        rects = validate_rects(rects, frame.shape)
        rects = convert_rects(rects)
        average_channel_one, average_channel_two, average_channel_three = defect_threshold_regeneration(cam, average_channel_one, average_channel_two, average_channel_three, channel_one_priority, channel_two_priority, channel_three_priority, rects)

    #Wait for a key to be pressed to quit the program
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

end_program()
