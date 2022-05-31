from djitellopy import tello
import numpy as np
import cv2
import pygame
import time

#drone = tello.Tello()
#cap = cv2.VideoCapture(1)   #reads the data from the camera

maxArea = 6000
minArea = 4000

maxHeight = 280
minHeight = 200

WinW = 360
WinH = 240

#method to get pressed key
def GetKey(pressedkey):
    FoundKey = False
    for event in pygame.event.get():    pass

    keypressed = pygame.key.get_pressed()
    decodedKey = getattr(pygame, 'K_{}'.format(pressedkey))
    if keypressed[decodedKey]:
        FoundKey = True
        print(f"Pressed key: {decodedKey}")
    pygame.display.update()
    return FoundKey


#method for manual driving of drone
def ManualFlight(drone, standardvals):
    lr, fb, ud, yv = 0, 0, 0, 0
    speed = 25

    if GetKey('w'):
        fb = speed

    if GetKey('s'):
        fb = -speed

    if GetKey('a'):
        lr = -speed

    if GetKey('d'):
        lr = speed

    if GetKey('LEFT'):
        yv = -speed
    elif GetKey('RIGHT'):
        yv = speed

    if GetKey('UP'):
        ud = speed
    elif GetKey('DOWN'):
        ud = -speed

    if GetKey('q'):
        drone.land()
    elif GetKey('e'):
        drone.takeoff()


    drone.send_rc_control(lr + standardvals[0], fb + standardvals[1], ud + standardvals[2], yv)

#method for calibrating the drifting of the drone
def CalibrateDrifting(drone):
    #used for comparing current numbers with previous numbers
    prevX = 0
    prevY = 0
    prevA = 0
    #added to the speed applied, once the face is centered only these values should be affecting the
    standardfb = 0
    standardlr = 0
    standardud = 0
    #starts the clock and
    start = time.perf_counter()
    drone.send_rc_control(0, 0, 10, 0)
    time.sleep(3)
    drone.send_rc_control(0, 0, 0, 0)

    timerStandStill = 0
    #drone.takeoff()
    #used for checking if the timer is on or not
    timer = False
    fb, ud, lr = 0, 0, 0
    prevfbError, prevudError, prevlrError = 0, 0, 0
    while True:

        #defines the video
        video = drone.get_frame_read().frame
        video = cv2.resize(video, (WinW, WinH))
        #draws rectangles around the face and displays the camera feed
        video, currentvals = findFace(video)
        cv2.imshow("Drift calibration", video)
        currentX, currentY, currentA = currentvals[0][0], currentvals[0][1], currentvals[1]

        cv2.waitKey(1)
        #gets the speeds of different directions
        lrError = currentX - WinW//2
        #lr = (lrError * 0.01) + (lrError - prevlrError) + standardlr
        lr = lrError * 0.1

        udError = WinH//2 - currentY
        #ud = (0.01 * udError) + (udError - prevudError) + standardud
        ud = 0.1 * udError

        fbError = currentA - ((maxArea + minArea)//2)
        #fb = (0.005 * fbError) + (fbError - prevfbError) + standardfb
        fb = 0.005 * fbError

        print(f"lr: {lr}, fb: {fb}, ud: {ud}")

        # np.clip would not work correctly, these if statements are here to do the work instead
        """if lr > 10:     lr = 10
        elif lr < -10:  lr = -10

        if ud > 10:     ud = 10
        elif ud < -10:  ud = -10

        if fb > 5:      fb = 5
        elif fb < -5:   fb = -5
        lr = int(lr)
        ud = int(ud)
        fb = int(fb)"""
        lr = int(np.clip(lr, -20, 20))
        ud = int(np.clip(ud, -20, 20))
        fb = int(np.clip(fb, -10, 10))


        if currentX:
            drone.send_rc_control(lr, fb, ud, 0)
        else:   #stops the drone from moving when not having a face in frame
            drone.send_rc_control(0, 0, 0, 0)
        #checks every 1 seconds what the current status of the movement is
        clock = time.perf_counter()
        clock -= start
        #print(clock)
        #sometimes it takes a long time to start the loop, this if statement is here to prevent that from disrupting the
        #flow of the code
        if clock > 0.1:
            start += clock
        if clock > 0.1:
            #bool to check if a error has been found
            foundError = False
            #if the face is on the right side and the camera has been moved to the right
            if currentX > WinW//2 and prevX > currentX:
                standardlr += 1
                print("Too far to the right")
            #if the face is on the left side and the camerade has been moved to the left
            elif currentX < WinW//2 and prevX < currentX:
                standardlr -= 1
                print("Too far to the left")

            #if the face is on the lower part of the screen and the drone has been moving down
            if currentY > WinH//2 and prevY < currentY:
                standardud += 1
                print("Too low")
            #if the face is on the upper part of the screen and the drone has been moving upwards
            elif currentY < WinH//2 and prevY > currentY:
                standardud -= 1
                print("Too high")

            #if the camera is too far away and the camera has been moved backwards
            if currentA < minArea and prevA < currentA:
                standardfb += 1
                print("Too far away")
            #if the camera is too close and the camera has been moved forwards
            elif currentA > maxArea and prevA < currentA:
                standardfb -= 1
                print("Too close")

            #if the face is not centered enough it will prevent the timer from starting
            #split it into multiple if elif to make it more readable
            if currentX > prevX + 6 or currentX < prevX - 6:
                foundError = True
                print("drone had moved too much")
            elif currentY > prevY + 4 or currentY < prevY - 4:
                foundError = True
                print("drone had moved too much")
            elif currentA > prevA + 200 or currentA < prevA - 200:
                foundError = True
                print("drone had moved too much")

            #checks if the camera is still holding still enough, same requirements as the if statement below
            if timer:
                if not foundError and currentX and start - timerStandStill > 1:
                    print("Calibration completed")
                    return [standardlr, standardfb, standardud]
                timer = False


            #if there has not been an error and there is a face in frame, start the timer bool
            if not foundError and currentX:
                print("Starting timer")
                timer = True
                timerStandStill = start
            prevA = currentA
            prevX = currentX
            prevY = currentY

            start += 0.1





def findFace(grayImg):  #method to find the closest face
    faceCascade = cv2.CascadeClassifier("Data/haarcascade_frontalface_default.xml") #defines the facetracking ai
    faces = faceCascade.detectMultiScale(grayImg, 1.2, 8)   #finds the face

    faceC = []  #list of all current faces
    faceA = []  #list of all areas of the faces

    for (x, y, w, h) in faces:
        cv2.rectangle(grayImg, (x, y), (x + w, y + h), (0, 255, 0), 2)  #draws rectangle on face
        cx = x + w//2
        cy = y + h //2
        cv2.circle(grayImg, (cx, cy), 10, (0, 255, 0), cv2.FILLED)   #draws a circle in the center of the face

        area = w*h

        faceC.append([cx, cy])
        faceA.append(area)
        print(f"Area: {area}")
        print(f"x value: {cx}, y value: {cy}, width: {w}, height: {h}")
    if len(faceA):
        i = faceA.index(max(faceA))
        return grayImg, [faceC[i], faceA[i]]    #returns the closest face
    else:
        return grayImg, [[0, 0], 0]   #if there are no faces, return 0 on everything



def trackFace(drone, info, standardvals, pYVError, pUDError):
    area = info[1]
    x, y = info[0]

    fbSpeed = 0

    yvError = x - WinW //2
    yvSpeed = (0.1 * yvError + 0.1 * (yvError - pYVError))
    yvSpeed = int(np.clip(yvSpeed, -100, 100))

    udError = WinH//2 - y
    udSpeed = 0.2 * udError + 0.2 * (udError - pUDError)
    udSpeed = int(np.clip(udSpeed, -10, 10))

    if area < maxArea and area > minArea:
        fbSpeed = 0
    elif area > maxArea:
        fbSpeed = -5
    elif area < minArea and area != 0:
        fbSpeed = 5

    if udError < maxHeight and udError > minHeight:
        ud = 0
    elif udError > maxHeight or udError < minHeight:
        ud = udSpeed



    if x == 0:
        yvSpeed = 0
        yvError = 0
        udSpeed = 0
        udError = 0
    print(f"0, {fbSpeed}, {udSpeed}, {yvSpeed}")
    if x:
        drone.send_rc_control(standardvals[0], fbSpeed + standardvals[1], udSpeed + standardvals[2], yvSpeed)
    else:
        drone.send_rc_control(0, 0, 0, 0)
    return yvError, udError

def main(drone):
    pYVError = 0
    pUDError = 0
    FaceTrackingMode = False
    #standardvals = CalibrateDrifting(drone)
    standardvals = [0, 0, 0]
    #sets up pygame window
    pygame.init()
    window = pygame.display.set_mode((400, 400))
    pictureCount = 0
    while True:
        video = drone.get_frame_read().frame
        video = cv2.resize(video, (WinW, WinH))
        cv2.imshow("Camera feed", video)
        grayImg = cv2.cvtColor(video, cv2.COLOR_BGR2GRAY)
        grayImg = cv2.resize(grayImg, (WinW, WinH))
        grayImg, info = findFace(grayImg)
        cv2.imshow("Facetracking feed", grayImg)
        if GetKey("f"):
            FaceTrackingMode = not FaceTrackingMode
        elif GetKey("p"):
            cv2.imwrite(f"Pictures/image{pictureCount}.jpg", video)
        if FaceTrackingMode:
            pYVError, pUDError = trackFace(drone, info, standardvals, pYVError, pUDError)
        else:
            ManualFlight(drone, standardvals)
        cv2.waitKey(1)


if __name__ == '__main__':
    #sets up drone
    drone = tello.Tello()
    drone.connect()
    print(drone.get_battery())
    drone.streamon()
    drone.takeoff()

    main(drone)
