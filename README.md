# Hand-Gesture-interface
Dissertation project at university. First class honors. Voted "best overall project" in 2017.

This application allows the user to control their computer mouse using only hand gestures and an RGB camera as an input device. 

It does this through the following steps:

# Setup
1. User gives program a picture of the palm of their hand. Picture taken by the program itself. 
2. Program creates a colour model for the users skin based on given picture of users palm.
3. Program shows user the camera feed where objects within the colour model are white and everything else is black.
4. User tweaks options in order to isolate only skin, as best as possible.

#Programs main execution loop
1. Program finds largest body of white within frame. This is hopefully the users hand.
2. Program creates a convex hull (a shell) around the users hand.
3. Program compares convex hull against users hand to determine how. Many fingers are being held up (i.e. gestures)
4. Program parses gesture input and invokes mouse events as output.
5. Program checks hand position and moves computer mouse accordingly. 


Any questions on implementation can be answered if you have them.
