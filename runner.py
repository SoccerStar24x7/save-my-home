from oldEval import evaluate
import time
import os
from crops import crop
#import playsound

def on():
    #playsound("ding.mp3")
    print("ITS ON RAAAH")

minWait = 5    

onC = 0
while(True):
    print("looped")
    try:
    	os.system("ssh admin@192.168.1.170 \"python /home/admin/coding/stove/main.py\"")
    except Exception as e:
	    continue
    print("ssh successful")


    crop("image.jpeg") # crops image

    num = evaluate("image.jpeg") # evaluate it 

    print("fein")

    if num == 2:
        onC += 1 
        if onC > 6:
            on()
            onC = 0
    
    else:
        onC = 0
        if num == 0:
            print("off")
        else:
            print("none")
    
    time.sleep(minWait * 60)