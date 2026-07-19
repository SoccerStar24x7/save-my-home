from ollama import chat
import base64
import os


img = 'image1.jpeg'

onF = 'data/on crop'
offF = 'data/off crop'


on = 'on.jpeg'
off = 'off.jpeg'

with open(on, "rb") as f:
    onBase = base64.b64encode(f.read()).decode('utf-8')
    
with open(off, "rb") as f:
    offBase = base64.b64encode(f.read()).decode('utf-8')

with open(img, "rb") as f:
    imageBase = base64.b64encode(f.read()).decode('utf-8')

"""
onImgs = []
for filename in os.listdir(onF):
    onImgs.append(base64.b64encode(filename.read()).decode('utf-8'))

offImgs = []
for filename in os.listdir(offF):
    offImgs.append(base64.b64encode(filename.read()).decode('utf-8'))
"""




response = chat(
  model='gemma4',
  messages=[
    {
      'role': 'user',
      'content': 
'You must identify if a stove is on based off an image of the stove knobs. ' 
'Three images have been given to you. ' 
'The first image is what the stove knobs look like when the stove is OFF.'
'The second image is what the stove knobs look like when the stove is ON.'
'You must identify if the stove is on or off in the third image.'
'To identify, look at each of the 5 knobs individually. '
'Notice the angle each knob is at in the first image, when they are all off. '
'Remember that specific angle for each knob as the "off" angle.'
'When looking at the second image, look at each knob individually, '
'and compare the angle of the knob to the saved angle you have of the knob off in the first image.'
'Notice how one of them is angled differently in the second image, which means that knob is NOT off, meaning the stove is NOT off, meaning it is on.'
'Tell me which knob you think it is.'
'Now, look at the third image. Use the same identification and classification you used for the second image;'
'Compare the angle of each of the 5 knobs to the angle of the knobs in the first image, when it was in the off poisition. '
'Does the angle look EXACTLY the same? Is there enough angle so that the stove would actually be on?'
'The stove is ONLY on if the angle of 1 or more knobs in image 3 is different than the angle for that SPECIFIC knob in image 1, the off one.'
'This time, tell me if the stove is on or off in the third image, and explain your reasoning.',

      'images': [offBase, onBase, imageBase]
    }
  ]
)

print(response.message.content)
"""
try:
    if(int(response.message.content)):
        print("stove is on")
    else:
        print("stove is off")

except Exception as e:
    print(f"Error when retriving model output and converting: {e}")
"""