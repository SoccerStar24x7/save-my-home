import os
import time

timer = int(time.time())

takePhotos = f'ssh admin@192.168.1.170 "python /home/admin/coding/stove/train.py"'

get_data = f"rsync -az admin@192.168.1.170:/home/admin/coding/stove/data/ /home/arnavhegde/Documents/coding/save-my-home/data/imported"

removeData = f'ssh admin@192.168.1.170 "rm -rf /home/admin/coding/stove/data/"'

removeData = f'ssh admin@192.168.1.170 "mkdir /home/admin/coding/stove/data/"'


os.system(takePhotos)
os.system(get_data)
