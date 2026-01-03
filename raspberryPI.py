import os
import subprocess

remote = "admin@192.168.1.170"

cmd1 = "python /home/admin/coding/stove/train.py"

cmd3 = "python /home/admin/coding/stove/rebirth.py"

# os.system(f"ssh {remote} '{cmd1}'")
result = subprocess.run(
        ["ssh", remote, cmd1],
        check=True, # Raise an exception if the command fails
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True # Decode output as text (Python 3.5+)
    )
print(result.stdout)


os.system(f"scp -r {remote}:/home/admin/coding/stove/data \"C:\\Users\\Arnav H\\Documents\\coding\\save-my-home\\data\"")


# os.system(f"ssh {remote} 'python /home/admin/coding/stove/rebirth.py'")
result = subprocess.run(
        ["ssh", remote, cmd3],
        check=True, # Raise an exception if the command fails
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True # Decode output as text (Python 3.5+)
    )
print(result.stdout)
