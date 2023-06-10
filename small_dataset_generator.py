import random, os, shutil

dir = "15SceneData_small"

# remove 90% of all files in dir
for root, dirs, files in os.walk(dir):
    for name in files:
        if random.random() < 0.9:
            os.remove(os.path.join(root, name))