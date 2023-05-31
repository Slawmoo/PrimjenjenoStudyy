import os
import shutil

testCSV = "Test.csv"
testDir = "C:/Users/student/Desktop/Karavla8/Test"

os.makedirs(testDir, exist_ok=True)

rows = open(testCSV).read().strip().split("\n")[1:]

for r in rows:
    (label, imagePath) = r.strip().split(",")[-2:]
    os.makedirs(os.path.join(testDir, label), exist_ok=True)
    shutil.copy(os.path.join("C:/Users/student/Desktop/Karavla8", imagePath), os.path.join(testDir,label))