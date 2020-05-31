"""
This script is for seperating the data that is organized and downloaded from the engineer5 group.
See the notes.txt for how to download the data and put it into the folders that this file is expecting. 
Note: This file is as if you are working on the Udacity workspace provided.
"""


import shutil
import os
import numpy as np

dataBaseFolder = '/home/workspace/Udacity_Capstone/data/'
dataTL = dataBaseFolder + 'tl_engineer5/'

#it had the classes in the folder so each folder was /green, /yellow, etc.
folders = os.listdir(dataTL)
for folder in folders:
    os.makedirs(dataTL +'train/' + folder)
    os.makedirs(dataTL +'val/' + folder)
    os.makedirs(dataTL +'test/' + folder)

#using those same class labels
for folder in folders:
    src = dataTL + folder
    allFileNames = os.listdir(src)
    np.random.shuffle(allFileNames)
    train_FileNames, val_FileNames, test_FileNames = np.split(np.array(allFileNames),
                                                          [int(len(allFileNames)*0.7), int(len(allFileNames)*0.85)])
    train_FileNames = [src+'/'+ name for name in train_FileNames.tolist()]
    val_FileNames = [src+'/' + name for name in val_FileNames.tolist()]
    test_FileNames = [src+'/' + name for name in test_FileNames.tolist()]
    print('Class: ', folder)
    print('\t Total images: ', len(allFileNames))
    print('\t Training: ', len(train_FileNames))
    print('\t Validation: ', len(val_FileNames))
    print('\t Testing: ', len(test_FileNames))
    # Copy-pasting images
    for name in train_FileNames:
        shutil.move(name, dataTL +'train/' + folder + name[name.rfind('/'):])
    for name in val_FileNames:
        shutil.move(name, dataTL +'val/' + folder + name[name.rfind('/'):])
    for name in test_FileNames:
        shutil.move(name, dataTL +'test/' + folder + name[name.rfind('/'):])

for folder in folders:
    shutil.rmtree(dataTL + folder)
    
# if os.path.isdir(datasdcnd+"unidentified"):
#     shutil.rmtree(datasdcnd+"unidentified")

# folders = os.listdir(dataBaseFolder + d)
# print('Number of folders: ',len(folders))
# for fname in folders:
#     files = os.listdir(dataBaseFolder + d +'/'+ fname)
#     print('\t'+fname)
#     print('\t\t'+'Number of files: ',len(files))


# for fname in os.listdir(dataTL):
#     if 'png' in fname:
#         shutil.move(dataTL+fname,dataBaseFolder+'mixedupData/'+fname)

