import os

dir_path = './original/'
for story_name in os.listdir(dir_path):
    old_path=dir_path+story_name
    new_name=story_name.replace(" ","_").replace(",","")
    new_path=dir_path+new_name
    os.rename(old_path,new_path)

dir_path = './annotations/'
for story_name in os.listdir(dir_path):
    old_path=dir_path+story_name
    new_name=story_name.replace(" ","_").replace(",","")
    new_path=dir_path+new_name
    os.rename(old_path,new_path)