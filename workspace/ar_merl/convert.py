import os
import glob
import argparse
from cv2 import cv2
import numpy as np
import pandas as pd
from scipy.io import loadmat
from PIL import Image

parser = argparse.ArgumentParser(description='Detection to recognition for MERL Shopping.')
parser.add_argument('--start', default=1, type=int, help='start video')
parser.add_argument('--end', default=2, type=int, help='end video')
parser.add_argument('--labels_path', type=str, help='define path for label folder')
parser.add_argument('--video_path', type=str, help='define path for video folder')
parser.add_argument('--data_split', type=str, help='define output data folder name: test/val/train')

def crop_center(frame):
  y, x = frame.shape[0:2]
  min_dim = min(y, x)
  start_x = (x // 2) - (min_dim // 2)
  start_y = (y // 2) - (min_dim // 2)
  return frame[start_y:start_y+min_dim,start_x:start_x+min_dim]


def extract_frames(video):
  id_str = os.path.basename(video)  
  cap = cv2.VideoCapture(video)
  print('Stream start.', video)
  
  h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
  w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
  t = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
  
  clip_array = []
  
  while cap.isOpened():
      
      ret, frame = cap.read()
      if not ret:
          print('Stream end.\n')
          break    
      frame = crop_center(frame)
      frame = cv2.resize(frame, (224,224))
      frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) 
      clip_array.append(frame)
      
  print ('releasing the cap')
  cap.release() 
  return  id_str, clip_array


def main():
  global args
  args = parser.parse_args()
  # map index to class ids
  class_idx = {'0': 'Reach_to_Shelf','1': 'Retract_from_Shelf', '2':'Hand_in_Shelf', '3':'Inspect_Product', '4':'Inspect_Shelf'}

  actions = glob.glob(args.labels_path + '/' + '/*.mat')
  videos = glob.glob(args.video_path + '/' + '/*.mp4')
  start_video = args.start
  end_video = args.end

 # j = 1

  for i in range(start_video, end_video+1):

    video = videos[i-1]
#    print(f'video_{i}')
    id_str, clip_mat = extract_frames(video)
    id_str = id_str[:-8] 
    idx = actions.index(args.labels_path + '/' + id_str + 'label.mat')
    action = loadmat(actions[idx])
   
    for index, value in enumerate(action['tlabs']):
      print("index", index)
      k = 1 
      for start, stop in value[0]:
          img_id =1
          for x in range (start,stop):

              im = Image.fromarray(clip_mat[x])
              path_save_dir =  args.data_split + '/' + class_idx[str(index)] + f'/clip_{i}_{k}/rgb/'
              path_save = path_save_dir + str(img_id).zfill(6)+".png"

              if not (os.path.isdir(path_save_dir)):
                  print('creatig new dir',path_save_dir)
                  os.makedirs(path_save_dir)
              im.save(path_save)
              img_id+= 1

          k+= 1
    #j+= 1

  return


if __name__ == '__main__':
  main()
