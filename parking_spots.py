from __future__ import division
import matplotlib.pyplot as plt
import cv2
import os, glob
import numpy as np
os.environ["IMAGEIO_FFMPEG_EXE"] = "/usr/bin/ffmpeg"
from moviepy.editor import VideoFileClip

cwd = os.getcwd()

def show_images(images, cmap=None):
  cols = 2
  rows = (len(images) + 1) // 2
  plt.figure(figsize=(15, 12))
  
  for i, image in enumerate(images):
    plt.subplot(rows, cols, i+1)
    cmap = 'gray' if len(image.shape) == 2 else cmap
    plt.imshow(image, cmap=cmap)
    plt.xticks([])
    plt.yticks([])
  
  plt.tight_layout(pad=0, h_pad=0, w_pad=0)
  plt.show()


test_images = [plt.imread(path) for path in glob.glob('test_images/*.jpg')]
show_images(test_images)

def select_rgb_white_yellow(image):
  #Numpy array of unsigned long values on 8 bits, so by adding +1 to 255, resets to 0 
  lower = np.uint8([120, 120, 120])
  upper = np.uint8([255, 255, 255])
  white_mask = cv2.inRange(image, lower, upper)

  lower = np.uint8([190, 190, 0])
  upper = np.uint8([255, 255, 255])
  yellow_mask = cv2.inRange(image, lower, upper)

  mask = cv2.bitwise_or(white_mask, yellow_mask)
  masked = cv2.bitwise_and(image, image, mask = mask)
  return masked

white_yellow_images = list(map(select_rgb_white_yellow, test_images))
show_images(white_yellow_images)

def convert_gray_scale(image):
  return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

gray_images = list(map(convert_gray_scale, white_yellow_images))
show_images(gray_images)

def detect_edges(image, low_threshold = 50, high_threshold = 200):
  return cv2.Canny(image, low_threshold, high_threshold)

edge_images = (list(map(detect_edges, gray_images)))
show_images(edge_images)

def filter_region(image, vertices):
  mask = np.zeros_like(image)
  if len(mask.shape) == 2:
    cv2.fillPoly(mask, vertices, (255,))
  else:
    cv2.fillPoly(mask, vertices, (255,)*mask.shape[2])
  return cv2.bitwise_and(image, mask)  

def select_region(image):
  rows, cols = image.shape[:2]
  pt_1  = [cols*0.05, rows*0.90]
  pt_2 = [cols*0.05, rows*0.70]
  pt_3 = [cols*0.30, rows*0.55]
  pt_4 = [cols*0.6, rows*0.15]
  pt_5 = [cols*0.90, rows*0.15] 
  pt_6 = [cols*0.90, rows*0.90]
  vertices = np.array([[pt_1, pt_2, pt_3, pt_4, pt_5, pt_6]], dtype=np.int32)
  return filter_region(image, vertices)
roi_images = list(map(select_region, edge_images))
show_images(roi_images)

def hough_lines(image):
  return cv2.HoughLinesP(image, rho=0.1, theta=np.pi/10, threshold = 15, minLineLength=9, maxLineGap=4)

list_of_lines = list(map(hough_lines, roi_images))

def draw_lines(image, lines, color=[255, 0, 0], thickness=2, make_copy=True):
  if make_copy:
    image = np.copy(image)
  cleaned = []
  for line in lines:
    for x1, y1, x2, y2 in line:
      if abs(y2- y1) <= 1 and abs(x2 - x1) >= 25 and abs(x2 - x1) <= 55:
        cleaned.append((x1, y1, x2, y2))
        cv2.line(image, (x1, y1), (x2, y2), color, thickness)
  print(" No of lines detected: ", len(cleaned))
  return image

line_images = []
for image, lines in zip(test_images, list_of_lines):
  line_images.append(draw_lines(image, lines))
show_images(line_images) 

def identify_blocks(image, lines, make_copy=True):
  if make_copy:
    new_image = np.copy(image)
  
  cleaned = []
  for line in lines:
    for x1, y1, x2, y2 in line:
      if abs(y2 - y1) <= 1 and abs(x2 - x1) >= 25 and abs(x2 - x1) <= 55:
        cleaned.append((x1, y1, x2, y2))
  
  import operator
  sorted_cleaned = sorted(cleaned, key = operator.itemgetter(0, 1))
  
  clusters = {}
  dIndex = 0
  clust_dist = 10

  for i in range(len(sorted_cleaned) - 1):
    if abs(sorted_cleaned[i][0] - sorted_cleaned[i+1][0]) <= clust_dist:
      if not dIndex in clusters.keys(): clusters[dIndex] = []
      clusters[dIndex].append(sorted_cleaned[i])
      clusters[dIndex].append(sorted_cleaned[i+1])
    else:
      dIndex+=1
  
  rects = {}
  i = 0
  for key in clusters:
    all_list = clusters[key]
    current_list = list(set(all_list)) #removing the double added (x1, y1, x2, y2) from previous for
    if len(current_list) > 5:
      sorted_cluster = sorted(current_list, key = operator.itemgetter(1))
      avg_y1 = sorted_cluster[0][1]
      avg_y2 = sorted_cluster[-1][1]
      avg_x1 = 0
      avg_x2 = 0
      for tup in current_list:
        avg_x1 += tup[0]
        avg_x2 += tup[2]
      avg_x1 = avg_x1 / len(current_list)
      avg_x2 = avg_x2 / len(current_list)
      rects[i] = (avg_x1, avg_y1, avg_x2, avg_y2)
      i += 1

  #draw the rectangles
  buff = 7
  for key in rects:
    tup_topLeft = (int(rects[key][0] - buff), int(rects[key][1]))
    tup_botRight = (int(rects[key][2] + buff), int(rects[key][3]))
    cv2.rectangle(new_image, tup_topLeft, tup_botRight, (0, 255, 0), 3)
  return new_image, rects

rect_images = []
rect_coords = []
for image, lines in zip(test_images, list_of_lines):
  new_image, rects = identify_blocks(image, lines)
  rect_images.append(new_image)
  rect_coords.append(rects)
show_images(rect_images)  
print(rect_coords)

def draw_parking(image, rects, make_copy = True, color=[255, 0, 0], thickness=2, save = True):
    if make_copy:
        new_image = np.copy(image)
    gap = 15.5
    spot_dict = {} # maps each parking ID to its coords
    tot_spots = 0
    adj_y1 = {0: 20, 1:-10, 2:0, 3:-11, 4:28, 5:5, 6:-15, 7:-15, 8:-10, 9:-30, 10:9, 11:-32}
    adj_y2 = {0: 30, 1: 50, 2:15, 3:10, 4:-15, 5:15, 6:15, 7:-20, 8:15, 9:15, 10:0, 11:30}
    
    adj_x1 = {0: -8, 1:-15, 2:-15, 3:-15, 4:-15, 5:-15, 6:-15, 7:-15, 8:-10, 9:-10, 10:-10, 11:0}
    adj_x2 = {0: 0, 1: 15, 2:15, 3:15, 4:15, 5:15, 6:15, 7:15, 8:10, 9:10, 10:10, 11:0}
    for key in rects:
        # Horizontal lines
        tup = rects[key]
        x1 = int(tup[0]+ adj_x1[key])
        x2 = int(tup[2]+ adj_x2[key])
        y1 = int(tup[1] + adj_y1[key])
        y2 = int(tup[3] + adj_y2[key])
        cv2.rectangle(new_image, (x1, y1),(x2,y2),(0,255,0),2)
        num_splits = int(abs(y2-y1)//gap)
        for i in range(0, num_splits+1):
            y = int(y1 + i*gap)
            cv2.line(new_image, (x1, y), (x2, y), color, thickness)
        if key > 0 and key < len(rects) -1 :        
            #draw vertical lines
            x = int((x1 + x2)/2)
            cv2.line(new_image, (x, y1), (x, y2), color, thickness)
        # Add up spots in this lane
        if key == 0 or key == (len(rects) -1):
            tot_spots += num_splits +1
        else:
            tot_spots += 2*(num_splits +1)
            
        # Dictionary of spot positions
        if key == 0 or key == (len(rects) -1):
            for i in range(0, num_splits+1):
                cur_len = len(spot_dict)
                y = int(y1 + i*gap)
                spot_dict[(x1, y, x2, y+gap)] = cur_len +1        
        else:
            for i in range(0, num_splits+1):
                cur_len = len(spot_dict)
                y = int(y1 + i*gap)
                x = int((x1 + x2)/2)
                spot_dict[(x1, y, x, y+gap)] = cur_len +1
                spot_dict[(x, y, x2, y+gap)] = cur_len +2   
    
    print("total parking spaces: ", tot_spots, cur_len)
    if save:
        filename = 'with_parking.jpg'
        cv2.imwrite(filename, new_image)
    return new_image, spot_dict

delineated = []
spot_pos = []
for image, rects in zip(test_images, rect_coords):
    new_image, spot_dict = draw_parking(image, rects)
    delineated.append(new_image)
    spot_pos.append(spot_dict)
    
show_images(delineated)

final_spot_dict = spot_pos[1]

def assign_spots_map(image, spot_dict = final_spot_dict, make_copy = True, color = [255, 0, 0], thickness = 2):
  if make_copy:
    new_image = np.copy(image)
  for spot in spot_dict.keys():
    (x1, y1, x2, y2) = spot
    cv2.rectangle(new_image, (int(x1), int(y1)), (int(x2), int(y2)), color, thickness)
  return new_image  
marked_spots_images = list(map(assign_spots_map, test_images))
show_images(marked_spots_images)

### Save spot dictionary as pickle

import pickle
with open('spot_dict.pickle', 'wb') as handle:
  pickle.dump(final_spot_dict, handle, protocol = pickle.HIGHEST_PROTOCOL)

### Save image for CNN
def save_images_for_cnn(image, spot_dict = final_spot_dict, folder_name = 'for_cnn'):
  for spot in spot_dict.keys():
    (x1, y1, x2, y2) = spot
    (x1, y1, x2, y2) = (int(x1), int(y1), int(x2), int(y2))
    spot_img = image[y1:y2, x1:x2]
    spot_img = cv2.resize(spot_img, (0, 0), fx = 2.0, fy = 2.0)
    spot_id = spot_dict[spot]
    filename = 'spot' + str(spot_id) + '.jpg'
    # print(spot_img.shape, filename, (x1, y1, x2, y2))
    cv2.imwrite(os.path.join(folder_name, filename), spot_img)

save_images_for_cnn(test_images[0])

from PIL import Image
from keras.applications.imagenet_utils import preprocess_input
from keras.models import load_model
from keras.preprocessing import image

cwd = os.getcwd()
top_model_weights_path = 'car1.h5'

class_dictionary = {}
class_dictionary[0] = 'empty'
class_dictionary[1] = 'occupied'

model = load_model(top_model_weights_path)

def make_prediction(image):
  img = image/255.
  image = np.expand_dims(img, axis = 0)
  class_predicted = model.predict(image)
  intID = np.argmax(class_predicted[0])
  label = class_dictionary[intID]
  return label

def predict_on_image(image, spot_dict = final_spot_dict, make_copy = True, color = [0, 255, 0], alpha = 0.5):
  if make_copy:
    new_image = np.copy(image)
    overlay = np.copy(image)
  cnt_empty = 0
  all_spots = 0
  for spot in spot_dict.keys():
    all_spots += 1
    (x1, y1, x2, y2) = spot
    (x1, y1, x2, y2) = (int(x1), int(y1), int(x2), int(y2))
    spot_img = image[y1:y2, x1:x2]
    spot_img = cv2.resize(spot_img, (48, 48))
    label = make_prediction(spot_img)
    if(label == 'empty'):
      cv2.rectangle(overlay, (int(x1), int(y1)), (int(x2), int(y2)), color, -1)
      cnt_empty += 1
  cv2.addWeighted(overlay, alpha, new_image, 1-alpha, 0, new_image)
  cv2.putText(new_image, "Available: %d spots" %cnt_empty, (30, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
  cv2.putText(new_image, "Total: %d spots" %all_spots, (30, 125), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255 ,255), 2)
    
  save = False
  if save:
    filename = 'with_marking.jpg'
    cv2.imwrite(filename, new_image)
  return new_image

predicted_images = list(map(predict_on_image, test_images))    
show_images(predicted_images)    
