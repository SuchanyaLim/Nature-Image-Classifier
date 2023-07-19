#Naive Bayes classifier
# implementation based on given set of priors and probabilities

import numpy
from PIL import Image

priors = [0.03, 0.10, 0.11, 0.76]
settings = ["tundra", "forest", "desert", "ocean"]

#probabilities "channel value | setting", in the following order: R | tundra, G | tundra, B | tundra,
# then rgb for forest, rgb for desert, rgb for ocean
# if color channel > 128
p_high_rgb_values = [(0.85, 0.71, 0.89), (0.53, 0.88, 0.12), (0.94, 0.06, 0.03), (0.18, 0.27, 0.98)]
#if color channel < 128
p_low_rgb_values = [(0.15, 0.29, 0.11), (0.47, 0.12, 0.88), (0.06, 0.94, 0.97), (0.82, 0.73, 0.02)]

def classifier(input_filepath):
  '''
  implementation of a naive bayes classifier
    Parameters:
      input_filepath: the full file path to a JPG file containing an RGB image
    Returns:
      most_likely_class: a string indicating the most likely class, either “tundra”, “forest”, “desert”, or “ocean”
      class_probabilities: a four element list indicating the probability of each class in the order [tundra, forest, desert, ocean]
  '''
  # Getting image RBG
  img = numpy.asarray(Image.open(input_filepath))
  rgb_list = numpy.mean(img, axis=(0,1))
  red = rgb_list[0]
  green = rgb_list[1]
  blue = rgb_list[2]
  
  # Computing all unknown elements to plug in in the naive bayes classifier formula
  #step 1: compute 4 P(R∧G∧B) | i, where i = tundra, forest, desert, ocean
  rgb_given_setting = p_rgb_given_setting(red, green, blue)

  #step2: compute the denominator of the formula P(R∧G∧B) using values obtained in step 1 and prior probailities
  p_rgb = compute_p_rgb(rgb_given_setting)

  #step3: compute 4 P (i | P(R∧G∧B), where where i = tundra, forest, desert, ocean, using values obtained in steps 1,2 and
  #prior probabilities
  class_probabilities = compute_p_setting_given_rgb(rgb_given_setting, p_rgb)

  #step 4: find the most likely setting for the input image
  most_likely_class = find_max_p(class_probabilities)

  return most_likely_class, class_probabilities

# step 1
def p_rgb_given_setting(r,g,b):
  list_rgb_given_setting =[]

  for i in range (4):
    j=0
    if r<128:
      first_num = p_low_rgb_values[i][j]
    else:
      first_num = p_high_rgb_values[i][j]
    if g < 128:
      second_num = p_low_rgb_values[i][j+1]
    else:
      second_num = p_high_rgb_values[i][j+1]
    if b < 128:
      third_num = p_low_rgb_values[i][j+2]
    else:
      third_num = p_high_rgb_values[i][j+2]

    product = first_num*second_num*third_num
    list_rgb_given_setting.append(product)

  return list_rgb_given_setting

#step2
def compute_p_rgb(list_rgb_given_setting):
  denominator = 0
  for i in range (4):
    denominator+=(list_rgb_given_setting[i]*priors[i])
  return denominator

#step3
def compute_p_setting_given_rgb(rgb_given_setting, denominator):
  list_setting_given_rgb = []
  for i in range(4):
    p_setting_given_rgb = (rgb_given_setting[i]*priors[i])/denominator
    list_setting_given_rgb.append(p_setting_given_rgb)

  return list_setting_given_rgb

#step4
def find_max_p(list_setting_given_rgb):
  max_p = max(list_setting_given_rgb)
  for i in range (4):
    if list_setting_given_rgb[i]==max_p:
      setting = settings[i]
  return setting
