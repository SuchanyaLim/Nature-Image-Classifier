# Fuzzy Classifier
# implementation based on given fuzzy sets and fuzzy rules

import numpy
from PIL import Image

#variables
highest_membership_class = ""   
# highest_membership_class is a string indicating the highest membership class, either “tundra”, “forest”, “desert”, or “ocean”
class_memberships = []          
# class_memberships is a four element list indicating the membership in each class in the order [tundra value, forest value, desert value, ocean value]

#values for each fuzzy set and characteristic
red_low = [0, 0, 85, 125]
red_med = [85, 125, 130, 190]
red_high = [130, 190, 255, 255]

green_low = [0, 0, 60, 120]
green_med = [60, 120, 125, 185]
green_high = [125, 185, 255, 255]

blue_low = [0, 0, 55, 130]
blue_med = [55, 130, 140, 190]
blue_high = [140, 190, 255, 255]


def classifier(input_filepath):
  '''
  implementation of a fuzzy classifier
    Parameters:
      input_filepath: the full file path to a JPG file containing an RGB image
    Returns:
      highest_membership_class: a string indicating the most likely class, either “tundra”, “forest”, “desert”, or “ocean”
      class_memberships: a four element list indicating the probability of each class in the order [tundra, forest, desert, ocean]
  '''
  #get image values
  img = numpy.asarray(Image.open(input_filepath))
  img_r = numpy.mean(img[:,:,0])
  img_g = numpy.mean(img[:,:,1])
  img_b = numpy.mean(img[:,:,2])
  
  # step 1. Compute fuzzy truth value of each proposition
  m_rl = get_fuzzy_value(img_r, red_low)
  m_rm = get_fuzzy_value(img_r, red_med)
  m_rh = get_fuzzy_value(img_r, red_high)

  m_gl = get_fuzzy_value(img_g, green_low)
  m_gm = get_fuzzy_value(img_g, green_med)
  m_gh = get_fuzzy_value(img_g, green_high)

  m_bl = get_fuzzy_value(img_b, blue_low)
  m_bm = get_fuzzy_value(img_b, blue_med)
  m_bh = get_fuzzy_value(img_b, blue_high)

  # step 2. Compute fuzzy rule strength
  #rules
  # IF red_high             AND green_high AND blue_high               THEN tundra.
  rule_tundra = t(t(m_rh, m_gh), m_bh)

  # IF (red_low OR red_med) AND green_high AND (blue_low OR blue_med)  THEN forest.
  rule_forest = t(t(s(m_rl, m_rm), m_gh), s(m_bl, m_bm))

  # IF red_high             AND green_low  AND blue_low                THEN desert.
  rule_desert = t(t(m_rh, m_gl), m_bl)

  # IF red_low              AND blue_high                              THEN ocean.
  rule_ocean = t(m_rl, m_bh)
  
  class_memberships = [rule_tundra, rule_forest, rule_desert, rule_ocean]

  # step 3. find class with highest membership - the prediction
  highest_membership_class = find_class(class_memberships)

  return highest_membership_class, class_memberships

def get_fuzzy_value(input, values):
  '''
    Calculates fuzzy truth value based on given input
      Parameters:
        input: the image r/g/b value
        values: an array containing the values that will be used in the following function
      Returns:
        fuzz: the calculated fuzzy truth value
    The function:
            0 if x<=a
            x-a / b-a if a<x<b      
    f(x) =  1 if b<=x<=c
            d-x / d-c if c<x<d
            0 if d<=x
  '''
  fuzz = 0

  if (input <= values[0]):
    fuzz = 0
  elif (input < values[1]):
    fuzz = (input-values[0])/(values[1]-values[0])
  elif (input <= values[2]):
    fuzz = 1
  elif (input < values[3]):
    fuzz = (values[3]-input)/(values[3]-values[2])
  else:
    fuzz = 0
  
  return fuzz

def t(x, y):
  '''
  Calculates the Goguen t-norm
  '''
  return x*y

  # Godel t-norm
  #return min(x,y)

def s(x, y):
  '''
  Calculates the Goguen s-norm
  '''
  return (x+y)-(x*y)

  # Godel s-norm
  #return max(x,y)

def find_class(classes):
  '''
  Finds the class with the highest membership value based on outputs of fuzzy rule strength
    Parameters:
      classes: the list cotaining possible classes
    Returns:
      a string indicating the class with the highest membership value, either “tundra”, “forest”, “desert”, or “ocean”
  '''
  max_value_index = 0
  for i in range(len(classes)):
    if (classes[i] > classes[max_value_index]):
      max_value_index = i

  if (max_value_index == 0): return "tundra"
  if (max_value_index == 1): return "forest"
  if (max_value_index == 2): return "desert"
  if (max_value_index == 3): return "ocean"
