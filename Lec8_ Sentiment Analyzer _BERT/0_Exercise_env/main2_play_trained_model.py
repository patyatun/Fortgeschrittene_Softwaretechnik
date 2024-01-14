import os
import numpy as np
import pandas as pd
# import matplotlib.pyplot as plt

# Get and print the current working directory
current_working_directory = os.getcwd()
print(f"The current working directory is: {current_working_directory}")

# Change the working directory to a new directory (replace with the path you want)
new_working_directory = "C:/(...)/Transformers - BERT/src"
os.chdir(new_working_directory)

# Get and print the new current working directory
new_current_working_directory = os.getcwd()
print(f"The new current working directory is: {new_current_working_directory}")


#  ----------------------------------------------------------------------------
from transformers import pipeline

# load trained, serialized model
newmodel = pipeline('text-classification', model='my_saved_model') #, device=0)

# Sentiment: positive
newmodel('This movie is great!')

# Sentiment: negative
newmodel('This movie sucks')

# Sentiment: positive
newmodel('This movie is not bad')

# Sentiment: positive
newmodel('This movie is not that bad')

# Sentiment: positive
newmodel('I want to see it again')

# Sentiment: negative
newmodel('I don\'t want to see it again')


