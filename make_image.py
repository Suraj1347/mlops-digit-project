import numpy as np
from PIL import Image
from sklearn import datasets

print("Generating image...")
digits = datasets.load_digits()

# Grab a '4' from the training dataset
img_data = digits.images[4] 

# Convert it to standard image colors and invert it (black text on white background)
img_data = (img_data / 16.0) * 255
img_data = 255 - img_data 

# Save it as a normal-sized PNG file
img = Image.fromarray(img_data.astype(np.uint8)).convert('L')
img = img.resize((200, 200), Image.Resampling.NEAREST)
img.save("perfect_4.png")

print("Success! Check your folder for perfect_4.png")