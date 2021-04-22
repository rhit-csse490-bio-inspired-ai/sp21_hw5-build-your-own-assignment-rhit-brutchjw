# Image Recognition
Image recognition using NEAT.
To improve the runtime of the project, images were converted to 16 by 16 pixels, otherwise the neat-python library cannot handle the massive number of inputs.
# Various levels of image recognition
* `evolve01` Uses two distinct fruits (apples and bananas), where the images are converted to grayscale to test if the fruits' shapes can be distinguished.

* `evolve02` Uses two similarly shaped fruits (apples and oranges), where the images are now in full RGB color to test if similar shapes can be distinguished by color.

* `evolve03` Expands the dataset to include two more fruits, with full RGB color to test the performance and accuracy.    

* `evolve04` Expands the dataset further to include different types of the same fruits. For example testing recognition of Granny Smith apples versus Braeburn apples. 

* `evolve05` Image recognition using the full dataset of fruits and vegetables.
# How to use
Open the folders with the label "0x_xxx_evolution" and open the evolve0x.py file and run it. Plots will appear in the viz folder, and output data will appear in the output_files folder.
