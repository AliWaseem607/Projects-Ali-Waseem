# Web Page Interaction
## Background
In the following `.ipynb` file is the code I used to interact with a webpage hosted by the teaching assistants of the course 'Energy Converstion and Renewable Energy'.
For our project in the course we had to analyze the impacts of various energy mixes on the energy future of switzerland and the teaching assistants provided a model to use for predicting energy consumption and energy mixed based on an input excel workbook that we provided.
The excel workbook was uploaded to a webpage which then ran the calculations after which the page would allow you download a results excel workbook. 
My team wanted to run a fine grain analysis by running several simulations with small changes in inputs through the model interface.

## Coding Solution
To perform this fine grain analysis I decided to create some code that could upload the inputs, run the model, and then download and store the results rather than do it manually.
The code uses selenium to interact with the webpage and uses the os library to move aorund files.