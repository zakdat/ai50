Initially struggled with tensorflow installation, as unsupported on python 3.9. Had to install virtual environment
package and run Python 3.8.
Began by creating convolutional network based on lecture source code for handwriting.py, using 32 filters and a 
3x3 kernel, with one hidden layer of 128 filters. Accuracy stagnated around 5%.
Added a hidden layer, changed both hidden layers to 32 units. Accuracy not much improved.
Added a third hidden layer - organised units in hidden layers in decreasing numbers (number of outputs multiplied 
by 32, 16 and 8 respectively). Much better accuracy ~ 95%, but slow run time (90ms/step).
Halved number of units in each hidden layer. Accuracy ~ 93%, runtime halved (46ms/step).
Increased pool size in pool layer to 3x3. Slightly decreased accuracy, faster runtime (27ms/step).
Adding an additional convolutional and pooling layer decreased accuracy without affecting runtime.
Added dropout to visual layer and increased dropout in hidden layers to 0.5. Decreased accuracy to <80%.
Increased filters in convolutional layer to 64. Accuracy ~80%. Long run time.
Added convolutional layer with 8 filters - second back to 32. No change, removed additional layer.
Decreased dropout in each layer to 0.2. Accuracy ~ 94%, reasonable run time, loss ~ 0.28.
