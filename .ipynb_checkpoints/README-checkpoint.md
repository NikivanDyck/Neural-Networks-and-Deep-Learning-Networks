## Neural Networks and Deep Learning Networks
Module 13 Notes

![image_of_nn](https://miro.medium.com/max/583/1*DW0Ccmj1hZ0OvSXi7Kz5MQ.jpeg)

## Overview

Neural networks, also known as artificial neural networks (ANN), are a set of algorithms that are modeled after the human brain. They are an advanced form of machine learning that recognizes patterns and features of input data and provides a clear quantitative output.

In its simplest form, a neural network contains layers of neurons that perform individual computations. These computations are connected and weighed against one another until the neurons reach the final layer. In the final layer, the neurons return either a numerical result or an encoded categorical result.

In the finaance industry you can use nural networks for fraud detection, risk management, money laundering prevetion, algorithmic trading.   

---

## Principles of Neural Networks 
### Input Layer
The input layer is visiable data in its raw form. Consists of values transformed by weight coefficients.


### Activation Layer/ Activiation functions (AKA: Hidden Layer)
The activation function is a mathematical function applied to the end of each neuron (that is, each individual perceptron model). This function transforms each neuron’s output into a quantitative value. The quantitative output value is then used as the input value for the next layer in the neural network model. Although activation functions can introduce both linear and nonlinear properties to a neural network, nonlinear activation functions are more common.  ML Glossary’s “Activation Function” sections (Link https://ml-cheatsheet.readthedocs.io/en/latest/activation_functions.html)

### Output Layer
The output layer takes in the inputs which are passed in from the layers before it, performs the calculations via its neurons and then the output is computed.  The output layer is responsible for producing the final result. 

---

## Tools and/or Imports Used 
Jupyter, Pandas, Numpy, Pathlib, Sklearn, Imblearn, tensorflow, keras

---

## Appendix:  
List of terms definitions and code used to complete the analysis

| Term | Description | links |
| :---: | :----------- | :--- |
|TensorFlow|TensorFlow is an end-to-end open-source platform for machine learning. It allows us to run our code across multiple platforms in a highly efficient way. |  |  
|Keras| Keras is an abstraction layer on top of TensorFlow that makes it easier to build models.|---|
|Neuron| a single network that contains input vlaues, weights, bias, summary, output | |
|Linear function |The linear function returns the sum of the weighted inputs without transformation. The linear activation function allows for the multiple outputs Example: 1–10 wine (rather than just a binary 0 or 1).| https://ml-cheatsheet.readthedocs.io/en/latest/activation_functions.html#linear |
|Sigmoid function|The sigmoid function transforms the neuron’s output to a range between 0 and 1, which is especially useful for predicting probabilities. A neural network that uses the sigmoid function will output a model with a characteristic S curve.  | https://ml-cheatsheet.readthedocs.io/en/latest/activation_functions.html#sigmoid |
|tanh function| The tanh function transforms the output to a range between −1 and 1, and the resulting model also forms a characteristic S curve. The tanh function is primarily used for classification between two classes. | https://ml-cheatsheet.readthedocs.io/en/latest/activation_functions.html#tanh | 
|Optimization function| This function shapes and molds a neural network model while the model is trained on the data, to ensure that the model performs to the best of its ability. | ---|
|Gradient descent| Gradient descent is an optimization algorithm. Neural networks use it to identify the combination of function parameters that will allow the model to learn as efficiently as possible, until it has learned everything it can. When gradient descent works properly, the model learns the greatest amount in the early iterations. The amount learned declines with each iteration until the optimization algorithm approaches the local minimum value, or the point where it cannot learn anything additional. The number of model iterations required for the model to learn everything it can varies widely—and is often only discovered through trial and error. |--- |
|Evaluation Metrics| There are two main evaluation metrics: model predictive accuracy and model mean squared error (MSE). We use model predictive accuracy (accuracy) for classification models, and we use MSE (mse) for regression models. For classification models, the highest possible accuracy value is 1. A higher accuracy value indicates more accurate predictions. However, for regression models, we want the MSE to reduce to zero. The closer to 0 our MSE is, the more accurate the model’s predictions are.| https://www.tensorflow.org/api_docs/python/tf/keras/metrics| 
|ReLU |The rectified linear unit (ReLU) function returns a value from 0 to infinity. This activation function transforms any negative input to 0. It is the most commonly used activation function in neural networks due to its faster learning and simplified output. However, it is not always appropriate for simpler models. |https://ml-cheatsheet.readthedocs.io/en/latest/activation_functions.html#relu |
|Leaky ReLU |The leaky ReLU function is a “leaky” alternative to the ReLU function. This means that instead of transforming negative input values to 0, it transforms negative input values into much smaller negative values. | https://ml-cheatsheet.readthedocs.io/en/latest/activation_functions.html#leakyrelu |
|Softmax | The softmax function calculates the probabilities distribution of the event over ‘n’ different events. In general way of saying, this function will calculate the probabilities of each target class over all possible target classes. Later the calculated probabilities will be helpful for determining the target class for the given inputs.|https://ml-cheatsheet.readthedocs.io/en/latest/activation_functions.html#softmax |


## Footnotes: 
https://ml-cheatsheet.readthedocs.io/en/latest/activation_functions.html#softmax
https://www.youtube.com/watch?v=-7scQpJT7uo&t=331s
https://towardsdatascience.com/machine-learning-fundamentals-ii-neural-networks-f1e7b2cb3eef
