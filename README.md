# Micrograd

This is a Micrograd-based neural-network framework to uncover the core mathematics of deep learning. The implementation constructs scalar computational graphs, performs forward propagation, computes loss, applies reverse-mode backpropagation, accumulates gradients, updates parameters through simple optimization, and model save and load. 

## Example : Decimal Digit Recognizer 
To test the framework, a 64–32–10 multilayer perceptron was trained to recognize decimal digits from 8×8 images. The model achieved approximately 93%  test accuracy after 30 epochs (training time: 77mins53.3seconds) on sampling dataset (sklearn digits data), demonstrating the practical effectiveness of micrograd.
- location: `./example/decimal_digits`

## How to run 
The whole project is setup using poetry : read the pyproject.toml for required information.

```bash
# make poetry is installed in your device and then run
 poetry lock && poetry install
```

## Thank you
Andrej Karpathy
(ref: [karpathy/micrograd](https://github.com/karpathy/micrograd))
