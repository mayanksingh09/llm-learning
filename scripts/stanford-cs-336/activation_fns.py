import numpy as np


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def relu(x):
    return np.maximum(0, x)

def tanh(x):
    return np.tanh(x)

def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=0)

def swish(x):
    return x * sigmoid(x)

class GLU:
    """
    Gated Linear Unit (Numpy, forward-only)
    y = (x * W_v + b_v) ⊙ sigmoid(x * W_g + b_g)
    """

    def __init__(self, input_feature_dim: int, output_feature_dim: int, random_number_generator: np.random.Generator = np.random):

        # fan_avg is the average of fan-in (in_features) and fan-out (out_features)
        # This is used for Xavier/Glorot initialization to maintain variance across layers
        fan_avg = (input_feature_dim + output_feature_dim) / 2.0
        std = np.sqrt(2 / fan_avg)

        self.W_v = random_number_generator.randn(input_feature_dim, output_feature_dim) * std
        self.b_v = np.zeros(output_feature_dim)
        self.W_g = random_number_generator.randn(input_feature_dim, output_feature_dim) * std
        self.b_g = np.zeros(output_feature_dim)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        v = x @ self.W_v + self.b_v
        g = sigmoid(x @ self.W_g + self.b_g)
        return v * g
    

class SwiGLU:
    """
    Swish-Gated Linear Unit
    y = (x * W_v + b_v) ⊙ swish(x * W_g + b_g)
    """

    def __init__(self, input_feature_dim: int, output_feature_dim: int, random_number_generator: np.random.Generator = np.random):
        fan_avg = (input_feature_dim + output_feature_dim) / 2.0
        std = np.sqrt(2 / fan_avg)

        self.W_v = random_number_generator.randn(input_feature_dim, output_feature_dim) * std
        self.b_v = np.zeros(output_feature_dim)
        self.W_g = random_number_generator.randn(input_feature_dim, output_feature_dim) * std
        self.b_g = np.zeros(output_feature_dim)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        v = x @ self.W_v + self.b_v
        g = swish(x @ self.W_g + self.b_g)
        return v * g


if __name__ == "__main__":
    # relu, tanh, softmax, swish will preserve the shape of the input
    # glu and swiglu will have a different shape than the input if d_out is different
        # they can project the input to a different dimension (if needed)

    batch, d_in, d_out = 4, 128, 128 # batch size, input feature dimension, output feature dimension
    
    x = np.random.randn(batch, d_in)

    print("Input: \n", x[:5, :5])
    print("Input shape: ", x.shape)
    print("-" * 50)

    y_relu = relu(x)
    print("ReLU: \n", y_relu[:5, :5])
    print("ReLU shape: ", y_relu.shape)
    print("-" * 50)
    
    y_tanh = tanh(x)
    print("Tanh: \n", y_tanh[:5, :5])
    print("Tanh shape: ", y_tanh.shape)
    print("-" * 50)

    y_softmax = softmax(x)
    print("Softmax: \n", y_softmax[:5, :5])
    print("Softmax shape: ", y_softmax.shape)
    print("-" * 50)
    
    y_swish = swish(x)
    print("Swish: \n", y_swish[:5, :5])
    print("Swish shape: ", y_swish.shape)
    print("-" * 50)
    
    glu_layer = GLU(d_in, d_out)
    y_glu = glu_layer(x)
    print("GLU: \n", y_glu[:5, :5])
    print("GLU shape: ", y_glu.shape)
    print("-" * 50)
    
    swiglu_layer = SwiGLU(d_in, d_out)
    y_swiglu = swiglu_layer(x)
    print("SwiGLU: \n", y_swiglu[:5, :5])
    print("SwiGLU shape: ", y_swiglu.shape)
    print("-" * 50)