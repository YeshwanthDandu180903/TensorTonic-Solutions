import numpy as np

def rnn_cell(x_t: np.ndarray, h_prev: np.ndarray, 
             W_xh: np.ndarray, W_hh: np.ndarray, b_h: np.ndarray) -> np.ndarray:
    """
    Single RNN cell forward pass.
    """
    # YOUR CODE HERE
    h_t=np.tanh(np.dot(h_prev,W_hh)+np.dot(x_t,np.transpose(W_xh))+b_h)

    return h_t
    pass