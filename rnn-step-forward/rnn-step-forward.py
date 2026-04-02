import numpy as np

def rnn_step_forward(x_t, h_prev, Wx, Wh, b):
    """
    Returns: h_t of shape (H,)
    """
    # Write code here
    h_t=np.tanh(np.dot(h_prev,Wh)+np.dot(x_t,Wx)+b)
    return h_t
    pass
