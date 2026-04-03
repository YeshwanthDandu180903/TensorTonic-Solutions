import numpy as np

def bptt_single_step(dh_next: np.ndarray, h_t: np.ndarray, h_prev: np.ndarray,
                     x_t: np.ndarray, W_hh: np.ndarray) -> tuple:
    """
    Backprop through one RNN time step.
    Returns (dh_prev, dW_hh).
    """
    # YOUR CODE HERE
    
    # dh_next → gradient of loss w.r.t current hidden state (∂L/∂h_t)

    # h_t → current hidden state at time t
    # h_prev → previous hidden state at time t-1
    # x_t → current input at time t
    # W_hh → recurrent weight matrix connecting h_prev → h_t

    # Step 1: Backprop through tanh activation
    dh_raw = dh_next * (1 - h_t**2)
    # dh_raw → gradient after passing through tanh derivative

    # Step 2: Gradient flowing back to previous hidden state
    dh_prev = dh_raw @ W_hh
    # dh_prev → ∂L/∂h_prev

    # Step 3: Gradient of loss w.r.t recurrent weights
    dW_hh = dh_raw.T @ h_prev
    # dW_hh → ∂L/∂W_hh

    return dh_prev, dW_hh
    
    pass