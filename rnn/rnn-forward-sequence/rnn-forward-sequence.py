import numpy as np

def rnn_forward(X: np.ndarray, h_0: np.ndarray,
                W_xh: np.ndarray, W_hh: np.ndarray, b_h: np.ndarray) -> tuple:
    """
    Forward pass through entire sequence.
    """
    # YOUR CODE HERE
    hidden_states=[] # for storing the all the hidden states
    h_prev=h_0

    T=X.shape[1]#no.of sequence length or max time steps
    
    for t in range(T):
        x_t= X[:,t,:]#input sentence  at current timestep(t)
        h_t=np.tanh(np.dot(h_prev,np.transpose(W_hh))+np.dot(x_t,np.transpose(W_xh))+b_h) #gets current hidden state output
        hidden_states.append(h_t)
        h_prev=h_t
    h_all=np.stack(hidden_states,axis=1)
    h_final=h_prev

    return h_all,h_final
                           
    
    pass