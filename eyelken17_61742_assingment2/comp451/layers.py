from builtins import range
import numpy as np


def affine_forward(x, w, b):
    """
    Computes the forward pass for an affine (fully-connected) layer.

    The input x has shape (N, d_1, ..., d_k) and contains a minibatch of N
    examples, where each example x[i] has shape (d_1, ..., d_k). We will
    reshape each input into a vector of dimension D = d_1 * ... * d_k, and
    then transform it to an output vector of dimension M.

    Inputs:
    - x: A numpy array containing input data, of shape (N, d_1, ..., d_k)
    - w: A numpy array of weights, of shape (D, M)
    - b: A numpy array of biases, of shape (M,)

    Returns a tuple of:
    - out: output, of shape (N, M)
    - cache: (x, w, b)
    """
    out = None
    
    ###########################################################################
    # TODO: Implement the affine forward pass. Store the result in out. You   #
    # will need to reshape the input into rows.                               #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    
    x_shape = x.shape[0]
    w_shape = w.shape[0]
    
    out = np.dot(np.reshape(x,(x_shape, w_shape)),w) + b
    
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    
    cache = (x, w, b)
    return out, cache


def affine_backward(dout, cache):
    """
    Computes the backward pass for an affine layer.

    Inputs:
    - dout: Upstream derivative, of shape (N, M)
    - cache: Tuple of:
      - x: Input data, of shape (N, d_1, ... d_k)
      - w: Weights, of shape (D, M)
      - b: Biases, of shape (M,)

    Returns a tuple of:
    - dx: Gradient with respect to x, of shape (N, d1, ..., d_k)
    - dw: Gradient with respect to w, of shape (D, M)
    - db: Gradient with respect to b, of shape (M,)
    """
    x, w, b = cache
    dx, dw, db = None, None, None
    ###########################################################################
    # TODO: Implement the affine backward pass.                               #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****    
    
    dx=np.dot(dout,w.T).reshape(x.shape)
    
    dw=np.dot(x.reshape(x.shape[0],-1).T,dout)
    
    db=np.sum(dout,axis=0)

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dw, db

def sigmoid_forward(x):
    """
    Computes the forward pass for a layer of sigmoid units.

    Input:
    - x: Inputs, of any shape

    Returns a tuple of:
    - out: Output, of the same shape as x
    - cache: x
    """
    out = None
    ###########################################################################
    # TODO: Implement the Sigmoid forward pass.                               #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    out = 1 / (1.0 + np.exp(-x))
    cache = out
    
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = x
    return out, cache

def sigmoid_backward(dout, cache):
    """
    Computes the backward pass for a layer of sigmoid units.

    Input:
    - dout: Upstream derivatives, of any shape
    - cache: Input x, of same shape as dout

    Returns:
    - dx: Gradient with respect to x 
    """
    dx, x = None, cache
    ###########################################################################
    # TODO: Implement the Sigmoid backward pass.                              #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    
    dx = dout * 1 / (1.0 + np.exp(-x)) * (1 - 1 / (1.0 + np.exp(-x)))
    
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx


def relu_forward(x):
    """
    Computes the forward pass for a layer of rectified linear units (ReLUs).

    Input:
    - x: Inputs, of any shape

    Returns a tuple of:
    - out: Output, of the same shape as x
    - cache: x
    """
    out = None
    ###########################################################################
    # TODO: Implement the ReLU forward pass.                                  #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    out = np.maximum(x, np.zeros(x.shape))

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = x
    return out, cache


def relu_backward(dout, cache):
    """
    Computes the backward pass for a layer of rectified linear units (ReLUs).

    Input:
    - dout: Upstream derivatives, of any shape
    - cache: Input x, of same shape as dout

    Returns:
    - dx: Gradient with respect to x
    """
    dx, x = None, cache
    ###########################################################################
    # TODO: Implement the ReLU backward pass.                                 #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    
    dx = dout * (x > 0)

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx


def leaky_relu_forward(x, lrelu_param):
    """
    Computes the forward pass for a layer of leaky rectified linear units (Leaky ReLUs).

    Input:
    - x: Inputs, of any shape
    - lrelu_param: Dictionary with the following key:
        - alpha: scalar value for negative slope

    Returns a tuple of:
    - out: Output, of the same shape as x
    - cache: (x, lrelu_param).
            Input x, of same shape as dout,
            lrelu_param, needed for backward pass.
    """
    out = None
    alpha = lrelu_param.get('alpha', 2e-3)
    ###########################################################################
    # TODO: Implement the Leaky ReLU forward pass.                            #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    flag = (x < 0)
    not_flag = (x >= 0)
    out = (flag * alpha * x) + not_flag * x 

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, lrelu_param)
    return out, cache


def leaky_relu_backward(dout, cache):
    """
    Computes the backward pass for a layer of leaky rectified linear units (Leaky ReLUs).
    Note that, the negative slope parameter (i.e. alpha) is fixed in this implementation.
    Therefore, you should not calculate any gradient for alpha.
    Input:
    - dout: Upstream derivatives, of any shape
    - cache: (x, lr_param)

    Returns:
    - dx: Gradient with respect to x
    """
    dx = None
    (x, lr_param) = cache
    alpha = lr_param["alpha"]
    ###########################################################################
    # TODO: Implement the Leaky ReLU backward pass.                           #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    
    flag = (x >= 0)
    not_flag = (x < 0)
    dx = np.multiply(flag,dout) + np.multiply(not_flag, alpha * dout)

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx


def dropout_forward(x, dropout_param):
    """
    Performs the forward pass for (inverted) dropout.

    Inputs:
    - x: Input data, of any shape
    - dropout_param: A dictionary with the following keys:
      - p: Dropout parameter. We keep each neuron output with probability p.
      - mode: 'test' or 'train'. If the mode is train, then perform dropout;
        if the mode is test, then just return the input.
      - seed: Seed for the random number generator. Passing seed makes this
        function deterministic, which is needed for gradient checking but not
        in real networks.

    Outputs:
    - out: Array of the same shape as x.
    - cache: tuple (dropout_param, mask). In training mode, mask is the dropout
      mask that was used to multiply the input; in test mode, mask is None.

    NOTE: Please implement **inverted** dropout, not the vanilla version of dropout.
    See http://cs231n.github.io/neural-networks-2/#reg for more details.

    NOTE 2: Keep in mind that p is the probability of **keep** a neuron
    output; this might be contrary to some sources, where it is referred to
    as the probability of dropping a neuron output.
    """
    p, mode = dropout_param['p'], dropout_param['mode']
    if 'seed' in dropout_param:
        np.random.seed(dropout_param['seed'])

    mask = None
    out = None

    if mode == 'train':
        #######################################################################
        # TODO: Implement training phase forward pass for inverted dropout.   #
        # Store the dropout mask in the mask variable.                        #
        #######################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        
        mask = (np.random.rand(* x.shape) < p) / p
        out = x * mask

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################
    elif mode == 'test':
        #######################################################################
        # TODO: Implement the test phase forward pass for inverted dropout.   #
        #######################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        mask = None
        out = x

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        #######################################################################
        #                            END OF YOUR CODE                         #
        #######################################################################

    cache = (dropout_param, mask)
    out = out.astype(x.dtype, copy=False)

    return out, cache


def dropout_backward(dout, cache):
    """
    Perform the backward pass for (inverted) dropout.

    Inputs:
    - dout: Upstream derivatives, of any shape
    - cache: (dropout_param, mask) from dropout_forward.
    """
    dropout_param, mask = cache
    mode = dropout_param['mode']

    dx = None
    if mode == 'train':
        #######################################################################
        # TODO: Implement training phase backward pass for inverted dropout   #
        #######################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        
        dx = dout * mask

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        #######################################################################
        #                          END OF YOUR CODE                           #
        #######################################################################
    elif mode == 'test':
        dx = dout
    return dx


def conv_forward_naive(x, w, b, conv_param):
    """
    A naive implementation of the forward pass for a convolutional layer.
    Note that, the filter is not flipped as in the regular convolution operation
    in signal processing domain. Therefore, technically this implementation
    is a cross-correlation.

    The input consists of N data points, each with C channels, height H and
    width W. We convolve each input with F different filters, where each filter
    spans all C channels and has height HH and width WW.

    Input:
    - x: Input data of shape (N, C, H, W)
    - w: Filter weights of shape (F, C, HH, WW)
    - b: Biases, of shape (F,)
    - conv_param: A dictionary with the following keys:
      - 'stride': The number of pixels between adjacent receptive fields in the
        horizontal and vertical directions.
      - 'pad': The number of pixels that will be used to zero-pad the input. 
        

    During padding, 'pad' zeros should be placed symmetrically (i.e equally on both sides)
    along the height and width axes of the input. Be careful not to modfiy the original
    input x directly.

    Returns a tuple of:
    - out: Output data, of shape (N, F, H', W') where H' and W' are given by
      H' = 1 + (H + 2 * pad - HH) / stride
      W' = 1 + (W + 2 * pad - WW) / stride
    - cache: (x, w, b, conv_param)
    """
    out = None
    ###########################################################################
    # TODO: Implement the convolutional forward pass.                         #
    # Hint: you can use the function np.pad for padding.                      #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    stride = conv_param['stride']
    pad = conv_param['pad']
    N,C,H,W = x.shape
    F,C,HH,WW = w.shape
    
    H1 = np.int32(1+ (H + 2 * pad - HH)/stride)
    W1 = np.int32(1+ (W + 2 * pad - WW)/stride)
    
    out = np.zeros((N,F,H1,W1))
    
    x_pad = np.zeros((N, C, H + 2 * pad, W + 2 * pad))
    for n in range(N):
        for c in range(C):
            x_pad[n][c]=np.pad(x[n][c], pad, 'constant', constant_values=0)
            
    for n in range(N):
        for f in range(F):
            for i in range(H1):
                for j in range(W1):
                    i1 = i * stride
                    i2 = i * stride + HH
                    j1 = j * stride
                    j2 = j * stride + WW
                    out[n, f, i, j] = np.sum(w[f] * x_pad[n, :, i1:i2, j1:j2]) + b[f]    
    
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, x_pad, w, b, conv_param)
    return out, cache


def conv_backward_naive(dout, cache):
    """
    A naive implementation of the backward pass for a convolutional layer.

    Inputs:
    - dout: Upstream derivatives.
    - cache: A tuple of (x, w, b, conv_param) as in conv_forward_naive

    Returns a tuple of:
    - dx: Gradient with respect to x
    - dw: Gradient with respect to w
    - db: Gradient with respect to b
    """
    dx, dw, db = None, None, None
    dx, dw, db = None, None, None
    ###########################################################################
    # TODO: Implement the convolutional backward pass.                        #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    x,x_pad,w,b,conv_param=cache
    pad=conv_param['pad']
    stride=conv_param['stride']
    
    N,F,H1,W1=dout.shape
    N,C,H,W=x.shape
    
    HH=w.shape[2]
    WW=w.shape[3]
    
    dx=np.zeros((N,C,H,W))
    dx_pad=np.zeros(x_pad.shape)
    
    dw=np.zeros(w.shape)
    db=np.zeros(b.shape)
    for n in range(N):
        for f in range(F):
            for i in range(H1):
                for j in range(W1):
                    i1 = i * stride
                    i2 = i * stride + HH
                    j1 = j * stride
                    j2 = j * stride + WW
                    
                    db[f]+=dout[n,f,i,j] 
                    dw[f]+=dout[n,f,i,j] * x_pad[n,:,i1:i2, j1:j2]
                    
                    dx_pad[n, :, i1:i2, j1:j2] += dout[n, f, i, j] * w[f]
    dx = dx_pad[:, :, pad:pad + H, pad:pad + W].copy()

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dw, db


def max_pool_forward_naive(x, pool_param):
    """
    A naive implementation of the forward pass for a max-pooling layer.

    Inputs:
    - x: Input data, of shape (N, C, H, W)
    - pool_param: dictionary with the following keys:
      - 'pool_height': The height of each pooling region
      - 'pool_width': The width of each pooling region
      - 'stride': The distance between adjacent pooling regions

    No padding is necessary here. Output size is given by 

    Returns a tuple of:
    - out: Output data, of shape (N, C, H', W') where H' and W' are given by
      H' = 1 + (H - pool_height) / stride
      W' = 1 + (W - pool_width) / stride
    - cache: (x, pool_param)
    """
    out = None
    ###########################################################################
    # TODO: Implement the max-pooling forward pass                            #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    N,C,H,W=x.shape
    HH=pool_param['pool_height']
    WW=pool_param['pool_width']
    stride=pool_param['stride']   
    
    H1=np.int32(1+(H-HH)/stride)
    W1=np.int32(1+(W-WW)/stride)
    out=np.zeros((N,C,H1,W1))
    for n in range(N):
        for c in range(C):
            for i in range(H1):
                for j in range(W1):
                    i1 = i * stride
                    i2 = i * stride + HH
                    j1 = j * stride
                    j2 = j * stride + WW
                    
                    out[n,c,i,j]=np.max(x[n, c, i1:i2, j1:j2])

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, pool_param)
    return out, cache


def max_pool_backward_naive(dout, cache):
    """
    A naive implementation of the backward pass for a max-pooling layer.

    Inputs:
    - dout: Upstream derivatives
    - cache: A tuple of (x, pool_param) as in the forward pass.

    Returns:
    - dx: Gradient with respect to x
    """
    dx = None
    ###########################################################################
    # TODO: Implement the max-pooling backward pass                           #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    x,pool_param=cache
    N,C,H,W=x.shape
    HH=pool_param['pool_height']
    WW=pool_param['pool_width']
    stride=pool_param['stride']
    
    H1 = np.int32(1 + (H - HH) / stride)
    W1 = np.int32(1 + (W - WW) / stride)
    
    dx=np.zeros(x.shape)
    for n in range(N):
        for c in range(C):
            for i in range(H1):
                for j in range(W1):
                    i1 = i * stride
                    i2 = i * stride + HH
                    j1 = j * stride
                    j2 = j * stride + WW
                    
                    index=np.where(x[n,c]==np.max(x[n, c, i1:i2, j1:j2]))
                    length=len(index[0])
                    
                    for m in range(length):
                        dx[n,c,index[0][m],index[1][m]]=dout[n,c,i,j]
                    

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx


def avg_pool_forward_naive(x, pool_param):
    """
    A naive implementation of the forward pass for a avg-pooling layer.

    Inputs:
    - x: Input data, of shape (N, C, H, W)
    - pool_param: dictionary with the following keys:
      - 'pool_height': The height of each pooling region
      - 'pool_width': The width of each pooling region
      - 'stride': The distance between adjacent pooling regions

    No padding is necessary here. Output size is given by 

    Returns a tuple of:
    - out: Output data, of shape (N, C, H', W') where H' and W' are given by
      H' = 1 + (H - pool_height) / stride
      W' = 1 + (W - pool_width) / stride
    - cache: (x, pool_param)
    """
    out = None
    ###########################################################################
    # TODO: Implement the avg-pooling forward pass                            #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    N, C, H, W = x.shape
    height = pool_param['pool_height']
    width = pool_param['pool_width']
    stride = pool_param['stride']
    
    HH = int(1 + (H - height) / stride)
    WW = int(1 + (W - width) / stride)
    
    out = np.zeros((N, C, HH, WW))
    
    for i in range(N):
        for h in range(HH):
            for w in range(WW):
                for c in range(C):
                    v1 = stride * h
                    v2 = v1 + height
                    h1 = stride * w
                    h2 = h1 + width
                    
                    slicex = x[i, c, v1:v2, h1:h2]
                    
                    out[i, c, h, w] = np.average(slicex)
                    
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, pool_param)
    return out, cache

def avg_pool_backward_naive(dout, cache):
    """
    A naive implementation of the backward pass for a avg-pooling layer.

    Inputs:
    - dout: Upstream derivatives
    - cache: A tuple of (x, pool_param) as in the forward pass.

    Returns:
    - dx: Gradient with respect to x
    """
    dx = None
    ###########################################################################
    # TODO: Implement the avg-pooling backward pass                           #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    x, pool_param = cache
    N, C, H, W = x.shape
    N, n_C, n_H, n_W = dout.shape
    height = pool_param['pool_height']
    width = pool_param['pool_width']
    stride = pool_param['stride']
    
    dx = np.zeros((N, C, H, W))
    
    for i in range(N):
        for h in range(n_H):
            for w in range(n_W):
                for c in range(n_C):
                    v1 = stride * h
                    v2 = v1 + height 
                    h1 = stride * w
                    h2 = h1 + width
                    
                    average = np.float(dout[i, c, h, w]) / np.float(height * width)
                    
                    dx[i, c, v1:v2, h1:h2] += np.ones((height, width)) * average
                    
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx




def svm_loss(x, y):
    """
    Computes the loss and gradient using for multiclass SVM classification.

    Inputs:
    - x: Input data, of shape (N, C) where x[i, j] is the score for the jth
      class for the ith input.
    - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
      0 <= y[i] < C

    Returns a tuple of
    - loss: Scalar giving the loss
    - dx: Gradient of the loss with respect to x
    """
    N = x.shape[0]
    correct_class_scores = x[np.arange(N), y]
    margins = np.maximum(0, x - correct_class_scores[:, np.newaxis] + 1.0)
    margins[np.arange(N), y] = 0
    loss = np.sum(margins) / N
    num_pos = np.sum(margins > 0, axis=1)
    dx = np.zeros_like(x)
    dx[margins > 0] = 1
    dx[np.arange(N), y] -= num_pos
    dx /= N
    return loss, dx


def softmax_loss(x, y):
    """
    Computes the loss and gradient for softmax classification.

    Inputs:
    - x: Input data, of shape (N, C) where x[i, j] is the score for the jth
      class for the ith input.
    - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
      0 <= y[i] < C

    Returns a tuple of:
    - loss: Scalar giving the loss
    - dx: Gradient of the loss with respect to x
    """
    shifted_logits = x - np.max(x, axis=1, keepdims=True)
    Z = np.sum(np.exp(shifted_logits), axis=1, keepdims=True)
    log_probs = shifted_logits - np.log(Z)
    probs = np.exp(log_probs)
    N = x.shape[0]
    loss = -np.sum(log_probs[np.arange(N), y]) / N
    dx = probs.copy()
    dx[np.arange(N), y] -= 1
    dx /= N
    return loss, dx
