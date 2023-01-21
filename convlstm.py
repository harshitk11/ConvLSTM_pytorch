import torch.nn as nn
import torch

# This script defines the architecture of the ConvLSTM.
# Reference to the Github repository : https://github.com/ndrplz/ConvLSTM_pytorch

# This is a single LSTM cell
# Dimensions of input tensor (which is an image) : Number of channels x Height of image x Width of image
class ConvLSTMCell(nn.Module):

    def __init__(self, input_dim, hidden_dim, kernel_size, bias):
        """
        Initialize ConvLSTM cell.

        Parameters
        ----------
        input_dim: int
            Number of channels of input tensor.
        hidden_dim: int
            Number of channels of hidden state.
        kernel_size: (int, int)
            Size of the convolutional kernel.
        bias: bool
            Whether or not to add the bias.
        """

        super(ConvLSTMCell, self).__init__()
        # In ConvLSTM, the inputs, cell states, hidden states, and the gates are all 3D tensors whose last two dimensions are spatial dimensions (rows and columns)

        self.input_dim = input_dim # Number of channels (not dimension) of the input tensor
        self.hidden_dim = hidden_dim # Number of channels (not dimension) of the hidden state tensor    

        self.kernel_size = kernel_size # Kernel size of the Convolution. (int, int) 2-D kernel
        self.padding = kernel_size[0] // 2, kernel_size[1] // 2  # // : Floor division - Rounds the result down to the nearest whole number. Padding = (Filter Size - 1)/2 to keep the output shape same as the input shape after convolution
        self.bias = bias

        # Initializing a 2D convolutional layer
        # We are performing two convolutions : Conv(x_t;W_x) [Convolution over the input tensor] and Conv(h_t-1;W_h) [convolution over the hidden state tensor]
        # So we perform the two convolutions together [Parrallelization]
        # On the output side, we need to perform the two convolutions for each of the input gate (i_t), forget gate (f_t), output gate (o_t), and intermediate cell state (g_t)
        # Therefore, for every channel in the input, we create 4 channels of output, in our Conv2d. We will separate the 4 channels in the output later.

        # Reference for Conv2d: https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html
        # NOTE: arguments are CHANNELS and not features
        self.conv = nn.Conv2d(in_channels=self.input_dim + self.hidden_dim,
                              out_channels=4 * self.hidden_dim,
                              kernel_size=self.kernel_size,
                              padding=self.padding,
                              bias=self.bias)

    # Defining the forward pass
    def forward(self, input_tensor, cur_state):
        # Shape of the tensor : Number of channels * H * W  (Last two dimensions are spatial dimensions)
        h_cur, c_cur = cur_state    # current hidden state, current cell state [They are both 3D tensor]

        combined = torch.cat([input_tensor, h_cur], dim=1)  # concatenate along channel axis (axis0 [batch] * axis1 [channel] * axis2 [height] * axis3 [width])

        combined_conv = self.conv(combined) # Convolve

        # Remember we had 4 output channels for the convolutions for it, ft, ot, gt
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1) # Split along the channel axis
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)

        return h_next, c_next

    # Use this to initialize the initial hidden state tensors in your code
    def init_hidden(self, batch_size, image_size):
        # Returns a zero initialized tensor for the cell state and the hidden state
        height, width = image_size # Spatial dimensions of your image (Must be a tuple)
        return (torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device),
                torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device))


# This creates a multi-layered LSTM. Each layer is defined by ConvLSTMCell class that is defined above.
class ConvLSTM(nn.Module):

    """

    Parameters:
        input_dim: Number of channels in input
        hidden_dim: Number of hidden channels  (If using multiple layers, then list of hidden_dim for each layer)
        kernel_size: Size of kernel in convolutions (If using multiple layers, then list of kernel_size for each layer. Each kernel_size is a tuple)
        num_layers: Number of LSTM layers stacked on each other
        batch_first: Whether or not dimension 0 is the batch or not
        bias: Bias or no bias in Convolution
        return_all_layers: Return the list of computations for all layers
        Note: Will do same padding.

    Input:
        A tensor of size B, T, C, H, W or T, B, C, H, W
    Output:
        A tuple of two lists of length num_layers (or length 1 if return_all_layers is False).
            0 - layer_output_list is the list of lists of length T of each output
            1 - last_state_list is the list of last states
                    each element of the list is a tuple (h, c) for hidden state and memory
    Example:
        >> x = torch.rand((32, 10, 64, 128, 128))
        >> convlstm = ConvLSTM(64, 16, 3, 1, True, True, False)
        >> _, last_states = convlstm(x)
        >> h = last_states[0][0]  # 0 for layer index, 0 for h index
    """

    def __init__(self, input_dim, hidden_dim, kernel_size, num_layers,
                 batch_first=False, bias=True, return_all_layers=False):
        super(ConvLSTM, self).__init__()

        # To check that kernel size is a tuple or a list of tuples
        self._check_kernel_size_consistency(kernel_size)

        # Make sure that both `kernel_size` and `hidden_dim` are lists having len == num_layers
        kernel_size = self._extend_for_multilayer(kernel_size, num_layers)
        hidden_dim = self._extend_for_multilayer(hidden_dim, num_layers)
        if not len(kernel_size) == len(hidden_dim) == num_layers: # Checks to make sure that we have hidden_dim and kernel_size for every layer
            raise ValueError('Inconsistent list length.')

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.bias = bias
        self.return_all_layers = return_all_layers
        
        # For the final output convolution
        self.out_conv_kernel = kernel_size[-1] # Using the same kernel size as the last output. Can change this.
        self.out_conv_padding = self.out_conv_kernel[0] // 2, self.out_conv_kernel[1] // 2  # Padding to keep the input and output size same

        cell_list = []
        for i in range(0, self.num_layers):
            cur_input_dim = self.input_dim if i == 0 else self.hidden_dim[i - 1] # If first layer, then input is the original input. Else input is the the output of the previous layer

            cell_list.append(ConvLSTMCell(input_dim=cur_input_dim,
                                          hidden_dim=self.hidden_dim[i],
                                          kernel_size=self.kernel_size[i],
                                          bias=self.bias))

        self.cell_list = nn.ModuleList(cell_list) # Holds submodules in a list: List of ConvLSTMCell. Each ConvLSTM cell represents one layer.

        # Convolution layer for converting hidden states over all the time steps (B x T x Ch x H x W) to output (B x T x Ci x H x W)
        # Since conv2d takes input of the form [B' x Ch x H x W] we will reshape the tensor such that B'= B x T 
        # Applying conv2d to the hidden states of the last layer only. Output will have same number of channels as that of the input i.e. Ci 
        self.out_conv = nn.Conv2d(in_channels=self.hidden_dim[-1],
                              out_channels=self.input_dim,
                              kernel_size=self.out_conv_kernel,
                              padding=self.out_conv_padding,
                              bias=self.bias)

    def forward(self, input_tensor, hidden_state=None):
        """

        Parameters
        ----------
        input_tensor: todo
            5-D Tensor either of shape (t, b, c, h, w) or (b, t, c, h, w)
        hidden_state: todo
            None. todo implement stateful

        Returns
        -------
        last_state_list, layer_output
        """
        temperature =1
        if not self.batch_first: # If batch first is not selected then reshape the input tensor
            # (t, b, c, h, w) -> (b, t, c, h, w)
            input_tensor = input_tensor.permute(1, 0, 2, 3, 4)

        b, _, _, h, w = input_tensor.size()

        # Implement stateful ConvLSTM
        if hidden_state is not None:
            # You need a list of last hidden state for each layer from the previous time chunk
            # [(h_last1,c_last1),(h_last2,c_last2),...]
            # raise NotImplementedError()
            hidden_state = hidden_state
        else:
            # Since the init is done in forward. Can send image size here
            # List of tuples of the initial hidden state for each layer [(h_init1,c_init1),(h_init2,c_init2),...]
            hidden_state = self._init_hidden(batch_size=b,
                                             image_size=(h, w))

        layer_output_list = [] # Output over all the time steps for all the layers. Each element contains output of all the time steps for one layer.
        last_state_list = [] # Hidden state after the last time step for all the layers. Each element contains [h,c] after the last time step for one layer.

        seq_len = input_tensor.size(1)  # Number of time steps in the input tensor (input tensor shape : b x t x c x h x w). To iterate over all the time
        
        cur_layer_input = input_tensor # Input to the first layer is the input_tensor

        for layer_idx in range(self.num_layers): # Iterating over all the layers

            h, c = hidden_state[layer_idx] # Initial hidden state of layer layer_idx
            output_inner = [] # Stores the output of a given layer for all the time steps
            for t in range(seq_len): 
                #h,c shape : b x c x h x w
                h, c = self.cell_list[layer_idx](input_tensor=cur_layer_input[:, t, :, :, :],
                                                 cur_state=[h, c])
                                                 
                output_inner.append(h) 

            layer_output = torch.stack(output_inner, dim=1) # Shape :B x T x Ch x H x W | Output (hidden states h) over all the time steps for this layer.
            
            cur_layer_input = layer_output # Output of this layer will be the input to the next layer

            layer_output_list.append(layer_output) # Append the hidden states of this layer to a list
            last_state_list.append([h, c]) # 

        if not self.return_all_layers:
            layer_output_list = layer_output_list[-1:] # return output of the last layer only. Includes all the time steps of the last layer.
            last_state_list = last_state_list[-1:] # return the hidden state of the last layer only.

        # Need output tensor in the shape B x T x C x H x W
        last_hidden_state = layer_output_list[-1] # b,t,Ch,h,w
        
        B,_,Ch,H,W = last_hidden_state.size()
        
        # final_out : B,T,Cin,H,W (For a given batch, we get output for all the timesteps in the chunk)
        final_out = (self.out_conv(last_hidden_state.reshape(-1,Ch,H,W))).reshape(B,-1,self.input_dim,H,W)
        final_out = torch.sigmoid(final_out/temperature)
        
        return layer_output_list, last_state_list, final_out

    # To initialize the hidden state for each layer.
    def _init_hidden(self, batch_size, image_size): 
        init_states = [] 
        for i in range(self.num_layers): # Need to initialize hidden state for each layer
            init_states.append(self.cell_list[i].init_hidden(batch_size, image_size)) 
        return init_states

    @staticmethod
    def _check_kernel_size_consistency(kernel_size):
        if not (isinstance(kernel_size, tuple) or
                (isinstance(kernel_size, list) and all([isinstance(elem, tuple) for elem in kernel_size]))):
            raise ValueError('`kernel_size` must be tuple or list of tuples')

    @staticmethod
    def _extend_for_multilayer(param, num_layers):
        if not isinstance(param, list): # If not a list, then extend to a list
            param = [param] * num_layers
        return param
