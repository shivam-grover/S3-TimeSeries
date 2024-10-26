import torch
from torch import nn
import torch.nn.functional as F
import torch.nn.init as init

class S3(nn.Module):
    """
    Stack of S3 layers.

    Args:
        num_layers (int): Number of S3 layers to stack.
        initial_num_segments (int): Number of segments for the first S3 layer. If segments_per_layer is provided, this will be ignored.
        shuffle_vector_dim (int): Dimensionality of the shuffle vector for each S3 layer.
        segment_multiplier (float): Multiplier for the number of segments in each consecutive layer. Only used if segments_per_layer is not provided.
        segments_per_layer (List[int], optional): A list specifying the number of segments for each layer. Overrides initial_num_segments and segment_multiplier if provided.
        use_conv_w_avg (bool): If True, use convolution-based weighted average. Otherwise, use learnable parameters for weighted averaging.
        initialization_type (str): Initialization type for shuffle vectors ('kaiming' or 'manual').
        use_stitch (bool): If True, apply stitching to combine original and shuffled sequences.
    """
    def __init__(self, num_layers, initial_num_segments, shuffle_vector_dim=1, segment_multiplier=1, segments_per_layer=None, use_conv_w_avg=True, initialization_type="kaiming", use_stitch=True):
        super(S3, self).__init__()
        self.S3_layers = nn.ModuleList()
        next_segment_num = initial_num_segments
        for i in range(0, num_layers):
            print("Building S3 Layer", i)
            self.S3_layers += [S3Layer(num_segments=next_segment_num, shuffle_vector_dim=shuffle_vector_dim, use_conv_w_avg=use_conv_w_avg, 
               initialization_type=initialization_type, use_stitch=use_stitch)]
            next_segment_num = int(next_segment_num * segment_multiplier)
            if next_segment_num==0:
                next_segment_num=1
    
    def forward(self, x):
        x_copy = x.clone()
        for S3_layer in self.S3_layers:
            # Only process if the input sequence length is greater or equal to the number of segments
            if x_copy.shape[1] >= S3_layer.num_segments:
                # Calculate how many time steps to truncate
                sample_num_to_truncate = x_copy.shape[1] % S3_layer.num_segments
                # If truncation is needed, slice the input to make it divisible by num_segments
                if sample_num_to_truncate > 0:
                    x_copy = x_copy[:, sample_num_to_truncate:, :]

                # Pass through the current S3 layer
                x_copy = S3_layer(x_copy)
        # Concatenate the truncated part of the original input with the processed part
        if x.shape[1] > x_copy.shape[1]:
            x_copy = torch.cat([x[:, 0:x.shape[1] - x_copy.shape[1], :], x_copy], dim=1)

        return x_copy

class S3Layer(nn.Module):
    def __init__(self, num_segments, shuffle_vector_dim=1, use_conv_w_avg=True, initialization_type="kaiming", use_stitch=True):
        """
        Segment-Shuffle-Stitch Layer (S3).

        Args:
            num_segments (int): Number of segments to divide the input into. This determines how the input is split along the time dimension.
            shuffle_vector_dim (int): Dimensionality of the shuffle vector, which defines the complexity of the learned shuffling. 
                                    For example, if num_segments=4 and shuffle_vector_dim=2, the shape of the shuffle vector will be (4, 4).
            use_conv_w_avg (bool): If True, use convolution-based weighted average. Otherwise, use learnable scalar weights.
            initialization_type (str): Initialization method for the shuffle vector ('kaiming' for Kaiming initialization, 'manual' for manual initialization).
            use_stitch (bool): If True, stitch the shuffled output with the original input using a weighted average.
        """
        super(S3Layer, self).__init__()
        self.num_segments = int(num_segments)
        self.activation = "relu"
        self.use_conv_w_avg = use_conv_w_avg
        self.initialization_type = initialization_type
        self.use_stitch = use_stitch

        # This decides how many dimensions a shuffle vector will have
        # For example, if num_segments -> 4
        #   ex 1. if shuffle_vector_dim = 1
        #       then shape of shuffle_vector is (4)

        #   ex 2. if shuffle_vector_dim = 2
        #       then shape of shuffle_vector is (4,4)

        #   ex 3. if shuffle_vector_dim = 3
        #       then shape of shuffle_vector is (4,4,4)

        # The idea was to add some complexity to the shuffle_vector tensor so that it can learn more complex relationships if necessary
        self.shuffle_vector_dim = shuffle_vector_dim

        # Code to make shuffle vector dimension dynamic
        # I just add num_segments to a tuple as many times as the shuffle_vector_dimension variable is
        # Here the shuffle_vector_shape could be one of the following depending on the value of shuffle vector dim:
        #   (n)
        #   (n,n)
        #   (n,n,n)
        #   (n,n,n,n)
        shuffle_vector_shape = tuple([self.num_segments] * self.shuffle_vector_dim)

        # I create an empty parametric tensor
        # It's empty because I will add weights to it using He initialsation (or manual if the user asks for it)
        self.shuffle_vector = nn.Parameter(torch.empty(shuffle_vector_shape))
        self.initialize_shuffle_vector()
        self.descending_indices = None

        if self.use_conv_w_avg:
            self.w_avg = nn.Conv1d(in_channels=2, out_channels=1, kernel_size=1)
        else:
            # Learnable scalars for weighted average if not using conv
            # Start with the same weight (0.5) to assign equal weightage to both shuffled and original sequence initially
            self.weights = nn.Parameter(torch.ones(2) * 0.5)  # Initialize with two random values for w1, w2

    def initialize_shuffle_vector(self):
        """Initialize the shuffle vector based on the user-specified initialization type."""
        if self.initialization_type == "kaiming" and self.shuffle_vector_dim > 1:
            # He initialization for higher-dimensional tensors
            init.kaiming_normal_(self.shuffle_vector, mode='fan_out', nonlinearity=self.activation)
            # Scale and shift values
            scale_factor, shift_value = 0.001, 0.01
            self.shuffle_vector.data.mul_(scale_factor).add_(shift_value)

        elif self.initialization_type == "manual" or self.shuffle_vector_dim == 1:
            # Manual initialization with user-defined scale and shift
            scale_factor, shift_value = 0.1, 0.5
            self.shuffle_vector.data.fill_(scale_factor).add_(shift_value)

        else:
            raise ValueError(f"Unsupported initialization type: {self.initialization_type}")

    def forward(self, x):

        # Ensure shuffle_vector is on the same device as the input tensor
        if self.shuffle_vector.device != x.device:
            self.shuffle_vector = self.shuffle_vector.to(x.device)

        # Total time steps means how many data points are there in the input sequence
        total_time_steps = x.size(1)
        # # Now we know the number of steps in the input sequence
        # # And we know how many segments to divide them into using num_segments hyperparameter passed to the model
        # # So we calculate how many steps should be in each segment
        # steps_per_segment = total_time_steps // self.shuffle_vector.size(0)
        # # The code below divides the input sequence into n segments and returns a list
        # segments = [x[:, i * steps_per_segment: (i + 1) * steps_per_segment, :] for i in range(self.shuffle_vector.size(0))]

        # Recently I replaced the above three lines of code with torch.chunk for better readability and good practice. 
        # However, the torch.chunk function may return fewer than the specified number of chunks and also return the last chunk incomplete.
        # So if you face issues with that, comment the next lines out, and uncomment the previous code.
        segments = torch.chunk(x, self.num_segments, dim=1)
        
        # Now the logic I use to re-arrange the segments is explained below using a simple 1 dimensional shuffle vector:
        # 
        #   Let's say the num_segments is 4 
        #   and shuffle_vector for this iteration is [0.01, 0.05, 0.06, 0.005]

        #   Take the index of the largest number and put segment at that index in the first position
        #       Index of largest weight -> 2
        #       Put segment at index 2 in the first position

        #   Then take the index of the second largest number and put the segment at that index in the second position
        #       Index of next largest weight -> 1
        #       Put segment at index 1 in the second position
        
        #   Do this for all segments and at the end
        #   you will have a segment list that is shuffled according to the shuffle vector


        # The above example was for a single dimensional tensor
        # If the shuffle vector is higher dimensional, then take the sum of each row in certain dimensions so that it becomes one dimensional in the end
        if len(self.shuffle_vector.shape)>1:
            shuffle_vector_sum = self.shuffle_vector.sum(tuple(range(len(self.shuffle_vector.shape)-1)))
        else:
            shuffle_vector_sum = self.shuffle_vector
        
        # Now get the list of indices in the descending order of the weight values
        # So if shuffle vector is [0.01, 0.05, 0.06, 0.005]
        # The descending indices are [2,1,0,3]

        # So if shuffle_vector (or shuffle_vector_sum) is [0.05, 0.06, 0.006, 0.0001, 0.005, 0.01]
        # The descending indices are [1, 0, 5, 2, 4, 3]
        self.descending_indices = torch.argsort(shuffle_vector_sum, descending=True)
        
        # Simply re-arranging the segments using the descending_indices tensor does not flow gradients through the shuffle vectors
        # So the code below helps in redirecting the gradient flow through it through some hacks

        # Create an intermediate tensor of zeros with the shape (n,n) where n is the number of segments
        result_matrix = torch.zeros((len(shuffle_vector_sum), len(shuffle_vector_sum)), device=x.device)

        # Now fill in the values in each row according to the descending indices.
        # So if shuffle_vector (or shuffle_vector_sum) is [0.01, 0.05, 0.06, 0.005]
        # The descending indices are [2,1,0,3]

        # The result matrix will be
        # 0,        0,      0.06,   0
        # 0,        0.05,   0,      0
        # 0.01,     0,      0,      0
        # 0,        0,      0,      0.005
        
        # for index, i in enumerate(descending_indices):
        #     result_matrix[index][i] = shuffle_vector_sum[i]

        result_matrix.scatter_(1, self.descending_indices.unsqueeze(1), shuffle_vector_sum.unsqueeze(1))

        # The code below will convert the non-zero elements in the result matrix to 1
        non_zero_mask = result_matrix != 0
        scaling_factors = 1.0 / torch.abs(result_matrix[non_zero_mask])
        result_matrix[non_zero_mask] *= scaling_factors
        result_matrix = torch.abs(result_matrix)

        # The result matrix will be
        # 0,        0,      1,      0
        # 0,        1,      0,      0
        # 1,        0,      0,      0
        # 0,        0,      0,      1

        # Stack the list of segments into a tensor with an extra dimension
        # If input shape ->     (b, t, c)        and num_segments -> n
        # Then stack shape->    (b, t/n, c, n)
        stacked_segments = torch.stack(segments, dim=-1)
        
        # Add an extra dimension to help us with row wise matrix multiplication between result matrix and stacked_segments
        stacked_segments = stacked_segments.unsqueeze(-1).expand(-1, -1, -1, -1, stacked_segments.shape[-1])
        # stack shape->    (b, t, c, n, n)

        # The idea is that we can assume each segment as a single scalar element in the matrix (since everything inside a segment remains the same throughout), 
        # and then perform dot product of each segment with the corresponding element in one row of result matrix

        # For n = 4
        # segments -> [S0, S1, S2, S3] (they are all tensors but let's treat them as a single element)
        # 
        # Result Matrix ->
        # [
            # [0,0,1,0],
            # [0,1,0,0],
            # [1,0,0,0],
            # [0,0,0,1],
        # ]

        # So we want to multiply like this:
        #   1. segments * result_matrix[0] (dot product of segments and 0th row of result_matrix)           ->          S0*0 + S1*0 + S2*1 + S3*0 = S2
        #   2. segments * result_matrix[1] (dot product of segments and 1st row of result_matrix)           ->          S0*0 + S1*1 + S2*0 + S3*0 = S1
        #   3. segments * result_matrix[2] (dot product of segments and 2nd row of result_matrix)           ->          S0*1 + S1*0 + S2*0 + S3*0 = S0
        #   4. segments * result_matrix[3] (dot product of segments and 3rd row of result_matrix)           ->          S0*0 + S1*0 + S2*0 + S3*1 = S3

        # What this will do is, for each row it will only retain the segment whose index had a 1 in the result matrix.

        # Dot Product
        ## Multiply the matrices
        multiplication_out = stacked_segments * torch.transpose(result_matrix, 0, 1)
        multiplication_out = torch.transpose(multiplication_out, -2, -1)
        ## Get the sum
        shuffled_segments_stack = multiplication_out.sum(dim=-1)
        
        # Remove extra dimension we had added
        shuffled_segments_list = shuffled_segments_stack.unbind(dim=-1)
        
        # Concatenate the shuffled segments into the original input shape
        # shape of concatenated_segments -> (b, t, c)
        concatenated_segments = torch.cat(shuffled_segments_list, dim=1)

        # Now the concatenated_segments tensor has the final shuffled sequence

        b,t,c = x.shape

        if(self.use_stitch):
            stacked_shuffle_original = torch.stack((concatenated_segments, x), dim=-1)

            if self.use_conv_w_avg:
                # Use convolution-based weighted average
                stacked_shuffle_original_reshaped = stacked_shuffle_original.view(b * t * c, 2, 1)
                out = self.w_avg(stacked_shuffle_original_reshaped)
                out = out.view(b, t, c)
                # out shape -> (b, t, c)
                return out
            else:
                # Use parameter-based weighted average (with softmax constraint)
                weights_normalized = torch.softmax(self.weights, dim=0)  # Ensure weights sum to 1
                out = (stacked_shuffle_original * weights_normalized).sum(dim=-1)
                # out shape -> (b, t, c)
            return out
        else:
            return concatenated_segments

    def __repr__(self):
        return f"S3(num_segments={self.num_segments}, shuffle_vector_dim={self.shuffle_vector_dim}, initialization_type={self.initialization_type}, use_conv_w_avg={self.use_conv_w_avg})"