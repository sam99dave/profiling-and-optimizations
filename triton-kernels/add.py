import os

# use this for debugging & if incase GPU limit has been exhausted :(
# uncomment this is using GPU to avoid unwanted bugs
os.environ['TRITON_INTERPRET'] = '1' # needs to be set *before* triton is imported

import torch
import triton
import triton.language as tl

from utils import *

@triton.jit
def add_kernel(
    x_ptr, # internally prepared from torch.Tensor
    y_ptr, # internally prepared from torch.Tensor
    output_ptr, # internally prepared from torch.Tensor
    n_elements, # number of elements present in the Tensor (used for masking to prevent memory bound issues)
    BLOCK_SIZE: tl.constexpr # Size of the BLOCK which will be used by the kernel
  ):

  row_start = tl.program_id(0) # get the kernel info which is going to use a block | 0 -> is the axis (1D, 2D, 3D)

  # Whenever a kernel is launched it takes in a grid | Blocks inside this grid are individual compute units
  # preparing the offset, 0st block -> 0 * 3 + [0,1,2] = [0, 1, 2] (1st block locations)
  # preparing the offset, 1st block -> 1 * 3 + [0,1,2] = [3, 4, 5] (2nd block locations)
  offsets = row_start * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)  
  mask = offsets < n_elements # Masking to avoid going beyond total number of elements present

  x = tl.load(x_ptr + offsets, mask) # loading the vector of values of size BS
  y = tl.load(y_ptr + offsets, mask) # loading the vector of values of size BS

  output = x + y # Add operation on the vectors

  tl.store(output_ptr + offsets, output, mask) # Storing in the required output location 


def add(x: torch.Tensor, y: torch.Tensor):
  device = 'cuda' if torch.cuda.is_available() else 'cpu'
  z = torch.empty_like(x, device = device) # initializing the empty output
  # Uncomment the following assert if running on GPU
  # assert x.is_cuda and y.is_cuda and z.is_cuda, f"Inputs should be on CUDA"
  n_elements = x.numel()

  # Creating a function that returns a tuple, information on the grid -> no of blocks present in the grid
  grid = lambda meta: (cdiv(n_elements, meta['BLOCK_SIZE']), ) # The `meta` contains all the kwargs passed when calling the kernel
  add_kernel[grid](x, y, z, n_elements, BLOCK_SIZE = 1024) # call the kernel

  return z


if __name__ == '__main__':
    torch.manual_seed(0)
    size = 98432
    x = torch.rand(size, device='cpu')
    y = torch.rand(size, device='cpu')
    output_torch = x + y
    output_triton = add(x, y)
    print(output_torch)
    print(output_triton)
    print(f'The maximum difference between torch and triton is '
        f'{torch.max(torch.abs(output_torch - output_triton))}')