import os

# use this for debugging & if incase GPU limit has been exhausted :(
# uncomment this is using GPU to avoid unwanted bugs
os.environ['TRITON_INTERPRET'] = '1' # needs to be set *before* triton is imported

import torch
import triton
import triton.language as tl

from utils import *


@triton.jit
def subtract_kernel(
    x_ptr,
    y_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr 
  ):

  row_start = tl.program_id(0)

  offsets = row_start * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
  mask = offsets < n_elements

  x = tl.load(x_ptr + offsets, mask)
  y = tl.load(y_ptr + offsets, mask)

  output = x - y

  tl.store(output_ptr + offsets, output, mask)

def subtract(x: torch.Tensor, y: torch.Tensor):
  z = torch.empty_like(x, device = x.device) # initializing the empty output
  # Uncomment the following assert if running on GPU
  # assert x.is_cuda and y.is_cuda and z.is_cuda, f"Inputs should be on CUDA"
  n_elements = x.numel()

  grid = lambda meta: (cdiv(n_elements, meta['BLOCK_SIZE']), )
  subtract_kernel[grid](x, y, z, n_elements, BLOCK_SIZE = 1024)

  return z


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.manual_seed(0)
    size = 98432
    x = torch.rand(size, device=device)
    y = torch.rand(size, device=device)
    output_torch = x - y
    output_triton = subtract(x, y)
    print(output_torch)
    print(output_triton)
    print(f'The maximum difference between torch and triton is '
        f'{torch.max(torch.abs(output_torch - output_triton))}')