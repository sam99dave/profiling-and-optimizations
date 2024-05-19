import torch
import torch.utils.benchmark as benchmark

def batched_dot_mul_sum(a, b):
    '''Computes batched dot by multiplying and summing'''
    return a.mul(b).sum(-1)


def batched_dot_bmm(a, b):
    '''Computes batched dot by reducing to ``bmm``'''
    a = a.reshape(-1, 1, a.shape[-1])
    b = b.reshape(-1, b.shape[-1], 1)
    return torch.bmm(a, b).flatten(-3)

# Implementing using both `reference` & `value`  

batched_dot_src = """\
/* ---- Python ---- */
// def batched_dot_mul_sum(a, b):
//     return a.mul(b).sum(-1)

torch::Tensor batched_dot_mul_sum_v0(
    const torch::Tensor a,
    const torch::Tensor b) {
  return a.mul(b).sum(-1);
}

torch::Tensor batched_dot_mul_sum_v1(
    const torch::Tensor& a,
    const torch::Tensor& b) {
  return a.mul(b).sum(-1);
}
"""


# PyTorch makes it easy to test our C++ implementations by providing a utility
# to JIT compile C++ source into Python extensions:
import os
from torch.utils import cpp_extension
cpp_lib = cpp_extension.load_inline(
    name='cpp_lib',
    cpp_sources=batched_dot_src,
    extra_cflags=['-O3'],
    extra_include_paths=[
        # `load_inline` needs to know where to find ``pybind11`` headers.
        # os.path.join(os.getenv('CONDA_PREFIX'), 'include')
    ],
    functions=['batched_dot_mul_sum_v0', 'batched_dot_mul_sum_v1'],
    build_directory='./tmp'
)

# `load_inline` will create a shared object that is loaded into Python. When we collect
# instruction counts Timer will create a subprocess, so we need to re-import it. The
# import process is slightly more complicated for C extensions, but that's all we're
# doing here.
module_import_str = f"""\
# https://stackoverflow.com/questions/67631/how-to-import-a-module-given-the-full-path
import importlib.util
spec = importlib.util.spec_from_file_location("cpp_lib", {repr(cpp_lib.__file__)})
cpp_lib = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cpp_lib)"""

import textwrap
def pretty_print(result):
    """Import machinery for ``cpp_lib.so`` can get repetitive to look at."""
    print(repr(result).replace(textwrap.indent(module_import_str, "  "), "  import cpp_lib"))


t_baseline = benchmark.Timer(
    stmt='batched_dot_mul_sum(x, x)',
    setup='''\
from __main__ import batched_dot_mul_sum
x = torch.randn(2, 2)''')

t0 = benchmark.Timer(
    stmt='cpp_lib.batched_dot_mul_sum_v0(x, x)',
    setup=f'''\
{module_import_str}
x = torch.randn(2, 2)''')

t1 = benchmark.Timer(
    stmt='cpp_lib.batched_dot_mul_sum_v1(x, x)',
    setup=f'''\
{module_import_str}
x = torch.randn(2, 2)''')

# Moving to C++ did indeed reduce overhead, but it's hard to tell which
# calling convention is more efficient. v1 (call with references) seems to
# be a bit faster, but it's within measurement error.
pretty_print(t_baseline.blocked_autorange())
pretty_print(t0.blocked_autorange())
pretty_print(t1.blocked_autorange())


# Let's use ``Callgrind`` to determine which is better.
stats_v0 = t0.collect_callgrind()
stats_v1 = t1.collect_callgrind()

pretty_print(stats_v0)
pretty_print(stats_v1)

# `.as_standardized` removes file names and some path prefixes, and makes
# it easier to read the function symbols.
stats_v0 = stats_v0.as_standardized()
stats_v1 = stats_v1.as_standardized()

# `.delta` diffs the instruction counts, and `.denoise` removes several
# functions in the Python interpreter that are known to have significant
# jitter.
delta = stats_v1.delta(stats_v0).denoise()

# `.transform` is a convenience API for transforming function names. It is
# useful for increasing cancelation when ``diff-ing`` instructions, as well as
# just generally improving readability.
replacements = (
    ("???:void pybind11", "pybind11"),
    ("batched_dot_mul_sum_v0", "batched_dot_mul_sum_v1"),
    ("at::Tensor, at::Tensor", "..."),
    ("at::Tensor const&, at::Tensor const&", "..."),
    ("auto torch::detail::wrap_pybind_function_impl_", "wrap_pybind_function_impl_"),
)
for before, after in replacements:
    delta = delta.transform(lambda l: l.replace(before, after))

# We can use print options to control how much of the function to display.
torch.set_printoptions(linewidth=160)

# Once parsed, the instruction counts make clear that passing `a` and `b`
# by reference is more efficient as it skips some ``c10::TensorImpl`` bookkeeping
# for the intermediate Tensors, and is also works better with ``pybind11``. This
# is consistent with our noisy wall time observations.
print(delta)