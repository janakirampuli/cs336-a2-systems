import triton
import torch
import triton.language as tl

DEVICE = torch.device(f'cuda:{torch.cuda.current_device()}')

'''
x -> (n, d)
w -> (d,)

o = x @ w -> (n,)

do = dL/do
dw = x^T @ do
dx = do @ w^T
'''

def weighted_sum(x, w):
    return x @ w

@triton.jit
def weighted_sum_fwd(
    x_ptr, w_ptr,
    o_ptr,
    x_stride_row, x_stride_dim,
    w_stride_dim,
    o_stride_row,
    ROWS, D,
    ROWS_TILE_SIZE: tl.constexpr, D_TILE_SIZE: tl.constexpr,
):
    row_tile_idx = tl.program_id(0)

    x_block_ptr = tl.make_block_ptr(
        x_ptr,
        shape=(ROWS, D,),
        strides=(x_stride_row, x_stride_dim),
        offsets=(row_tile_idx * ROWS_TILE_SIZE, 0),
        block_shape=(ROWS_TILE_SIZE, D_TILE_SIZE),
        order=(1, 0),
    )

    w_block_ptr = tl.make_block_ptr(
        w_ptr,
        shape=(D,),
        strides=(w_stride_dim, ),
        offsets=(0,),
        block_shape=(D_TILE_SIZE,),
        order=(0,),
    )

    o_block_ptr = tl.make_block_ptr(
        o_ptr,
        shape=(ROWS,),
        strides=(o_stride_row,),
        offsets=(row_tile_idx * ROWS_TILE_SIZE,),
        block_shape=(ROWS_TILE_SIZE,),
        order=(0,),
    )

    o = tl.zeros((ROWS_TILE_SIZE,), dtype=tl.float32)

    for i in range(tl.cdiv(D, D_TILE_SIZE)):
        row = tl.load(x_block_ptr, boundary_check=(0, 1), padding_option="zero")
        weight = tl.load(w_block_ptr, boundary_check=(0,), padding_option="zero")

        o += tl.sum(row * weight[None, :], axis=1)

        x_block_ptr = x_block_ptr.advance((0, D_TILE_SIZE))
        w_block_ptr = w_block_ptr.advance((D_TILE_SIZE,))
    
    tl.store(o_block_ptr, o, boundary_check=(0,))

@triton.jit
def weighted_sum_backward(
    x_ptr, w_ptr,
    grad_o_ptr,
    grad_x_ptr, partial_grad_w_ptr,
    x_stride_row, x_stride_dim,
    w_stride_dim,
    grad_o_stride_row,
    grad_x_stride_row, grad_x_stride_dim,
    partial_grad_w_stride_row, partial_grad_w_stride_dim,
    NUM_ROWS, D,
    ROWS_TILE_SIZE: tl.constexpr, D_TILE_SIZE: tl.constexpr,
):
    row_tile_idx = tl.program_id(0)
    n_rows_tiles = tl.num_programs(0)

    grad_o_block_ptr = tl.make_block_ptr(
        grad_o_ptr,
        shape=(NUM_ROWS,), strides=(grad_o_stride_row,),
        offsets=(row_tile_idx * ROWS_TILE_SIZE,),
        block_shape=(ROWS_TILE_SIZE,),
        order=(0,),
    )

    x_block_ptr = tl.make_block_ptr(
        x_ptr,
        shape=(NUM_ROWS, D),
        strides=(x_stride_row, x_stride_dim),
        offsets=(row_tile_idx * ROWS_TILE_SIZE, 0),
        block_shape=(ROWS_TILE_SIZE, D_TILE_SIZE),
        order=(1, 0),
    )

    w_block_ptr = tl.make_block_ptr(
        w_ptr,
        shape=(D,),
        strides=(w_stride_dim,),
        offsets=(0,),
        block_shape=(D_TILE_SIZE,),
        order=(0,),
    )

    grad_x_block_ptr = tl.make_block_ptr(
        grad_x_ptr,
        shape=(NUM_ROWS, D),
        strides=(grad_x_stride_row, grad_x_stride_dim),
        offsets=(row_tile_idx * ROWS_TILE_SIZE, 0),
        block_shape=(ROWS_TILE_SIZE, D_TILE_SIZE),
        order=(1, 0),
    )

    partial_w_grad_block_ptr = tl.make_block_ptr(
        partial_grad_w_ptr,
        shape=(n_rows_tiles, D),
        strides=(partial_grad_w_stride_row, partial_grad_w_stride_dim),
        offsets=(row_tile_idx, 0),
        block_shape=(1, D_TILE_SIZE),
        order=(1, 0),
    )

    for i in range(tl.cdiv(D, D_TILE_SIZE)):
        grad_o = tl.load(grad_o_block_ptr, boundary_check=(0,), padding_option="zero")

        w = tl.load(w_block_ptr, boundary_check=(0,), padding_option="zero")
        grad_x_row = grad_o[:, None] * w[None, :]
        tl.store(grad_x_block_ptr, grad_x_row, boundary_check=(0, 1))

        row = tl.load(x_block_ptr, boundary_check=(0, 1), padding_option="zero")
        grad_w_row = tl.sum(row * grad_o[:, None], axis=0)[None, :]
        tl.store(partial_w_grad_block_ptr, grad_w_row, boundary_check=(0, 1))

        x_block_ptr = x_block_ptr.advance((0, D_TILE_SIZE))
        w_block_ptr = w_block_ptr.advance((D_TILE_SIZE,))
        partial_w_grad_block_ptr = partial_w_grad_block_ptr.advance((0, D_TILE_SIZE))
        grad_x_block_ptr = grad_x_block_ptr.advance((0, D_TILE_SIZE))

class WeightedSumFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, w):
        n_rows, D = x.shape

        ctx.save_for_backward(x, w)

        ctx.D_TILE_SIZE = triton.next_power_of_2(D)
        ctx.ROWS_TILE_SIZE = 16
        ctx.input_shape = x.shape

        o = torch.empty((n_rows,), device=x.device, dtype=torch.float32)

        grid = (triton.cdiv(n_rows, ctx.ROWS_TILE_SIZE),)

        weighted_sum_fwd[grid](
            x, w,
            o,
            x.stride(0), x.stride(1),
            w.stride(0),
            o.stride(0),
            ROWS=n_rows, D=D,
            ROWS_TILE_SIZE=ctx.ROWS_TILE_SIZE, D_TILE_SIZE=ctx.D_TILE_SIZE,
        )
        return o

    @staticmethod
    def backward(ctx, grad_o):
        x, w = ctx.saved_tensors
        ROWS_TILE_SIZE, D_TILE_SIZE = ctx.ROWS_TILE_SIZE, ctx.D_TILE_SIZE
        n_rows, D = x.shape

        partial_grad_w = torch.empty((triton.cdiv(n_rows, ROWS_TILE_SIZE), D), device=x.device, dtype=x.dtype)
        grad_x = torch.empty_like(x)

        grid = (triton.cdiv(n_rows, ctx.ROWS_TILE_SIZE),)

        weighted_sum_backward[grid](
            x, w,
            grad_o,
            grad_x, partial_grad_w,
            x.stride(0), x.stride(1),
            w.stride(0),
            grad_o.stride(0),
            grad_x.stride(0), grad_x.stride(1),
            partial_grad_w.stride(0), partial_grad_w.stride(1),
            NUM_ROWS=n_rows, D=D,
            ROWS_TILE_SIZE=ROWS_TILE_SIZE, D_TILE_SIZE=D_TILE_SIZE,
        )
        # cpu vs tl.atomic_ad?
        # which is better? large N -> contention of all tiles to write in atomic_add
        grad_w = partial_grad_w.sum(axis=0)
        return grad_x, grad_w


weighted_sum_triton = WeightedSumFunc.apply

def test_weighted_sum(size: tuple, atol=1e-3, rtol=1e-3, device=DEVICE):
    torch.manual_seed(0)

    N, D = size[0], size[1]
    x_ref = torch.randn((N, D), device=device, dtype=torch.float32)
    w_ref = torch.randn((D,), device=device, dtype=torch.float32)

    x_triton = x_ref.detach().clone().requires_grad_(True)
    w_triton = w_ref.detach().clone().requires_grad_(True)
    x_torch = x_ref.detach().clone().requires_grad_(True)
    w_torch = w_ref.detach().clone().requires_grad_(True)

    triton_res = weighted_sum_triton(x_triton, w_triton)
    torch_res = weighted_sum(x_torch, w_torch[:, None]).squeeze(-1)

    torch.testing.assert_close(triton_res, torch_res, atol=atol, rtol=rtol)
    print("FWD PASSED")

    grad_o = torch.randn((N,), device=device, dtype=torch.float32)
    triton_res.backward(grad_o)
    torch_res.backward(grad_o)

    torch.testing.assert_close(x_triton.grad, x_torch.grad, atol=atol, rtol=rtol)
    torch.testing.assert_close(w_triton.grad, w_torch.grad, atol=atol, rtol=rtol)
    print("BWD PASSED")


if __name__ == "__main__":
    N = 345
    D = 220
    test_weighted_sum((N, D))