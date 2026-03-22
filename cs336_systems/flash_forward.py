import torch
import torch.nn.functional as F
import math
import triton
import triton.language as tl

class TorchFlashAttention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q, K, V, is_causal=False):
        B, N_q, D = Q.shape
        _, N_k, _ = K.shape

        # Bq -> tile size for Q
        # Bk -> tile size for K
        B_q = 32
        B_k = 32

        O = torch.zeros_like(Q)
        L = torch.zeros((B, N_q), device=Q.device, dtype=torch.float32)

        scale = 1.0 / math.sqrt(D)

        T_q = math.ceil(N_q / B_q)
        T_k = math.ceil(N_k / B_k)

        for i in range(T_q):
            row_start = i * B_q
            row_end = min((i + 1) * B_q, N_q)

            Q_i = Q[ :, row_start:row_end, :]

            m_i = torch.full((B, row_end - row_start, 1), float("-inf"), device=Q.device, dtype=torch.float32)
            l_i = torch.zeros((B, row_end - row_start, 1), device=Q.device, dtype=torch.float32)
            O_i = torch.zeros((B, row_end - row_start, D), device=Q.device, dtype=torch.float32)

            for j in range(T_k):
                col_start = j * B_k
                col_end = min((j + 1) * B_k, N_k)

                if is_causal and col_start > row_end - 1:
                    continue

                K_j = K[:, col_start:col_end, :]
                V_j = V[:, col_start:col_end, :]

                S_ij = torch.matmul(Q_i, K_j.transpose(-2, -1)) * scale

                if is_causal:
                    r_idx = torch.arange(row_start, row_end, device=Q.device).view(-1, 1)
                    c_idx = torch.arange(col_start, col_end, device=Q.device).view(1, -1)
                    mask = r_idx >= c_idx
                    S_ij = torch.where(mask, S_ij, torch.tensor(float("-inf"), device=Q.device))

                row_max_S_ij = torch.max(S_ij, dim=-1, keepdim=True)[0]
                m_ij = torch.maximum(row_max_S_ij, m_i)

                P_ij = torch.exp(S_ij - m_ij)

                l_ij = torch.exp(m_i - m_ij) * l_i + torch.sum(P_ij, dim=-1, keepdim=True)

                O_ij = torch.exp(m_i - m_ij) * O_i + torch.matmul(P_ij.to(V.dtype), V_j)

                m_i = m_ij
                l_i = l_ij
                O_i = O_ij
            
            O_i = O_i / l_i

            l_i = (m_i + torch.log(l_i)).squeeze(-1)

            O[:, row_start:row_end, :] = O_i
            L[:, row_start:row_end] = l_i

        ctx.save_for_backward(Q, K, V, O, L)
        ctx.is_causal = is_causal

        return O
    
    @staticmethod
    def backward(ctx, dO):
        raise NotImplementedError('not implemented yet')

@triton.jit
def flash_attn_fwd_kernel(
    Q_ptr, K_ptr, V_ptr,
    O_ptr, L_ptr,
    stride_qb, stride_qq, stride_qd,
    stride_kb, stride_kq, stride_kd,
    stride_vb, stride_vq, stride_vd,
    stride_ob, stride_oq, stride_od,
    stride_lb, stride_lq,
    N_QUERIES, N_KEYS,
    scale,
    D: tl.constexpr,
    Q_TILE_SIZE: tl.constexpr,
    K_TILE_SIZE: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
):
    query_tile_index = tl.program_id(0)
    batch_index = tl.program_id(1)

    Q_block_ptr = tl.make_block_ptr(
        base=Q_ptr + batch_index * stride_qb,
        shape=(N_QUERIES, D),
        strides=(stride_qq, stride_qd),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )

    K_block_ptr = tl.make_block_ptr(
        base=K_ptr + batch_index * stride_kb,
        shape=(N_KEYS, D),
        strides=(stride_kq, stride_kd),
        offsets=(0, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0),
    )

    V_block_ptr = tl.make_block_ptr(
        base=V_ptr + batch_index * stride_vb,
        shape=(N_KEYS, D),
        strides=(stride_vq, stride_vd),
        offsets=(0, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0),
    )

    O_block_ptr = tl.make_block_ptr(
        base=O_ptr + batch_index * stride_ob,
        shape=(N_QUERIES, D),
        strides=(stride_oq, stride_od),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )

    m_i = tl.full([Q_TILE_SIZE], float("-inf"), dtype=tl.float32)
    l_i = tl.zeros([Q_TILE_SIZE], dtype=tl.float32)
    O_i = tl.zeros([Q_TILE_SIZE, D], dtype=tl.float32)

    Q_tile = tl.load(Q_block_ptr)

    T_k = tl.cdiv(N_KEYS, K_TILE_SIZE)

    if IS_CAUSAL:
        causal_T_k = tl.cdiv((query_tile_index + 1) * Q_TILE_SIZE, K_TILE_SIZE)
        if causal_T_k < T_k:
            T_k = causal_T_k

    q_idx = query_tile_index * Q_TILE_SIZE + tl.arange(0, Q_TILE_SIZE)

    for j in range(T_k):
        k_start = j * K_TILE_SIZE

        K_tile = tl.load(K_block_ptr)
        V_tile = tl.load(V_block_ptr)

        S_ij = tl.dot(Q_tile, tl.trans(K_tile)) * scale

        if IS_CAUSAL:
            k_idx = k_start + tl.arange(0, K_TILE_SIZE)
            mask = q_idx[:, None] >= k_idx[None, :]
            S_ij = tl.where(mask, S_ij, float("-inf"))

        m_ij = tl.maximum(m_i, tl.max(S_ij, axis=1))
        P_ij = tl.exp(S_ij - m_ij[:, None])

        l_ij = tl.exp(m_i - m_ij) * l_i + tl.sum(P_ij, axis=1)

        O_i = O_i * tl.exp(m_i - m_ij)[:, None]
        O_i = tl.dot(P_ij.to(V_tile.dtype), V_tile, acc=O_i)

        m_i = m_ij
        l_i = l_ij

        K_block_ptr = tl.advance(K_block_ptr, (K_TILE_SIZE, 0))
        V_block_ptr = tl.advance(V_block_ptr, (K_TILE_SIZE, 0))

    O_i = O_i / l_i[:, None]
    l_i = m_i + tl.log(l_i)

    tl.store(O_block_ptr, O_i.to(O_block_ptr.type.element_ty))

    L_ptr_offsets = L_ptr + batch_index * stride_lb + q_idx * stride_lq
    tl.store(L_ptr_offsets, l_i)


class TritonFlashAttention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q, K, V, is_causal=False):

        B, N_q, D = Q.shape
        _, N_k, _ = K.shape

        scale = 1.0 / math.sqrt(D)

        Q_TILE_SIZE = 32
        K_TILE_SIZE = 32

        O = torch.zeros_like(Q)
        L = torch.zeros((B, N_q), device=Q.device, dtype=torch.float32)

        grid = (triton.cdiv(N_q, Q_TILE_SIZE), B)

        flash_attn_fwd_kernel[grid](
            Q, K, V,
            O, L,
            Q.stride(0), Q.stride(1), Q.stride(2),
            K.stride(0), K.stride(1), K.stride(2),
            V.stride(0), V.stride(1), V.stride(2),
            O.stride(0), O.stride(1), O.stride(2),
            L.stride(0), L.stride(1),
            N_q, N_k,
            scale,
            D,
            Q_TILE_SIZE,
            K_TILE_SIZE,
            is_causal
        )

        ctx.save_for_backward(Q, K, V, O, L)
        ctx.is_causal = is_causal
        
        return O

    @staticmethod
    def backward(ctx, dO):
        raise NotImplementedError('not implemented yet')

torch_flash_attn = TorchFlashAttention.apply
triton_flash_attn = TritonFlashAttention.apply


def test_flash_attention():
    B, N_q, D = 4, 256, 64
    N_k = N_q

    Q = torch.randn(B, N_q, D, requires_grad=True, dtype=torch.float32, device="cuda")
    V = torch.randn(B, N_k, D, requires_grad=True, dtype=torch.float32, device="cuda")
    K = torch.randn(B, N_k, D, requires_grad=True, dtype=torch.float32, device="cuda")

    O_ref = F.scaled_dot_product_attention(Q, K, V, is_causal=True)

    O_torch = torch_flash_attn(Q, K, V, True)
    torch.testing.assert_close(O_torch, O_ref, atol=1e-4, rtol=1e-4)
    print("pytorch fwd passed")

    O_triton = triton_flash_attn(Q, K, V, True)
    torch.testing.assert_close(O_triton, O_ref, atol=5e-3, rtol=5e-3)
    print("triton fwd passed")

if __name__ == "__main__":
    test_flash_attention()
