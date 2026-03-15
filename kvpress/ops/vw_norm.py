import torch
import triton
import triton.language as tl
import time



def get_config(M, D):
    if torch.cuda.get_device_capability() == (8, 0):
        if D <= 64:
            BLOCK_M, BLOCK_N, num_stages, num_warps = 128, 64, 3, 4
        else:
            if M <= 1024:
                BLOCK_M, BLOCK_N, num_stages, num_warps = 128, 32, 3, 4
            else:
                BLOCK_M, BLOCK_N, num_stages, num_warps = 128, 128, 3, 8
    elif torch.cuda.get_device_capability() == (8, 6):
        if D <= 64:
            BLOCK_M, BLOCK_N, num_stages, num_warps = 128, 64, 3, 4
        else:
            BLOCK_M, BLOCK_N, num_stages, num_warps = 128, 32, 2, 4
    else:
        BLOCK_M, BLOCK_N, num_stages, num_warps = 32, 32, 1, 4
    return (BLOCK_M, BLOCK_N, num_stages, num_warps)




# @triton.autotune(configs=[
#     triton.Config(kwargs={'BLOCK_M': 128, 'BLOCK_N': 16}, num_stages=1, num_warps=4),
#     triton.Config(kwargs={'BLOCK_M': 128, 'BLOCK_N': 16}, num_stages=2, num_warps=4),
#     triton.Config(kwargs={'BLOCK_M': 128, 'BLOCK_N': 16}, num_stages=3, num_warps=4),
#     triton.Config(kwargs={'BLOCK_M': 128, 'BLOCK_N': 16}, num_stages=4, num_warps=4),

#     triton.Config(kwargs={'BLOCK_M': 128, 'BLOCK_N': 32}, num_stages=1, num_warps=4),
#     triton.Config(kwargs={'BLOCK_M': 128, 'BLOCK_N': 32}, num_stages=2, num_warps=4),
#     triton.Config(kwargs={'BLOCK_M': 128, 'BLOCK_N': 32}, num_stages=3, num_warps=4),
#     triton.Config(kwargs={'BLOCK_M': 128, 'BLOCK_N': 32}, num_stages=4, num_warps=4),

#     triton.Config(kwargs={'BLOCK_M': 128, 'BLOCK_N': 64}, num_stages=1, num_warps=4),
#     triton.Config(kwargs={'BLOCK_M': 128, 'BLOCK_N': 64}, num_stages=2, num_warps=4),
#     triton.Config(kwargs={'BLOCK_M': 128, 'BLOCK_N': 64}, num_stages=3, num_warps=4),
#     triton.Config(kwargs={'BLOCK_M': 128, 'BLOCK_N': 64}, num_stages=4, num_warps=4),

#     triton.Config(kwargs={'BLOCK_M': 64, 'BLOCK_N': 32}, num_stages=1, num_warps=4),
#     triton.Config(kwargs={'BLOCK_M': 64, 'BLOCK_N': 32}, num_stages=2, num_warps=4),
#     triton.Config(kwargs={'BLOCK_M': 64, 'BLOCK_N': 32}, num_stages=3, num_warps=4),
#     triton.Config(kwargs={'BLOCK_M': 64, 'BLOCK_N': 32}, num_stages=4, num_warps=4),

#     triton.Config(kwargs={'BLOCK_M': 64, 'BLOCK_N': 64}, num_stages=1, num_warps=4),
#     triton.Config(kwargs={'BLOCK_M': 64, 'BLOCK_N': 64}, num_stages=2, num_warps=4),
#     triton.Config(kwargs={'BLOCK_M': 64, 'BLOCK_N': 64}, num_stages=3, num_warps=4),
#     triton.Config(kwargs={'BLOCK_M': 64, 'BLOCK_N': 64}, num_stages=4, num_warps=4),

#     triton.Config(kwargs={'BLOCK_M': 32, 'BLOCK_N': 32}, num_stages=1, num_warps=4),
#     triton.Config(kwargs={'BLOCK_M': 32, 'BLOCK_N': 32}, num_stages=2, num_warps=4),
#     triton.Config(kwargs={'BLOCK_M': 32, 'BLOCK_N': 32}, num_stages=3, num_warps=4),
#     triton.Config(kwargs={'BLOCK_M': 32, 'BLOCK_N': 32}, num_stages=4, num_warps=4),
#   ],
#   key=['DIM', "HEADDIM", "SEQLEN"] # the two above configs will be evaluated anytime
#                  # the value of x_size changes
# )
@triton.jit
def _vw_l1norm_kernel(
    V, W, O, scale,
    stride_v_bs, stride_v_head, stride_v_seqlen, stride_v_headdim,
    stride_w_head, stride_w_headdim, stride_w_dim,
    stride_o_bs, stride_o_head, stride_o_seqlen,
    BS, HEAD, SEQLEN,
    HEADDIM: tl.constexpr,
    DIM: tl.constexpr,
    BLOCK_M: tl.constexpr, # NOTE: value_states block
    BLOCK_N: tl.constexpr, # NOTE: w block
):
    '''
    value_states.shape = (bs, head_num, seqlen, head_dim)
    o_proj_w.shape = (head_num, head_dim, head_num * head_dim)
    output.shape = (bs, head_num, seqlen, head_dim)

    grid = (cdiv(seqlen, BLOCK_M), bs * head)
    o = (bs, head_num, seqlen, head_dim)
    '''
    # v block
    start_m = tl.program_id(0)
    off_bs_head = tl.program_id(1)

    input_dtype = V.dtype.element_ty

    v_base_offset = off_bs_head * stride_v_head
    V_block_ptr = tl.make_block_ptr(
        base=V + v_base_offset,
        shape=(SEQLEN, HEADDIM),
        strides=(stride_v_seqlen, stride_v_headdim),
        offsets=(start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, HEADDIM),
        order=(1, 0),
    )

    w_base_offset = (off_bs_head // BS) * stride_w_head
    W_block_ptr = tl.make_block_ptr(
        base=W + w_base_offset,
        shape=(HEADDIM, DIM),
        strides=(stride_w_headdim, stride_w_dim),
        offsets=(0, 0),
        block_shape=(HEADDIM, BLOCK_N),
        order=(1, 0),
    )

    # (block_m, headdim)
    v = tl.load(V_block_ptr)
    v = (v * scale).to(input_dtype)
    # w range
    lo, hi = 0, DIM
    norm = tl.zeros([BLOCK_M], dtype=tl.float32)
    for start_n in range(lo, hi, BLOCK_N):
        w = tl.load(W_block_ptr)
        # (block_m, headdim) @ (headdim, block_n) => (block_m, block_n)
        vw = tl.dot(v, w)
        norm += tl.sum(tl.abs(vw), axis=-1)

        W_block_ptr = tl.advance(W_block_ptr, (0, BLOCK_N))

    off_m = tl.arange(0, BLOCK_M)
    o_base_offset = off_bs_head * stride_o_head
    O_block_ptr = O + o_base_offset + (start_m * BLOCK_M + off_m) * stride_o_seqlen

    tl.store(O_block_ptr, norm.to(input_dtype), start_m * BLOCK_M + off_m < SEQLEN)


@triton.jit
def _vw_l2norm_kernel(
    V, W, O, scale,
    stride_v_bs, stride_v_head, stride_v_seqlen, stride_v_headdim,
    stride_w_head, stride_w_headdim, stride_w_dim,
    stride_o_bs, stride_o_head, stride_o_seqlen,
    BS, HEAD, SEQLEN,
    HEADDIM: tl.constexpr,
    DIM: tl.constexpr,
    BLOCK_M: tl.constexpr, # NOTE: value_states block
    BLOCK_N: tl.constexpr, # NOTE: w block
):
    '''
    value_states.shape = (bs, head_num, seqlen, head_dim)
    o_proj_w.shape = (head_num, head_dim, head_num * head_dim)
    output.shape = (bs, head_num, seqlen, head_dim)

    grid = (cdiv(seqlen, BLOCK_M), bs * head)
    o = (bs, head_num, seqlen, head_dim)
    '''
    # v block
    start_m = tl.program_id(0)
    off_bs_head = tl.program_id(1)

    input_dtype = V.dtype.element_ty

    v_base_offset = off_bs_head * stride_v_head
    V_block_ptr = tl.make_block_ptr(
        base=V + v_base_offset,
        shape=(SEQLEN, HEADDIM),
        strides=(stride_v_seqlen, stride_v_headdim),
        offsets=(start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, HEADDIM),
        order=(1, 0),
    )

    w_base_offset = (off_bs_head // BS) * stride_w_head
    W_block_ptr = tl.make_block_ptr(
        base=W + w_base_offset,
        shape=(HEADDIM, DIM),
        strides=(stride_w_headdim, stride_w_dim),
        offsets=(0, 0),
        block_shape=(HEADDIM, BLOCK_N),
        order=(1, 0),
    )

    # (block_m, headdim)
    v = tl.load(V_block_ptr)
    v = (v * scale).to(input_dtype)
    # w range
    lo, hi = 0, DIM
    norm = tl.zeros([BLOCK_M], dtype=tl.float32)
    for start_n in range(lo, hi, BLOCK_N):
        w = tl.load(W_block_ptr)
        # (block_m, headdim) @ (headdim, block_n) => (block_m, block_n)
        vw = tl.dot(v, w)
        # norm += tl.sum(tl.abs(vw), axis=-1)
        norm += tl.sum((vw) * (vw), axis=-1)  # L2

        W_block_ptr = tl.advance(W_block_ptr, (0, BLOCK_N))
    norm = tl.sqrt(norm)
    off_m = tl.arange(0, BLOCK_M)
    o_base_offset = off_bs_head * stride_o_head
    O_block_ptr = O + o_base_offset + (start_m * BLOCK_M + off_m) * stride_o_seqlen

    tl.store(O_block_ptr, norm.to(input_dtype), start_m * BLOCK_M + off_m < SEQLEN)


def vw_l1norm(value_states, o_proj_w, scale=1.0):
    '''
    fused kernel for norm(v * w)

    value_states.shape = (bs, head_num, seqlen, head_dim)
    o_proj_w.shape = (head_num, head_dim, head_num * head_dim)
    output.shape = (bs, head_num, seqlen, head_dim)
    '''

    bs, head_num, seq_len, head_dim = value_states.shape
    assert value_states.size(-1) == o_proj_w.size(-2)
    # assert head_num * head_dim == o_proj_w.size(-1)
    assert head_dim in {16, 32, 64, 128, 256}

    o = torch.zeros((bs, head_num, seq_len), device=value_states.device, dtype=value_states.dtype)

    BLOCK_M, BLOCK_N, num_stages, num_warps = get_config(seq_len, head_dim)
    grid = (triton.cdiv(seq_len, BLOCK_M), bs * head_num, 1)
    _vw_l1norm_kernel[grid](
        value_states, o_proj_w, o, scale,
        value_states.stride(0), value_states.stride(1), value_states.stride(2), value_states.stride(3),
        o_proj_w.stride(0), o_proj_w.stride(1), o_proj_w.stride(2),
        o.stride(0), o.stride(1), o.stride(2),
        bs, head_num, seq_len,
        HEADDIM=head_dim,
        # DIM=head_dim * head_num,
        DIM=o_proj_w.size(-1),
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        num_warps=num_warps,
        num_stages=num_stages)


    # # NOTE: autotune
    # grid = lambda META: (triton.cdiv(seq_len, META["BLOCK_M"]), bs * head_num, 1)
    # _vw_l1norm_kernel[grid](
    #     value_states, o_proj_w, o,
    #     value_states.stride(0), value_states.stride(1), value_states.stride(2), value_states.stride(3),
    #     o_proj_w.stride(0), o_proj_w.stride(1), o_proj_w.stride(2),
    #     o.stride(0), o.stride(1), o.stride(2),
    #     bs, head_num, seq_len,
    #     HEADDIM=head_dim,
    #     DIM=head_dim * head_num,
    #     # BLOCK_M=BLOCK_M,
    #     # BLOCK_N=BLOCK_N,
    #     # num_warps=num_warps,
    #     # num_stages=num_stages
    #     )

    return o



def vw_l2norm(value_states, o_proj_w, scale=1.0):
    '''
    fused kernel for norm(v * w)

    value_states.shape = (bs, head_num, seqlen, head_dim)
    o_proj_w.shape = (head_num, head_dim, head_num * head_dim)
    output.shape = (bs, head_num, seqlen, head_dim)
    '''

    bs, head_num, seq_len, head_dim = value_states.shape

    assert value_states.size(-1) == o_proj_w.size(-2)
    assert head_num * head_dim == o_proj_w.size(-1)
    assert head_dim in {16, 32, 64, 128, 256}

    o = torch.zeros((bs, head_num, seq_len), device=value_states.device, dtype=value_states.dtype)

    BLOCK_M, BLOCK_N, num_stages, num_warps = get_config(seq_len, head_dim)
    grid = (triton.cdiv(seq_len, BLOCK_M), bs * head_num, 1)
    _vw_l2norm_kernel[grid](
        value_states, o_proj_w, o, scale,
        value_states.stride(0), value_states.stride(1), value_states.stride(2), value_states.stride(3),
        o_proj_w.stride(0), o_proj_w.stride(1), o_proj_w.stride(2),
        o.stride(0), o.stride(1), o.stride(2),
        bs, head_num, seq_len,
        HEADDIM=head_dim,
        DIM=head_dim * head_num,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        num_warps=num_warps,
        num_stages=num_stages)

    return o

def l1norm(v, w):
    vw = v @ w
    vw_norm = torch.norm(vw, p=1, dim=-1)
    return vw_norm

def l2norm(v, w):
    vw = v @ w
    vw_norm = torch.norm(vw, p=2, dim=-1)
    return vw_norm

def abs_sum(v, w):
    '''
    v.shape = (bs, head_num, seq_len, head_dim)
    w.shape = (head_num, head_dim, head_num * head_dim)
    '''
    vw = v @ w
    return torch.abs(vw).sum(dim=-1)

def performance_l2_test():
    bs, head_num, seq_len, head_dim = 1, 8, 20481, 64

    v = torch.rand((bs, head_num, seq_len, head_dim), device="cuda", dtype=torch.bfloat16)
    v /= seq_len

    w = torch.rand((head_num, head_dim, head_num * head_dim), device="cuda", dtype=torch.bfloat16)

    # NOTE: warmup
    for i in range(10):
        expected_res = l2norm(v, w)

    torch.cuda.synchronize()
    t0 = time.time()
    for i in range(100):
        expected_res = l2norm(v, w)
    torch.cuda.synchronize()
    t0 = time.time() - t0

    # NOTE: warmup
    for i in range(10):
        ops_res = vw_l2norm(v, w)

    torch.cuda.synchronize()
    t1 = time.time()
    for i in range(100):
        ops_res = vw_l2norm(v, w)
    torch.cuda.synchronize()
    t1 = time.time() - t1

    print(expected_res)
    print("---------")
    print(ops_res)
    print(t0, t1)
    print(v.shape)
    print(w.shape)

    assert torch.allclose(expected_res, ops_res, atol=1e-2)


def performance_l1_test():
    bs, head_num, seq_len, head_dim = 1, 8, 20481, 64

    v = torch.rand((bs, head_num, seq_len, head_dim), device="cuda", dtype=torch.bfloat16)
    v /= seq_len

    w = torch.rand((head_num, head_dim, head_num * head_dim), device="cuda", dtype=torch.bfloat16)

    # NOTE: warmup
    for i in range(10):
        expected_res = l1norm(v, w)

    torch.cuda.synchronize()
    t0 = time.time()
    for i in range(100):
        expected_res = l1norm(v, w)
    torch.cuda.synchronize()
    t0 = time.time() - t0

    # NOTE: warmup
    for i in range(10):
        ops_res = vw_l1norm(v, w)

    torch.cuda.synchronize()
    t1 = time.time()
    for i in range(100):
        ops_res = vw_l1norm(v, w)
    torch.cuda.synchronize()
    t1 = time.time() - t1

    print(expected_res)
    print("---------")
    print(ops_res)
    print(t0, t1)
    print(v.shape)
    print(w.shape)

    assert torch.allclose(expected_res, ops_res, atol=1e-2)


if __name__ == "__main__":
    # set seed
    torch.manual_seed(0)
    performance_l1_test()