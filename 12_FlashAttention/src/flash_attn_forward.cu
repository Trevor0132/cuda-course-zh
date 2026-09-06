#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cmath>
#include "flash_attn.h"


/**
 * FlashAttention-2 前向核函数 (极致性能最佳实践版)
 * 
 * 性能优化组合：
 * 1. 快速位移寻址 (Power-of-2 Indexing)：
 *    NUM_VEC 为 2 的整数幂 (16)，编译器可将二维寻址完全优化为快速按位移位 (r << 4) 与掩码 (c & 15)，零整数除法开销。
 * 2. 广播式共享内存读取 (SRAM Broadcast)：
 *    外层循环遍历 c，所有线程并发读取同一行 s_V[c][k]，天然触发片上广播，零 Bank 冲突。
 * 3. 0 权重动态剪枝 (Zero-Weight Pruning)：
 *    对于 Softmax 概率为 0（因果掩码遮蔽或下溢）的元素，直接跳过整个 V 向量乘加，大幅节省算力。
 * 4. Q 向量预缩放 (Q Pre-scaling)：
 *    在 Q 载入寄存器时一次性乘入 scale = 1/sqrt(d)，彻底消除后续内层循环的所有浮点缩放乘法。
 * 5. 硬件级快速指数指令 (__expf)：
 *    直接发射 GPU 特殊功能单元 (SFU) 硬件原生的 ex2.approx 指令。
 * 6. float4 128-bit 向量化访存：
 *    全局内存与共享内存之间单指令吞吐 128 位数据。
 */
template <int HEAD_DIM, int Br, int Bc>
__global__ void flash_attn_fwd_kernel(
    const float* __restrict__ Q,
    const float* __restrict__ K,
    const float* __restrict__ V,
    float* __restrict__ O,
    int B,
    int H,
    int N,
    int d,
    float scale,
    bool is_causal
) {
    constexpr int VEC_SIZE = 4;
    constexpr int NUM_VEC = HEAD_DIM / VEC_SIZE;

    const int b = blockIdx.z;
    const int h = blockIdx.y;
    const int row_block_idx = blockIdx.x;
    const int tid = threadIdx.x;
    const int row_idx = row_block_idx * Br + tid;

    const size_t head_stride = static_cast<size_t>(N) * d;
    const size_t head_offset = (static_cast<size_t>(b) * H + h) * head_stride;

    const float* Q_head = Q + head_offset;
    const float* K_head = K + head_offset;
    const float* V_head = V + head_offset;
    float* O_head = O + head_offset;

    union SharedStorage {
        float4 s_stage[Br][NUM_VEC];
        struct {
            float4 s_K[Bc][NUM_VEC];
            float4 s_V[Bc][NUM_VEC];
        } kv;
    };
    __shared__ SharedStorage smem;

    // 1. 快速移位寻址协同加载 Q 块到共享内存临时暂存区
    const float4* Q_head4 = reinterpret_cast<const float4*>(Q_head);
    for (int i = tid; i < Br * NUM_VEC; i += blockDim.x) {
        int r = i / NUM_VEC;
        int c = i % NUM_VEC;
        int global_r = row_block_idx * Br + r;
        smem.s_stage[r][c] = (global_r < N) ? Q_head4[global_r * NUM_VEC + c] : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    }
    __syncthreads();

    // 2. 线程独占寄存器，并对 Q 进行【预先缩放】
    float4 q_reg[NUM_VEC];
    float4 o_reg[NUM_VEC];

    #pragma unroll
    for (int k = 0; k < NUM_VEC; ++k) {
        float4 q_val = (row_idx < N) ? smem.s_stage[tid][k] : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        q_reg[k] = make_float4(q_val.x * scale, q_val.y * scale, q_val.z * scale, q_val.w * scale);
        o_reg[k] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    }
    // 确保所有线程读取完 smem.s_stage 后，再复用共享内存加载 K 和 V
    __syncthreads();

    float m_i = -INFINITY;
    float l_i = 0.0f;

    const int num_kv_blocks = (N + Bc - 1) / Bc;
    const float4* K_head4 = reinterpret_cast<const float4*>(K_head);
    const float4* V_head4 = reinterpret_cast<const float4*>(V_head);

    // 3. 内层循环：逐块遍历 K 与 V (复用共享内存)
    for (int j = 0; j < num_kv_blocks; ++j) {
        if (is_causal && (j * Bc > (row_block_idx + 1) * Br - 1)) {
            break;
        }

        // 向量化协同加载 K 与 V 到复用共享内存
        for (int i = tid; i < Bc * NUM_VEC; i += blockDim.x) {
            int r = i / NUM_VEC;
            int c = i % NUM_VEC;
            int global_r = j * Bc + r;
            smem.kv.s_K[r][c] = (global_r < N) ? K_head4[global_r * NUM_VEC + c] : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            smem.kv.s_V[r][c] = (global_r < N) ? V_head4[global_r * NUM_VEC + c] : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        }
        __syncthreads();

        if (row_idx < N) {
            float scores[Bc];
            float m_tile = -INFINITY;

            // 点积 Q * K^T: 采用双路累加器 (dot_a, dot_b) 消除 FMA 指令级依赖延迟
            #pragma unroll
            for (int c = 0; c < Bc; ++c) {
                const int col_idx = j * Bc + c;
                if (col_idx < N && (!is_causal || col_idx <= row_idx)) {
                    float dot_a = 0.0f;
                    float dot_b = 0.0f;
                    #pragma unroll
                    for (int k = 0; k < NUM_VEC; ++k) {
                        float4 q4 = q_reg[k];
                        float4 k4 = smem.kv.s_K[c][k];
                        dot_a += q4.x * k4.x + q4.z * k4.z;
                        dot_b += q4.y * k4.y + q4.w * k4.w;
                    }
                    float dot = dot_a + dot_b;
                    scores[c] = dot;
                    m_tile = fmaxf(m_tile, dot);
                } else {
                    scores[c] = -INFINITY;
                }
            }

            // 在线 Softmax 统计量动态更新 + 就地复用 scores 寄存器空间
            const float m_new = fmaxf(m_i, m_tile);
            const float alpha = (m_i == -INFINITY) ? 0.0f : __expf(m_i - m_new);

            float p_sum = 0.0f;
            #pragma unroll
            for (int c = 0; c < Bc; ++c) {
                if (scores[c] == -INFINITY) {
                    scores[c] = 0.0f;
                } else {
                    float p_val = __expf(scores[c] - m_new);
                    scores[c] = p_val;
                    p_sum += p_val;
                }
            }

            l_i = l_i * alpha + p_sum;

            // 当局部最大值未更新时 (alpha == 1.0f)，直接跳过现有累加向量缩放，省去全量乘法
            if (alpha != 1.0f) {
                #pragma unroll
                for (int k = 0; k < NUM_VEC; ++k) {
                    o_reg[k].x *= alpha;
                    o_reg[k].y *= alpha;
                    o_reg[k].z *= alpha;
                    o_reg[k].w *= alpha;
                }
            }

            // 广播式加权乘加 + 0 权重分支剪枝
            #pragma unroll
            for (int c = 0; c < Bc; ++c) {
                const float pc = scores[c];
                if (pc > 0.0f) {
                    #pragma unroll
                    for (int k = 0; k < NUM_VEC; ++k) {
                        float4 v4 = smem.kv.s_V[c][k];
                        o_reg[k].x += pc * v4.x;
                        o_reg[k].y += pc * v4.y;
                        o_reg[k].z += pc * v4.z;
                        o_reg[k].w += pc * v4.w;
                    }
                }
            }

            m_i = m_new;
        }

        __syncthreads();
    }

    // 4. 最终归一化与写回：复用共享内存暂存区进行连续向量化写回
    if (row_idx < N) {
        const float inv_l = (l_i > 0.0f) ? (1.0f / l_i) : 0.0f;
        #pragma unroll
        for (int k = 0; k < NUM_VEC; ++k) {
            smem.s_stage[tid][k] = make_float4(
                o_reg[k].x * inv_l,
                o_reg[k].y * inv_l,
                o_reg[k].z * inv_l,
                o_reg[k].w * inv_l
            );
        }
    }
    __syncthreads();

    float4* O_head4 = reinterpret_cast<float4*>(O_head);
    for (int i = tid; i < Br * NUM_VEC; i += blockDim.x) {
        int r = i / NUM_VEC;
        int c = i % NUM_VEC;
        int global_r = row_block_idx * Br + r;
        if (global_r < N) {
            O_head4[global_r * NUM_VEC + c] = smem.s_stage[r][c];
        }
    }
}

// C++ 包装分发函数
torch::Tensor flash_attn_forward(
    torch::Tensor Q,
    torch::Tensor K,
    torch::Tensor V,
    bool is_causal
) {
    TORCH_CHECK(Q.is_cuda(), "Q must be a CUDA tensor");
    TORCH_CHECK(K.is_cuda(), "K must be a CUDA tensor");
    TORCH_CHECK(V.is_cuda(), "V must be a CUDA tensor");
    TORCH_CHECK(Q.is_contiguous(), "Q must be contiguous");
    TORCH_CHECK(K.is_contiguous(), "K must be contiguous");
    TORCH_CHECK(V.is_contiguous(), "V must be contiguous");
    TORCH_CHECK(Q.dim() == 4, "Q must be 4D: (Batch, Heads, SeqLen, HeadDim)");

    const int B = Q.size(0);
    const int H = Q.size(1);
    const int N = Q.size(2);
    const int d = Q.size(3);

    TORCH_CHECK(d == K.size(3) && d == V.size(3), "Head dimension d must match across Q, K, V");
    TORCH_CHECK(N == K.size(2) && N == V.size(2), "Sequence length N must match across Q, K, V");

    auto O = torch::empty_like(Q);
    const float scale = 1.0f / std::sqrt(static_cast<float>(d));

    if (d == 32) {
        constexpr int BR = 128;
        constexpr int BC = 32;
        dim3 block(BR);
        dim3 grid((N + BR - 1) / BR, H, B);
        flash_attn_fwd_kernel<32, BR, BC><<<grid, block>>>(
            Q.data_ptr<float>(), K.data_ptr<float>(), V.data_ptr<float>(), O.data_ptr<float>(),
            B, H, N, d, scale, is_causal
        );
    } else if (d == 64) {
        constexpr int BR = 128;
        constexpr int BC = 32;
        dim3 block(BR);
        dim3 grid((N + BR - 1) / BR, H, B);
        flash_attn_fwd_kernel<64, BR, BC><<<grid, block>>>(
            Q.data_ptr<float>(), K.data_ptr<float>(), V.data_ptr<float>(), O.data_ptr<float>(),
            B, H, N, d, scale, is_causal
        );
    } else if (d == 128) {
        constexpr int BR = 64;
        constexpr int BC = 32;
        dim3 block(BR);
        dim3 grid((N + BR - 1) / BR, H, B);
        flash_attn_fwd_kernel<128, BR, BC><<<grid, block>>>(
            Q.data_ptr<float>(), K.data_ptr<float>(), V.data_ptr<float>(), O.data_ptr<float>(),
            B, H, N, d, scale, is_causal
        );
    } else {
        TORCH_CHECK(false, "Unsupported head dimension: ", d, ". Supported dims: 32, 64, 128");
    }

    return O;
}
