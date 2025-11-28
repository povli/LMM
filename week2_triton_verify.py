# week2_triton_verify.py
import torch
import torch.nn.functional as F
from fla.ops.gated_delta_rule import chunk_gated_delta_rule

def naive_titans_recurrence(q, k, v, beta, g):
    """
    用最简单的 Python 循环手写 Titans (Delta Rule) 公式。
    公式: S_t = g_t * S_{t-1} + beta_t * (v_t - S_{t-1} @ k_t) @ k_t.T
    """
    batch_size, seq_len, num_heads, head_dim = q.shape
    d_head = head_dim
    
    # 初始化状态 S (Batch, Heads, Dim, Dim)
    state = torch.zeros(batch_size, num_heads, d_head, d_head, device=q.device, dtype=torch.float32)
    outputs = []

    for t in range(seq_len):
        k_t = k[:, t].float() # (B, H, D)
        v_t = v[:, t].float()
        beta_t = beta[:, t].float()
        g_t = g[:, t].float() if g is not None else 1.0
        
        # --- Titans 的灵魂：误差计算 ---
        # 1. 预测/重构 (Recall): 看看当前记忆 S 能不能预测出 v
        # S: (B, H, D, D), k_t: (B, H, D, 1) -> v_pred: (B, H, D, 1)
        v_pred = torch.einsum('bhmn, bhnk -> bhmk', state, k_t[..., None]).squeeze(-1)
        
        # 2. 计算惊奇度 (Surprise/Error): 实际值 - 预测值
        error = v_t - v_pred
        
        # 3. 更新记忆 (Update): 用误差去修正记忆
        # delta = beta * error * k^T
        delta = torch.einsum('bhm, bhn -> bhmn', error * beta_t[..., None], k_t)
        
        # 应用遗忘门 g_t 并更新
        if g is not None:
            state = state * g_t[..., None, None]
        state = state + delta
        
        # 4. 计算当前步的输出 (Output): q * S
        o_t = torch.einsum('bhmn, bhnk -> bhmk', state, q[:, t].float()[..., None]).squeeze(-1)
        outputs.append(o_t)

    return torch.stack(outputs, dim=1)

def run_week2_verification():
    print("🔬 开始 Week 2 验证：Titans (Gated DeltaNet) 算子深度探究")
    device = "cuda"
    dtype = torch.bfloat16
    
    # --------------------------
    # 1. 数学一致性验证
    # --------------------------
    print("\n[实验 1] 数学公式对齐测试 (Triton vs Naive Python)")
    B, L, H, D = 2, 64, 4, 32
    torch.manual_seed(42)
    
    # 随机生成数据
    q = torch.randn(B, L, H, D, device=device, dtype=dtype)
    k = torch.randn(B, L, H, D, device=device, dtype=dtype)
    v = torch.randn(B, L, H, D, device=device, dtype=dtype)
    beta = torch.rand(B, L, H, device=device, dtype=dtype)
    g = torch.rand(B, L, H, device=device, dtype=dtype) # 遗忘门

    # A. 运行 Triton 算子 (MoM 项目里用的)
    # 注意：我们这里关闭 l2norm 以便和简单公式对齐，且防止之前遇到的 OOM
    o_triton, _ = chunk_gated_delta_rule(q, k, v, g, beta, 
                                         use_qk_l2norm_in_kernel=False, 
                                         output_final_state=False)

    # B. 运行手写 Naive Titans
    o_naive = naive_titans_recurrence(q, k, v, beta, g).to(dtype)

    # C. 比较差异
    diff = (o_triton - o_naive).abs().max().item()
    print(f"   >>> 最大误差: {diff:.6f}")
    
    if diff < 1e-2:
        print("   ✅ 验证通过：底层 Triton 算子完美执行了 Titans 的误差更新公式。")
    else:
        print("   ❌ 验证失败：算子行为与公式不一致，需检查。")

    # --------------------------
    # 2. 惊奇度机制验证 (The "Surprise" Test)
    # --------------------------
    print("\n[实验 2] 惊奇度机制验证 (The Surprise Test)")
    print("   目标：验证当输入重复信息时，Titans 是否会自动停止更新（因为误差为0）。")
    
    # 构造一个特殊的序列：第0步和第1步输入完全一样的 k, v
    # 假设 k 是归一化的，beta=1 (全量更新)
    L_toy = 2
    k_toy = torch.randn(1, L_toy, 1, 16, device=device, dtype=dtype)
    k_toy = F.normalize(k_toy, dim=-1) # 归一化很重要
    k_toy[:, 1] = k_toy[:, 0]          # 第二步完全重复第一步
    
    v_toy = torch.randn(1, L_toy, 1, 16, device=device, dtype=dtype)
    v_toy[:, 1] = v_toy[:, 0]          # Value 也重复
    
    beta_toy = torch.ones(1, L_toy, 1, device=device, dtype=dtype) # 学习率=1
    g_toy = torch.ones(1, L_toy, 1, device=device, dtype=dtype)    # 不遗忘
    q_toy = torch.randn(1, L_toy, 1, 16, device=device, dtype=dtype)

    # 运行手写 Titans 引擎来观察内部 State 变化
    # 我们稍微改一下 naive 函数来返回状态增量
    print("   ... 正在模拟输入序列: [Token A, Token A]")
    
    # --- Step 0 ---
    state = torch.zeros(1, 1, 16, 16, device=device, dtype=torch.float32)
    k0, v0 = k_toy[:, 0].float(), v_toy[:, 0].float()
    
    # 预测
    pred0 = state @ k0[..., None]
    # 误差
    err0 = v0[..., None] - pred0
    # 更新量
    delta0 = err0 @ k0[..., None].transpose(-1, -2)
    state = state + delta0
    print(f"   [Step 0] 初始状态为空，误差范数: {err0.norm().item():.4f} -> 产生更新量: {delta0.norm().item():.4f}")

    # --- Step 1 (重复输入) ---
    k1, v1 = k_toy[:, 1].float(), v_toy[:, 1].float() # k1 == k0, v1 == v0
    
    # 预测 (此时 State 已经记住了 k0->v0)
    pred1 = state @ k1[..., None]
    # 误差 (理论上应该接近0)
    err1 = v1[..., None] - pred1
    # 更新量
    delta1 = err1 @ k1[..., None].transpose(-1, -2)
    
    print(f"   [Step 1] 输入重复数据，误差范数: {err1.norm().item():.4f} -> 产生更新量: {delta1.norm().item():.4f}")
    
    # 验证逻辑
    ratio = delta1.norm() / (delta0.norm() + 1e-6)
    print(f"   >>> 更新量缩减比率 (Step1 / Step0): {ratio.item():.4%}")
    
    if ratio < 0.1:
        print("   ✅ 验证通过：模型表现出强烈的“测试时训练”特性！")
        print("      因为已经记住了该信息，第二次遇到时惊奇度为0，几乎不消耗记忆容量。")
    else:
        print("   ❌ 验证失败：模型仍在重复记忆，这退化成了普通的 Linear Attention。")

if __name__ == "__main__":
    run_week2_verification()