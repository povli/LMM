# week1_overfit_test.py
import torch
import time
from transformers import AutoConfig
from mom.models.mom.modeling_mom import MomForCausalLM
from mom.models.mom.configuration_mom import MomConfig

def run_overfit_test():
    print("🚀 开始 Week 1 过拟合测试：Mamba-2 Shared Backbone ...")

    # 1. 配置模型参数
    # 使用较小的模型配置以加快测试速度
    config = MomConfig(
        vocab_size=1000,      # 小词表
        hidden_size=1024,     # 对应 Mamba2 d_model
        num_hidden_layers=2,  # 只需要2层就能验证
        num_attention_heads=16, # 1024 / 16 = 64 (head_dim)
        max_position_embeddings=4096,
        
        # 关键 MoM 参数
        num_memories=4,
        topk=2,
        mom_backend="gated_deltanet", # 路由部分保持 DeltaNet (Titans)
        shared_mem=True,      # 必须开启，以测试 Mamba2
        
        # 其他优化
        use_cache=False,
        fuse_cross_entropy=False 
    )

    print("🔧 初始化模型 (Titans-MoM)...")
    try:
        model = MomForCausalLM(config).cuda().bfloat16() # 使用 A800 推荐 bf16
        print("✅ 模型初始化成功！")
    except Exception as e:
        print(f"❌ 模型初始化失败: {e}")
        return

    # 2. 构造虚假数据 (Batch=2, SeqLen=2048)
    print("🎲 构造 2k 长度的随机数据...")
    seq_len = 2048
    batch_size = 2
    
    # 随机生成 input_ids
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len)).cuda()
    # Labels 就是 input_ids 向后移一位（自回归任务）
    labels = input_ids.clone()

    # 3. 训练循环
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3) # 较大的 LR 加速过拟合
    
    print("🏃 开始训练循环 (Target: Loss -> 0)...")
    start_time = time.time()
    
    for step in range(50): # 跑 50 步通常足够过拟合
        optimizer.zero_grad()
        
        outputs = model(input_ids=input_ids, labels=labels)
        loss = outputs.loss
        
        loss.backward()
        optimizer.step()
        
        if step % 5 == 0:
            print(f"Step {step:02d} | Loss: {loss.item():.6f}")
            
        if loss.item() < 0.01:
            print(f"\n🎉 成功！Loss 已降至 {loss.item():.6f}")
            break
            
    total_time = time.time() - start_time
    print(f"\n⏱️ 测试耗时: {total_time:.2f}s")
    
    if loss.item() > 0.1:
        print("⚠️ 警告：Loss 下降缓慢，请检查梯度或学习率设置。")
    else:
        print("✅ Week 1 任务完成：Mamba-2 Shared Memory 集成正常，梯度回传通畅。")

if __name__ == "__main__":
    run_overfit_test()