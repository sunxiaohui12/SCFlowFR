## tiny推理代码 - 基于新配置文件！！！
import torch
import yaml
import os
from PIL import Image
from flowsr.flow import ShortcutFlowModel
from flowsr.models.unet.model import EfficientShortcutUnet
from flowsr.models.tiny_autoencoder import TAESD
from flowsr.models.swinir import SwinIR
from flowsr.ema import EMA
import torchvision.transforms as transforms

def load_model_from_config(config_path, checkpoint_path, device):
    """加载 ShortcutFlowModel 和 EMA 模型"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # 1) 构建网络
    fm_cfg = config['model']['params']['fm_cfg']
    net_cfg = fm_cfg['params']['net_cfg']['params']
    net = EfficientShortcutUnet(**net_cfg)

    # 2) 构建 ShortcutFlowModel，保留所有必要参数
    exclude_keys = {'net_cfg'}
    model_params = {k: v for k, v in fm_cfg['params'].items() if k not in exclude_keys}
    model = ShortcutFlowModel(net_cfg=net, **model_params)

    # 3) 加载 checkpoint
    ckpt = torch.load(checkpoint_path, map_location=device)
    state_dict = ckpt.get('state_dict', ckpt)

    model_state_dict = model.state_dict()

    def strip_prefixes(k: str) -> str:
        """去除常见的前缀"""
        known_prefixes = [
            'model.model.',
            'model.',
        ]
        new_k = k
        if new_k.startswith('module.'):
            new_k = new_k[len('module.'):]
        for p in known_prefixes:
            if new_k.startswith(p):
                new_k = new_k[len(p):]
                break
        return new_k

    filtered_state_dict = {}
    for k, v in state_dict.items():
        new_k = strip_prefixes(k)
        if new_k in model_state_dict and model_state_dict[new_k].shape == v.shape:
            filtered_state_dict[new_k] = v

    # 报告覆盖率
    num_total = len(model_state_dict)
    num_loaded = len(filtered_state_dict)
    cov = 100.0 * num_loaded / max(1, num_total)
    print(f"[FlowSR] Loaded {num_loaded}/{num_total} params ({cov:.1f}%) from checkpoint")

    # 加载权重
    model.load_state_dict(filtered_state_dict, strict=False)
    missing_keys = [k for k in model_state_dict.keys() if k not in filtered_state_dict]
    if len(missing_keys) > 0:
        print(f"[FlowSR] Missing keys: {len(missing_keys)} (showing up to 20)")
        print(missing_keys[:20])

    # 4) 初始化 EMA 包装器
    ema_model = EMA(
        model,
        beta=config['model']['params']['ema_rate'],
        update_after_step=config['model']['params']['ema_update_after_step'],
        update_every=config['model']['params']['ema_update_every'],
        include_online_model=False,
    )
    
    # 加载 EMA 权重
    if 'ema_model' in ckpt:
        try:
            ema_model.ema_model.load_state_dict(ckpt['ema_model'], strict=False)
            print("[FlowSR] EMA weights loaded from checkpoint")
        except Exception as e:
            print(f"[FlowSR] Failed to load EMA weights: {e}")
    
    ema_model.eval()
    return model.to(device), ema_model.ema_model.to(device)

def load_vae(config_path, device):
    """加载 Tiny VAE 模型"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    vae_cfg = config['model']['params']['first_stage_cfg']
    
    # 获取编码器和解码器路径
    encoder_path = vae_cfg['params']['encoder_path']
    decoder_path = vae_cfg['params']['decoder_path']
    
    # 使用项目根目录作为基准路径
    project_root = '/root/autodl-tmp/'
    encoder_path = os.path.join(project_root, encoder_path)
    decoder_path = os.path.join(project_root, decoder_path)
    
    if not os.path.exists(encoder_path):
        raise FileNotFoundError(f"Tiny VAE encoder not found at: {encoder_path}")
    if not os.path.exists(decoder_path):
        raise FileNotFoundError(f"Tiny VAE decoder not found at: {decoder_path}")
    
    # 创建 TAESD 模型
    vae = TAESD(encoder_path, decoder_path)
    
    print(f"[Tiny VAE] Loaded encoder from: {encoder_path}")
    print(f"[Tiny VAE] Loaded decoder from: {decoder_path}")
    
    vae.eval()
    return vae.to(device)

def load_swinir(config_path, checkpoint_path, device):
    """加载 SwinIR 模型，允许指定新的 checkpoint 路径"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    swinir_cfg = config['model']['params'].get('swinir_cfg')
    if swinir_cfg is None:
        raise KeyError("swinir_cfg not found in config['model']['params']")
    
    swinir = SwinIR(**swinir_cfg['params'])
    
    # 使用项目根目录作为基准路径
    project_root = '/root/autodl-tmp/'
    if not checkpoint_path:
        swinir_ckpt_path = config['model']['params'].get('swinir_path',"not-found")
    else:
        swinir_ckpt_path = os.path.join(project_root, checkpoint_path)
    if not os.path.exists(swinir_ckpt_path):
        raise FileNotFoundError(f"SwinIR checkpoint not found at: {swinir_ckpt_path}")
    
    ckpt = torch.load(swinir_ckpt_path, map_location=device)
    sd = ckpt.get('state_dict', ckpt)

    # 去除前缀并加载匹配的形状
    def strip_prefix(k: str) -> str:
        if k.startswith('module.'):
            k = k[len('module.'):]
        for p in ['swinir.', 'model.', 'network.']:
            if k.startswith(p):
                return k[len(p):]
        return k

    model_sd = swinir.state_dict()
    filtered = {}
    for k, v in sd.items():
        nk = strip_prefix(k)
        if nk in model_sd and model_sd[nk].shape == v.shape:
            filtered[nk] = v
    
    total = len(model_sd)
    loaded = len(filtered)
    print(f"[SwinIR] Loaded {loaded}/{total} params ({(100*loaded/max(1,total)):.1f}%) from {swinir_ckpt_path}")
    
    swinir.load_state_dict(filtered, strict=False)
    swinir.eval()
    return swinir.to(device)

def infer_sr_batch(model, ema_model, vae, swinir, lr_dataset_dir, output_dir, config_path, device='cuda', batch_size=4):
    """执行超分辨率推理"""
    if not torch.cuda.is_available() and device == 'cuda':
        print("CUDA is not available. Switching to CPU.")
        device = 'cpu'
    print(f"Using device: {device}")

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # 获取验证步数配置
    validation_timesteps = config['model']['params'].get('validation_timesteps', [1, 2, 4])
    segment_K = config['model']['params']['fm_cfg']['params']['segment_K']
    
    print(f"Validation timesteps: {validation_timesteps}")
    print(f"Segment K: {segment_K}")
    
    image_files = [f for f in os.listdir(lr_dataset_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    num_images = len(image_files)
    print(f"Processing {num_images} images from {lr_dataset_dir}")
    
    for i in range(0, num_images, batch_size):
        batch_files = image_files[i:i+batch_size]
        batch_lr_tensors = []
        
        for file in batch_files:
            lr_image_path = os.path.join(lr_dataset_dir, file)
            lr_image = Image.open(lr_image_path).convert('RGB')
            lr_tensor = transforms.ToTensor()(lr_image).unsqueeze(0).to(device)
            batch_lr_tensors.append(lr_tensor)
        
        lr_batch = torch.cat(batch_lr_tensors, dim=0)
        #print(f"LR batch shape: {lr_batch.shape}")
        
        with torch.no_grad():
            # SwinIR 预处理：输入 [0,1]，输出映射到 [-1,1]
            clean_lr_batch = swinir(lr_batch) * 2 - 1
           # print(f"Clean LR batch shape: {clean_lr_batch.shape}")
                
            # Tiny VAE 编码 - 注意：Tiny VAE 不需要 scale_factor
            lr_latent_batch = vae.encode(clean_lr_batch)
            #print(f"LR latent batch shape: {lr_latent_batch.shape}")
            
            # ShortcutFlow 输入构造：与训练时完全一致
            # 训练时：x_source=4通道（添加噪声后的低分辨率潜在表示）
            # 训练时：context=4通道（原始低分辨率潜在表示）
            # 模型内部拼接：4+4=8通道，符合in_channels: 8
            lr_channels = lr_latent_batch.shape[1]
            target_channels = 4  # 与训练时一致，使用4通道输入
            
            if lr_channels == 4:
                # 如果 VAE 输出4通道，直接使用作为x_source
                init_noise_batch = lr_latent_batch
            elif lr_channels > target_channels:
                # 如果 VAE 输出通道数过多，截取前4通道
                init_noise_batch = lr_latent_batch[:, :target_channels, :, :]
            else:
                # 如果 VAE 输出通道数不足，用零填充
                padding_channels = target_channels - lr_channels
                padding = torch.zeros(lr_latent_batch.shape[0], padding_channels, lr_latent_batch.shape[2], lr_latent_batch.shape[3], device=lr_latent_batch.device)
                init_noise_batch = torch.cat([lr_latent_batch, padding], dim=1)
            
            # 关键修复：添加训练时的噪声步骤
            # 训练时：noising_step: 400，对输入添加400步噪声
            # 这是FlowSR工作的关键！没有噪声，模型无法学习超分映射
            from flowsr.diffusion import ForwardDiffusion
            diffusion = ForwardDiffusion()
            noising_step = 400  # 与训练时一致
            init_noise_batch = diffusion.q_sample(x_start=init_noise_batch, t=noising_step)
            
            # print(f"VAE output channels: {lr_channels}")
            # print(f"Final input channels: {init_noise_batch.shape[1]}")
            # print(f"Init noise batch shape: {init_noise_batch.shape}")
            # print(f"Added noise with step {noising_step}")
            
            # 条件信息 - 与训练时完全一致
            # 训练时：context = lres_z_hr（4通道）
            # 模型内部会拼接：x_source(4通道) + context(4通道) = 8通道
            context_batch = lr_latent_batch  # 4通道，与训练时一致
            
            #print(f"Context batch shape: {context_batch.shape}")
            
            # 对每个验证步数进行采样
            for num_steps in validation_timesteps:
                # 验证步数必须为2的幂且不超过 segment_K
                if num_steps > segment_K:
                    print(f"Warning: num_steps {num_steps} > segment_K {segment_K}, skipping...")
                    continue
                
                if num_steps & (num_steps - 1) != 0:
                    print(f"Warning: num_steps {num_steps} is not a power of 2, skipping...")
                    continue
                
                print(f"Sampling with {num_steps} steps...")
                
                sample_kwargs = {
                    'num_steps': num_steps,
                    'method': 'euler',  # ShortcutFlow 只支持 euler
                    'use_sde': False,
                    'cfg_scale': 1.0,
                    'progress': True,
                    # 移除cond_key，因为我们直接传递context参数
                }

                # 生成高分辨率潜在表示
                #print(f"Before FlowSR - Input shape: {init_noise_batch.shape}")
                #print(f"Before FlowSR - Input range: [{init_noise_batch.min():.4f}, {init_noise_batch.max():.4f}]")
                
                hr_latent_batch = ema_model.generate(
                    x=init_noise_batch,
                    sample_kwargs=sample_kwargs,
                    context=context_batch,  # 直接传递context，而不是通过cond_key
                    return_intermediates=False
                )
                
                #print(f"After FlowSR - Output shape: {hr_latent_batch.shape}")
                #print(f"After FlowSR - Output range: [{hr_latent_batch.min():.4f}, {hr_latent_batch.max():.4f}]")
                
                # 检查输入输出是否相同
                if torch.allclose(init_noise_batch, hr_latent_batch, atol=1e-6):
                    print("WARNING: FlowSR output is identical to input! Model may not be working.")
                else:
                    diff = torch.abs(init_noise_batch - hr_latent_batch).mean()
                    #print(f"FlowSR is working! Mean difference: {diff:.6f}")
                
                #print(f"HR latent batch shape (s{num_steps}): {hr_latent_batch.shape}")

                # Tiny VAE 解码 - 注意：Tiny VAE 不需要 scale_factor
                hr_tensor_batch = vae.decode(hr_latent_batch)
                hr_tensor_batch = (hr_tensor_batch.clamp(-1, 1) + 1) / 2
                #print(f"HR tensor batch shape (s{num_steps}): {hr_tensor_batch.shape}")

                # 保存到步数特定的子文件夹
                step_dir = os.path.join(output_dir, f"s{num_steps}")
                os.makedirs(step_dir, exist_ok=True)
                
                for j, hr_tensor in enumerate(hr_tensor_batch):
                    output_path = os.path.join(step_dir, f"{os.path.splitext(batch_files[j])[0]}.png")
                    hr_image = transforms.ToPILImage()(hr_tensor.cpu())
                    hr_image.save(output_path)
                    print(f"Saved {output_path}")

if __name__ == "__main__":
    # 使用 Tiny FlowSR 的配置和权重
    config_path = 'logs/shortcutfm_face_tiny/exp_2025-08-29-23-29-42/config.yaml'
    checkpoint_path = 'logs/shortcutfm_face_tiny/exp_2025-08-29-23-29-42/checkpoints/step150000.ckpt'
    swinir_checkpoint_path = 'pretrained-models/DifFace/swinir_restoration512_L1.pth'
    # swinir_checkpoint_path = None
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 加载模型
    print("Loading models...")
    model, ema_model = load_model_from_config(config_path, checkpoint_path, device)
    vae = load_vae(config_path, device)
    swinir = load_swinir(config_path, swinir_checkpoint_path, device)
    
    # 设置输入输出路径
    lr_dataset_dir = 'load/wild/lfw'
    output_dir = 'logs/_results/lfw/flowsr_tiny_difFace_l1'
    os.makedirs(output_dir, exist_ok=True)
    
    # 执行推理
    print("Starting inference...")
    infer_sr_batch(model, ema_model, vae, swinir, lr_dataset_dir, output_dir, config_path, batch_size=4)
    print("Inference completed!")