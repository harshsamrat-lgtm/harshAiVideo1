"""
Model Status Inspector for AI Hindi Cinema Studio.
Checks presence and sizes of Wan2.1 (14B Flagship & 1.3B), SDXL, and MiniMax H3.
"""

import os

MODELS = [
    {
        "name": "Wan2.1-14B (Alibaba Flagship Master Video Model)",
        "expected_gb": 28.0,
        "paths": [
            "models/wan2_1/wan2.1_i2v_720p_14B_bf16.safetensors",
            "models/wan2_1/wan2.1_i2v_720p_14B_fp8.safetensors",
            "ComfyUI/models/checkpoints/wan2.1_i2v_720p_14B_bf16.safetensors"
        ]
    },
    {
        "name": "Wan2.1-1.3B (Alibaba Fast Video Model)",
        "expected_gb": 3.2,
        "paths": [
            "models/wan2_1/Wan2.1_T2V_1.3B_bf16.safetensors",
            "models/wan2_1/diffusion_pytorch_model.safetensors",
            "ComfyUI/models/checkpoints/Wan2.1_T2V_1.3B_bf16.safetensors"
        ]
    },
    {
        "name": "MiniMax H3 (Hailuo) 15s Video Model",
        "expected_gb": 18.5,
        "paths": [
            "ComfyUI/models/checkpoints/H3-Base-Ref2VA.safetensors",
            "ComfyUI/models/checkpoints/minimax_h3_ref2va_pruned_int8_convrot.safetensors"
        ]
    },
    {
        "name": "SDXL / Flux.1 4K Photorealism Model",
        "expected_gb": 6.4,
        "paths": [
            "models/image_gen/sd_xl_turbo_1.0_fp16.safetensors",
            "models/image_gen/flux1-schnell.safetensors"
        ]
    }
]

def check_status():
    print("=================================================================")
    print("📊 AI Hindi Cinema Studio - Model Status Report")
    print("=================================================================")
    
    for m in MODELS:
        found = False
        actual_size_gb = 0
        found_path = ""
        for p in m["paths"]:
            if os.path.exists(p):
                sz = os.path.getsize(p) / (1024**3)
                if sz > 0.1:
                    found = True
                    actual_size_gb = sz
                    found_path = p
                    break
        
        if found:
            print(f"✅ [DOWNLOADED] {m['name']}")
            print(f"   फ़ाइल: {found_path} ({actual_size_gb:.2f} GB)")
        else:
            print(f"⏳ [NOT ON DISK] {m['name']} (अनुमानित: ~{m['expected_gb']} GB)")
            
    print("-----------------------------------------------------------------")
    print("💡 मॉडल्स डाउनलोड करने के लिए चलाएं: ./download_fast.sh")
    print("💡 (नोट: जब तक मॉडल डाउनलोड हो रहा है, तब तक भी क्लाउड AI इंजन से 15s वीडियो बनती रहेगी!)")
    print("=================================================================")

if __name__ == "__main__":
    check_status()
