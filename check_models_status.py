"""
Model Status Inspector for AI Hindi Cinema Studio.
Checks disk presence and sizes of all AI models (Wan2.1, Flux.1, SDXL, MiniMax H3).
"""

import os

MODELS = [
    {
        "name": "Wan2.1 (Alibaba Video Diffusion)",
        "expected_gb": 3.2,
        "paths": [
            "models/wan2_1/diffusion_pytorch_model.safetensors",
            "models/wan2_1/Wan2.1_T2V_1.3B_bf16.safetensors",
            "ComfyUI/models/checkpoints/Wan2.1_T2V_1.3B_bf16.safetensors"
        ]
    },
    {
        "name": "SDXL / Flux.1 4K Photorealism",
        "expected_gb": 6.4,
        "paths": [
            "models/image_gen/sd_xl_turbo_1.0_fp16.safetensors",
            "models/image_gen/flux1-schnell.safetensors"
        ]
    },
    {
        "name": "MiniMax H3 (Hailuo) 15s Video Model",
        "expected_gb": 18.0,
        "paths": [
            "ComfyUI/models/checkpoints/H3-Base-Ref2VA.safetensors",
            "ComfyUI/models/checkpoints/minimax_h3_ref2va_pruned_int8_convrot.safetensors"
        ]
    }
]

def check_status():
    print("=================================================================")
    print("📊 AI Hindi Cinema Studio - Model Status Report")
    print("=================================================================")
    
    all_ready = True
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
            print(f"⏳ [NOT DOWNLOADED] {m['name']} (अपेक्षित: ~{m['expected_gb']} GB)")
            all_ready = False
            
    print("-----------------------------------------------------------------")
    if all_ready:
        print("🎉 सभी लोकल GPU मॉडल्स डिस्क पर मौजूद हैं और पूरी तरह तैयार हैं!")
    else:
        print("💡 मॉडल्स डाउनलोड करने के लिए चलाएं: ./download_fast.sh")
        print("💡 (नोट: यदि मॉडल डाउनलोड नहीं भी है, तो भी क्लाउड AI इंजन से वीडियो बनती रहेगी!)")
    print("=================================================================")

if __name__ == "__main__":
    check_status()
