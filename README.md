# 🎬 AI Hindi Cinema Studio v3.0

> **दुनिया का सबसे उत्तम AI-संचालित हिंदी स्टोरी-टू-मूवी जनरेटर प्लेटफॉर्म**  
> संचालित: **Flux.1 (4K Art & FaceID) + Wan2.1 / MiniMax H3 / SVD-XT (15s AI Video) + Neural Hindi Voices + Multi-Track FFmpeg**

---

## 🚀 नए रेंटेड GPU सर्वर (RunPod / Vast.ai / Lambda / Hetzner / Ubuntu) पर 1-क्लिक सेटअप

### 📋 सर्वर आवश्यकताएं (Server Requirements)
- **OS**: Ubuntu 22.04 LTS / Debian
- **GPU**: NVIDIA RTX 3090, RTX 4090, A100, H100, L40S, RTX 5070 Ti (16GB+ VRAM अनुशंसित)
- **CUDA**: 12.1 / 12.4 / 12.8 + PyTorch 2.x
- **खुले पोर्ट्स (Open Ports)**: `8000` (Studio Web UI & API) और `8188` (ComfyUI GPU Engine)

---

### ⚡ चरण 1: सर्वर टर्मिनल (SSH / Web Terminal) में यह कमांड चलाएं

```bash
# 1. रिपॉजिटरी क्लोन करें और डायरेक्टरी में जाएं
git clone https://github.com/harshsamrat-lgtm/harshAiVideo1.git
cd harshAiVideo1

# 2. ऑटोमेशन सेटअप स्क्रिप्ट चलाएं (सिस्टम टूल्स + डिपेंडेंसीज़)
chmod +x deploy/runpod_setup.sh
./deploy/runpod_setup.sh
```

---

### 🌐 चरण 2: अगर डायरेक्ट पोर्ट उपलब्ध न हो तो पब्लिक टनल से चलाएं (Cloudflare Tunnel)

अगर आप Vast.ai या ऐसे सर्वर पर हैं जहाँ पोर्ट 8000 सीधे बाहर नहीं खुलता, तो टनल स्क्रिप्ट चलाएं:

```bash
chmod +x start_server_tunnel.sh
./start_server_tunnel.sh
```
यह स्क्रीन पर एक फ्री और सुरक्षित **`https://*.trycloudflare.com`** पब्लिक लिंक देगा, जिसपर क्लिक करके आप सीधे अपने मोबाइल या कंप्यूटर ब्राउज़र में स्टूडियो खोल सकते हैं!

---

### 🔄 सर्वर या ऐप रीस्टार्ट करने के लिए:
```bash
chmod +x restart_app.sh
./restart_app.sh
```

---

## 💻 लोकल मशीन (Windows / Mac / Linux) पर कैसे चलाएं

```bash
# 1. डिपेंडेंसी इंस्टॉल करें
pip install -r backend/requirements.txt

# 2. सर्वर स्टार्ट करें
python start_server.py
```
👉 ब्राउज़र में खोलें: **http://127.0.0.1:8000**  
👉 API दस्तावेज़ (Swagger Docs): **http://127.0.0.1:8000/docs**

---

## 🌟 मुख्य विशेषताएं (v3.0 Core Features)

1. **🎭 15-सेकंड नेटिव AI वीडियो (15s Max Native Scenes)**: प्रत्येक सीन को 15 सेकंड की सिनेमाई अवधि में रेंडर करना।
2. **🏰 100% लोकेशन डीएनए (Location Continuity)**: जब भी कोई स्थान दोबारा आता है, उसका परिवेश और वास्तुकला 100% स्थिर रहती है।
3. **👤 360° कैरेक्टर फेस कंसिस्टेंसी (FaceID Consistency)**: मास्टर फेस प्रोफाइल और कॉस्ट्यूम निरंतरता।
4. **🎙️ न्यूरल हिंदी वॉइस स्टूडियो (Edge-TTS)**: भाव-आधारित (Emotion-aware) हिंदी आवाज़ें, ऑडियो नॉर्मलाइज़ेशन और स्टीरियो बैकग्राउंड स्कोर।
5. **🎬 मल्टी-मॉडल रूटिंग (Intelligent Model Router)**: Wan2.1-14B, MiniMax H3, Wan2.1-1.3B, SVD-XT या Cloud Diffusion में से 1-क्लिक चुनाव।
6. **🎞️ सिनेमा मास्टर असेंबलर**: ओपनिंग टाइटल कार्ड, सीन ट्रांजिशन्स, क्लोजिंग क्रेडिट्स और टाइमस्टैम्प्ड सबटाइटल (.SRT)।
7. **✨ प्रीमियम ग्लासमोर्फिज्म UI**: टोस्ट नोटिफिकेशन्स, प्रोग्रेस बार, कीबोर्ड शॉर्टकट (1-5), और ऑटो-सेव।
