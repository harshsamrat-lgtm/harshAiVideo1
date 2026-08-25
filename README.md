# 🎬 AI Hindi Cinema Studio (MiniMax H3 - 15s Native Engine)

AI-संचालित एंड-टू-एंड हिंदी स्टोरी-टू-मूवी जनरेटर सॉफ्टवेयर, जो **MiniMax H3 (Open-Weights)** और **पूर्ण त्रिकोणीय निरंतरता (कलाकार की शक्ल + आवाज + स्थानों की एकरूपता)** पर आधारित है।

---

## 🚀 रेंटेड GPU सर्वर (RunPod / Vast.ai / Hetzner) पर Git आधारित डिप्लॉयमेंट गाइड

### चरण 1: अपने लोकल प्रोजेक्ट को GitHub पर पुश करें (Push to GitHub)
```bash
# 1. अपने लोकल डायरेक्टरी में जाएँ
cd C:\Users\HP\.gemini\antigravity-ide\scratch\ai-hindi-cinema-studio

# 2. Git रिपॉजिटरी इनिशियलाइज़ करें
git init
git add .
git commit -m "feat: AI Hindi Cinema Studio with MiniMax H3 15s native scenes"

# 3. GitHub पर बनाई गई अपनी रिपॉजिटरी से जोड़ें
git remote add origin https://github.com/<your-username>/ai-hindi-cinema-studio.git
git branch -M main
git push -u origin main
```

---

### चरण 2: रेंटेड GPU सर्वर (RunPod / Vast.ai) किराए पर लें
1. **GPU चुनें**: `1x NVIDIA RTX 4090 (24GB VRAM)` या `1x NVIDIA A100 (80GB VRAM)`.
2. **Template / OS**: `Ubuntu 22.04 + PyTorch 2.4 + CUDA 12.4`.
3. **Exposed HTTP Ports**: `8000` (Studio UI & API) और `8188` (ComfyUI).

---

### चरण 3: सर्वर पर Git Clone और 1-क्लिक स्टार्ट करें
रेंटेड सर्वर के वेब टर्मिनल (Web Terminal / SSH) में निम्नलिखित कमांड्स चलाएं:

```bash
# 1. अपनी Git रिपॉजिटरी क्लोन करें
git clone https://github.com/<your-username>/ai-hindi-cinema-studio.git
cd ai-hindi-cinema-studio

# 2. ऑटोमेशन स्क्रिप्ट को अनुमति देकर रन करें
chmod +x deploy/runpod_setup.sh
./deploy/runpod_setup.sh
```

---

### चरण 4: अपने वेब ब्राउज़र में खोलें
- RunPod / Vast.ai पर **Connect -> HTTP Service (Port 8000)** पर क्लिक करें।
- आपका **AI Hindi Cinema Studio Dashboard** खुल जाएगा!

---

## 🌟 मुख्य विशेषताएं (Core Capabilities)
1. **MiniMax H3 अधिकतम 15-सेकंड सीन (15s Native Max Duration)**: प्रत्येक सीन को 15 सेकंड के सिनेमाई शॉट में उत्पन्न करना।
2. **स्थानों की एकरूपता (Location DNA)**: जब भी कहानी में कोई स्थान दोबारा आएगा, उसका आर्किटेक्चर और प्रॉप्स 100% स्थिर रहेंगे।
3. **कलाकार की शक्ल व वेशभूषा (Character Face Consistency)**: 360° मास्टर फेस प्रोफाइल और `H3-Base-Ref2VA`।
4. **कलाकार की आवाज (Voice Studio)**: फिक्सड न्यूरल हिंदी वॉइस क्लोनिंग और ऑटोमैटिक ऑडियो डकिंग।
5. **दो-स्तरीय अप्रूवल (Draft vs Final)**: पहले त्वरित 480p/720p ड्राफ्ट प्रीव्यू, स्वीकृति के बाद 1080p/4K मास्टर रेंडर।
