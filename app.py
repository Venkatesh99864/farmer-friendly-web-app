import os
import io
import json
import pickle
import traceback
import zipfile
from PIL import Image
from pathlib import Path
from dotenv import load_dotenv
from flask import Flask, request, jsonify, render_template
import numpy as np
import requests
import tensorflow as tf
from openai import OpenAI 
import mlflow 
import time

# ============================================================
# 🔐 ENV LOADING
# ============================================================
load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
WEATHER_API_KEY = os.getenv("WEATHER_API_KEY") 
SECRET_KEY = os.getenv("SECRET_KEY", "a-default-secret-key")
CROP_MODEL_DRIVE_URL = os.getenv("CROP_MODEL_DRIVE_URL")
DISEASE_MODEL_DRIVE_URL = os.getenv("DISEASE_MODEL_DRIVE_URL")

# Check Keys
if not GROQ_API_KEY: print("❌ ERROR: GROQ_API_KEY missing.")
if not WEATHER_API_KEY: print("❌ ERROR: WEATHER_API_KEY missing.")

# Local model paths
CROP_MODEL_PATH = Path("models/crop_recommendation.pkl")
DISEASE_MODEL_PATH = Path("models/crop_disease_cnn_model.keras")
CROP_MODEL_PATH.parent.mkdir(exist_ok=True)

# ============================================================
# 🛠️ Google Drive + Dropbox Download (FULLY FIXED)
# ============================================================
def is_valid_model_file(file_path):
    """Validate if file is real model (not HTML)"""
    try:
        if not file_path.exists() or file_path.stat().st_size < 1000:
            return False
        
        with open(file_path, 'rb') as f:
            header = f.read(50)
            
            # Pickle (.pkl) validation
            if file_path.suffix == '.pkl':
                return header.startswith(b'\x80')
            
            # Keras (.keras) = ZIP validation
            if file_path.suffix == '.keras':
                return header.startswith(b'PK\x03\x04')
            
        return False
    except:
        return False

def download_model(url, local_path, max_retries=3):
    """Universal download: Google Drive + Dropbox + Direct links"""
    if local_path.exists() and is_valid_model_file(local_path):
        print(f"✔ VALID model exists: {local_path} ({local_path.stat().st_size} bytes)")
        return True
    
    if local_path.exists() and not is_valid_model_file(local_path):
        print(f"❌ Corrupted file, deleting: {local_path}")
        local_path.unlink()
    
    print(f"📥 Downloading to {local_path}...")
    
    # 🚀 DROPBOX (NEW!)
    if 'dropbox' in url.lower():
        if 'scl/fi' in url:
            # Dropbox shared link format
            direct_url = url.replace('dl=0', 'raw=1').replace('dl=1', 'raw=1')
        else:
            direct_url = url.replace('www.dropbox.com', 'dl.dropboxusercontent.com').replace('?dl=0', '').replace('?dl=1', '')
        
        print(f"🔗 Dropbox: {direct_url[:60]}...")
        try:
            resp = requests.get(direct_url, stream=True, timeout=60)
            resp.raise_for_status()
            
            with open(local_path, "wb") as f:
                for chunk in resp.iter_content(chunk_size=8192):
                    if chunk: f.write(chunk)
            
            if is_valid_model_file(local_path):
                print(f"✔ Dropbox SUCCESS: {local_path.stat().st_size} bytes")
                return True
            else:
                local_path.unlink()
                print("❌ Dropbox file invalid")
                return False
        except Exception as e:
            print(f"❌ Dropbox failed: {e}")
            return False
    
    # 🚀 GOOGLE DRIVE
    if "/file/d/" in url:
        file_id = url.split("/file/d/")[1].split("/")[0]
    elif "id=" in url:
        file_id = url.split("id=")[1].split("&")[0]
    else:
        print(f"❌ Invalid URL format: {url}")
        return False
    
    download_urls = [
        f"https://drive.google.com/uc?export=download&id={file_id}",
        f"https://drive.google.com/uc?id={file_id}&export=download",
        f"https://docs.google.com/uc?export=download&id={file_id}"
    ]
    
    for i, d_url in enumerate(download_urls):
        for attempt in range(max_retries):
            try:
                print(f"🔄 Google Drive {i+1}/3, Attempt {attempt+1}/{max_retries}")
                resp = requests.get(d_url, stream=True, timeout=30)
                resp.raise_for_status()
                
                # Check HTML scam page
                content_preview = resp.content[:1000].decode('utf-8', errors='ignore')
                if any(x in content_preview.lower() for x in ['html', 'drive.google.com', 'virus']):
                    print("⚠️ Google Drive HTML - next URL")
                    break
                
                with open(local_path, "wb") as f:
                    for chunk in resp.iter_content(chunk_size=8192):
                        if chunk: f.write(chunk)
                
                if is_valid_model_file(local_path):
                    print(f"✔ Google Drive SUCCESS: {local_path.stat().st_size} bytes")
                    return True
                else:
                    local_path.unlink()
                    break
                    
            except Exception as e:
                print(f"⚠️ Google Drive failed: {e}")
                if attempt < max_retries - 1:
                    time.sleep(2)
    
    print(f"❌ All download attempts failed")
    return False

# ============================================================
# ⚙️ MLOps Setup
# ============================================================
try:
    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment("Farmer_App_Monitoring")
    print("✔ MLflow active.")
except Exception as e:
    print(f"⚠️ MLflow: {e}")

# ============================================================
# 🤖 AI Setup
# ============================================================
ai_client = None
try:
    if GROQ_API_KEY:
        ai_client = OpenAI(base_url="https://api.groq.com/openai/v1", api_key=GROQ_API_KEY)
        print("✔ Groq loaded.")
except Exception as e:
    print(f"❌ AI Error: {e}")

app = Flask(__name__, static_folder="static", template_folder="templates")
app.config["SECRET_KEY"] = SECRET_KEY

# ============================================================
# 🌐 Translations
# ============================================================
translations = {
    'en': {
        'app_title': 'Farmer Friendly App', 'developed_by': 'guided by', 'language': 'Language',
        'dashboard': 'Dashboard', 'crop_recommendation': 'Crop Recommendation', 'disease_detection': 'Disease Detection',
        'fertilizer_advisory': 'Fertilizer Advisory', 'ai_chatbot': 'AI Chatbot', 'weather': 'Weather',
        'crop_recommendation_desc': 'Get scientific crop suggestions based on soil.',
        'disease_detection_desc': 'Identify plant diseases instantly.',
        'fertilizer_advisory_desc': 'Get expert fertilizer and treatment recommendations.',
        'ai_chatbot_desc': 'Your 24/7 farming assistant.',
        'weather_desc': 'Live local weather updates & 15-day forecast.',
        'location_label': 'Location (village/district)', 'season_label': 'Season', 'soil_type_label': 'Soil Type',
        'irrigation_label': 'Irrigation', 'area_label': 'Field Area (hectares)', 'prev_crop_label': 'Previous Crop',
        'budget_label': 'Budget', 'market_label': 'Market preference', 'get_recommendation_btn': 'Get Recommendation',
        'top_recommendations': 'Top Recommendations', 'features_estimated': '',
        'upload_label': 'Upload leaf image', 'detect_btn': 'Detect Disease', 'get_fertilizer_btn': 'Get Fertilizer Advice',
        'disease_result': 'Disease', 'confidence_result': 'Confidence', 'healthy_result': 'The plant looks healthy!',
        'crop_label': 'Crop', 'disease_label': 'Disease (if known)', 'get_advisory_btn': 'Get Advisory',
        'city_label': 'City or use GPS', 'check_weather_btn': 'Check Weather', 'use_gps_btn': 'Use GPS',
        'weather_feels_like': 'Feels like', 'weather_humidity': 'Humidity', 'weather_wind': 'Wind',
        'forecast_title': '15-Day Forecast', 'forecast_date': 'Date', 'forecast_max': 'Max', 'forecast_min': 'Min', 'forecast_condition': 'Condition',
        'chat_label': 'Ask your question', 'chat_btn': 'Ask', 'chat_bot_label': 'Bot',
        'rice': 'Rice', 'maize': 'Maize', 'chickpea': 'Chickpea', 'Healthy': 'Healthy', 'Powdery': 'Powdery', 'Rust': 'Rust',
    },
    'hi': {
        'app_title': 'किसान फ्रेंडली ऐप', 'developed_by': 'मार्गदर्शन', 'language': 'भाषा',
        'dashboard': 'डैशबोर्ड', 'crop_recommendation': 'फसल की सिफारिश', 'disease_detection': 'रोग की पहचान',
        'fertilizer_advisory': 'उर्वरक सलाह', 'ai_chatbot': 'एआई चैटबॉट', 'weather': 'मौसम',
        'crop_recommendation_desc': 'वैज्ञानिक फसल सुझाव प्राप्त करें।',
        'disease_detection_desc': 'तुरंत पौधों के रोगों की पहचान करें।',
        'fertilizer_advisory_desc': 'विशेषज्ञ उर्वरक सिफारिशें प्राप्त करें।',
        'ai_chatbot_desc': 'आपका 24/7 कृषि सहायक।',
        'weather_desc': 'लाइव स्थानीय मौसम अपडेट और 15-दिन का पूर्वानुमान।',
        'location_label': 'स्थान (गाँव/जिला)', 'season_label': 'मौसम', 'soil_type_label': 'मिट्टी का प्रकार',
        'irrigation_label': 'सिंचाई', 'area_label': 'क्षेत्र (हेक्टेयर)', 'prev_crop_label': 'पिछली फसल',
        'budget_label': 'बजट', 'market_label': 'बाजार वरीयता', 'get_recommendation_btn': 'सिफारिश प्राप्त करें',
        'top_recommendations': 'शीर्ष सिफारिशें', 'features_estimated': '',
        'upload_label': 'पत्ती की छवि अपलोड करें', 'detect_btn': 'रोग का पता लगाएं', 'get_fertilizer_btn': 'उर्वरक सलाह',
        'disease_result': 'रोग', 'confidence_result': 'आत्मविश्वास', 'healthy_result': 'पौधा स्वस्थ है!',
        'crop_label': 'फसल', 'disease_label': 'रोग', 'get_advisory_btn': 'सलाह लें',
        'city_label': 'शहर या जीपीएस', 'check_weather_btn': 'मौसम जांचें', 'use_gps_btn': 'जीपीएस',
        'weather_feels_like': 'जैसा लगता है', 'weather_humidity': 'नमी', 'weather_wind': 'हवा',
        'forecast_title': '15-दिन का पूर्वानुमान', 'forecast_date': 'तारीख', 'forecast_max': 'अधिकतम', 'forecast_min': 'न्यूनतम', 'forecast_condition': 'स्थिति',
        'chat_label': 'प्रश्न पूछें', 'chat_btn': 'पूछें', 'chat_bot_label': 'बॉट',
        'rice': 'चावल', 'maize': 'मक्का', 'chickpea': 'चना', 'Healthy': 'स्वस्थ', 'Powdery': 'पाउडर फफूंद', 'Rust': 'जंग',
    },
    'te': {
        'app_title': 'రైతు స్నేహపూర్వక యాప్', 'developed_by': 'మార్గదర్శకత్వం', 'language': 'భాష',
        'dashboard': 'డాష్‌బోర్డ్', 'crop_recommendation': 'పంట సిఫార్సు', 'disease_detection': 'వ్యాధి గుర్తింపు',
        'fertilizer_advisory': 'ఎరువుల సలహా', 'ai_chatbot': 'AI చాట్‌బాట్', 'weather': 'వాతావరణం',
        'crop_recommendation_desc': 'శాస్త్రీయ పంట సూచనలను పొందండి.',
        'disease_detection_desc': 'మొక్కల వ్యాధులను గుర్తించండి.',
        'fertilizer_advisory_desc': 'ఎరువుల సిఫార్సులను పొందండి.',
        'ai_chatbot_desc': 'మీ 24/7 వ్యవసాయ సహాయకుడు.', 'weather_desc': 'వాతావరణ నవీకరణలు & 15-రోజుల సూచన.',
        'location_label': 'స్థానం', 'season_label': 'సీజన్', 'soil_type_label': 'నేల రకం',
        'irrigation_label': 'నీటిపారుదల', 'area_label': 'విస్తీర్ణం', 'prev_crop_label': 'మునుపటి పంట',
        'budget_label': 'బడ్జెట్', 'market_label': 'మార్కెట్ ప్రాధాన్యత', 'get_recommendation_btn': 'సిఫార్సు పొందండి',
        'top_recommendations': 'అగ్ర సిఫార్సులు', 'features_estimated': '',
        'upload_label': 'ఫోటో అప్‌లోడ్ చేయండి', 'detect_btn': 'గుర్తించండి', 'get_fertilizer_btn': 'సలహా పొందండి',
        'disease_result': 'వ్యాధి', 'confidence_result': 'విశ్వాసం', 'healthy_result': 'మొక్క ఆరోగ్యంగా ఉంది!',
        'crop_label': 'పంట', 'disease_label': 'వ్యాధి', 'get_advisory_btn': 'సలహా పొందండి',
        'city_label': 'నగరం/GPS', 'check_weather_btn': 'తనిఖీ చేయండి', 'use_gps_btn': 'GPS',
        'weather_feels_like': 'అనిపిస్తుంది', 'weather_humidity': 'తేమ', 'weather_wind': 'గాలి',
        'forecast_title': '15-రోజుల సూచన', 'forecast_date': 'తేదీ', 'forecast_max': 'గరిష్ట', 'forecast_min': 'కనిష్ట', 'forecast_condition': 'స్థితి',
        'chat_label': 'ప్రశ్న అడగండి', 'chat_btn': 'అడగండి', 'chat_bot_label': 'బాట్',
        'rice': 'బియ్యం', 'maize': 'మొక్కజొన్న', 'chickpea': 'శనగలు', 'Healthy': 'ఆరోగ్యకరమైన', 'Powdery': 'బూజు తెగులు', 'Rust': 'కుంకుమ తెగులు',
    },
    'gu': {}
}

def get_lang_data():
    lang_code = request.args.get('lang', 'en')
    if lang_code not in translations or not translations[lang_code]: 
        lang_code = 'en'
    return translations[lang_code], lang_code

# ============================================================
# 🌱 MODEL LOADING (FULLY ROBUST)
# ============================================================
crop_model = None
disease_model = None
DISEASE_LABELS = ["Healthy", "Powdery", "Rust"]

print("🚀 Starting model initialization...")

# CROP MODEL
print(f"🌱 CROP_MODEL_DRIVE_URL: {'✅ Set' if CROP_MODEL_DRIVE_URL else '❌ Missing'}")
if CROP_MODEL_DRIVE_URL:
    if download_model(CROP_MODEL_DRIVE_URL, CROP_MODEL_PATH):
        try:
            with open(CROP_MODEL_PATH, "rb") as f:
                crop_model = pickle.load(f)
            print("✔ CROP model LOADED!")
        except Exception as e:
            print(f"❌ Crop load error: {e}")

# DISEASE MODEL  
print(f"🌱 DISEASE_MODEL_DRIVE_URL: {'✅ Set' if DISEASE_MODEL_DRIVE_URL else '❌ Missing'}")
if DISEASE_MODEL_DRIVE_URL:
    if download_model(DISEASE_MODEL_DRIVE_URL, DISEASE_MODEL_PATH):
        try:
            if DISEASE_MODEL_PATH.exists() and DISEASE_MODEL_PATH.stat().st_size > 50000:
                disease_model = tf.keras.models.load_model(str(DISEASE_MODEL_PATH.absolute()))
                print("✔ DISEASE model LOADED!")
            else:
                print(f"❌ Disease file too small: {DISEASE_MODEL_PATH.stat().st_size if DISEASE_MODEL_PATH.exists() else 0} bytes")
        except Exception as e:
            print(f"❌ Disease load error: {e}")

print("✅ Model init complete!")
print(f"📊 Status: Crop={'✅' if crop_model else '❌'} | Disease={'✅' if disease_model else '❌'}")

# ============================================================
# 🛠️ HELPERS
# ============================================================
def preprocess_image(pil_img, size=(224, 224)):
    img = pil_img.convert("RGB").resize(size)
    return np.expand_dims(np.array(img) / 255.0, axis=0)

def get_weather_full(city=None):
    if not WEATHER_API_KEY or not city: return None
    try:
        base_url = "https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline"
        url = f"{base_url}/{city}?unitGroup=metric&key={WEATHER_API_KEY}&contentType=json"
        res = requests.get(url, timeout=10)
        if res.status_code != 200: return None
        data = res.json()
        return data if 'days' in data else None
    except: return None

def get_current_weather(city):
    data = get_weather_full(city)
    if not data: return None
    curr = data.get('currentConditions', {})
    return {
        "temperature": curr.get('temp'),
        "humidity": curr.get('humidity'),
        "rainfall_mm": curr.get('precip', 0) or 0
    }

def estimate_features(data, weather=None):
    soil, season = data.get("soil_type", "loamy").lower(), data.get("season", "kharif").lower()
    prev, budget = data.get("previous_crop", "").lower(), data.get("budget", "medium").lower()
    N, P, K = 50.0, 25.0, 25.0
    if "clay" in soil: P *= 0.9; K *= 1.1
    if "sandy" in soil: N *= 1.05; P *= 1.1; K *= 0.9
    if prev in ["groundnut", "peanut", "soybean", "legume"]: N *= 0.7
    if budget == "low": N *= 0.85; P *= 0.85; K *= 0.85
    ph = 6.5
    if weather:
        temp = float(weather.get("temperature", 26))
        hum = float(weather.get("humidity", 65))
        rain = 180.0 if 'kharif' in season else (40.0 if 'rabi' in season else 20.0)
    else:
        print(f"--- Simulating weather for {season} ---")
        if 'kharif' in season: temp, hum, rain = 28.0, 85.0, 200.0
        elif 'rabi' in season: temp, hum, rain = 20.0, 45.0, 40.0
        elif 'zaid' in season: temp, hum, rain = 35.0, 35.0, 10.0
        else: temp, hum, rain = 26.0, 65.0, 40.0
    import random
    N += random.uniform(-5, 5); temp += random.uniform(-2, 2); hum += random.uniform(-5, 5)
    return [int(N), int(P), int(K), round(temp, 2), round(hum, 2), round(ph, 2), round(rain, 2)]

def call_ai_model(prompt):
    if not ai_client: return "[AI Error] Client not loaded."
    try:
        response = ai_client.chat.completions.create(
            model="llama-3.1-8b-instant", 
            messages=[{"role": "system", "content": "You are a helpful farming assistant. Be concise. Use Markdown."}, {"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content
    except Exception as e: return f"[Groq Error] {str(e)}"

# ============================================================
# 🛣️ ROUTES
# ============================================================
@app.route("/")
def dash(): 
    lang, code = get_lang_data()
    return render_template("dashboard.html", lang=lang, current_lang=code)

@app.route("/<page>")
def serve_page(page):
    lang, code = get_lang_data()
    if page in ["dashboard.html", "crop_recommendation.html", "disease_detection.html", 
                "fertilizer_advisory.html", "weather.html", "ai_interactive.html"]:
        return render_template(page, lang=lang, current_lang=code)
    return "Page not found", 404

@app.route("/api/predict_crop", methods=["POST"])
def predict_crop():
    try:
        lang_data, _ = get_lang_data()
        data = request.json
        weather = get_current_weather(data.get("city"))
        features = estimate_features(data, weather)
        
        if not crop_model: 
            return jsonify({"error": "Crop model unavailable"}), 503
            
        prediction = crop_model.predict([features])[0]
        translated_name = lang_data.get(prediction.lower(), prediction.capitalize())
        
        try: 
            with mlflow.start_run(run_name="Crop_Prediction"):
                mlflow.log_param("city", data.get("city"))
                mlflow.log_param("prediction", prediction)
        except: pass

        return jsonify({
            "input_features": features,
            "recommendations": [{"crop": translated_name, "score": 1.0}],
            "weather_used": weather or "Simulated"
        })
    except Exception as e:
        print(f"❌ Crop error: {e}")
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route("/api/predict_disease", methods=["POST"])
def predict_disease():
    if not disease_model: 
        return jsonify({"error": "Disease model unavailable"}), 503
    try:
        lang_data, _ = get_lang_data()
        img = Image.open(request.files["image"])
        arr = preprocess_image(img)
        preds = disease_model.predict(arr)[0]
        idx = int(np.argmax(preds))
        
        label_key = DISEASE_LABELS[idx]
        translated_label = lang_data.get(label_key, label_key)
        
        try: 
            with mlflow.start_run(run_name="Disease_Detection"):
                mlflow.log_param("disease", label_key)
        except: pass
        
        return jsonify({"label": translated_label, "label_key": label_key, "confidence": float(preds[idx])})
    except Exception as e:
        print(f"❌ Disease error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/api/fertilizer_advice", methods=["POST"])
def fertilizer_advice():
    data = request.json
    _, code = get_lang_data()
    prompt = f"Act as agronomist. Language: {code}. Crop: {data.get('crop')}. Disease: {data.get('disease')}. Soil: {data.get('soil_type')}. Suggest NPK fertilizer + treatment. Markdown. Short."
    return jsonify({"recommendation": call_ai_model(prompt)})

@app.route("/api/chat", methods=["POST"])
def chatbot():
    data = request.json
    _, code = get_lang_data()
    prompt = f"""Helpful farmer assistant. ONLY {code}. Answer ONLY farming/crop/soil/weather/livestock. Decline other topics in {code}. Farmer: {data.get('message')}"""
    return jsonify({"response": call_ai_model(prompt)})

@app.route("/api/weather")
def api_weather():
    q = request.args.get('q')
    data = get_weather_full(city=q)
    if not data: return jsonify({"error": "Weather unavailable"}), 500
    
    current = data.get('currentConditions', {})
    days = data.get('days', [])[:15]
    
    return jsonify({
        "name": data.get('address', q),
        "current": {
            "temp": current.get('temp'),
            "humidity": current.get('humidity'),
            "feelslike": current.get('feelslike'),
            "conditions": current.get('conditions'),
            "wind": current.get('windspeed'),
            "icon": current.get('icon')
        },
        "forecast": days 
    })

if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
