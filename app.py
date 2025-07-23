import os
import numpy as np
from flask import Flask, request, render_template, jsonify
import tensorflow as tf
from prometheus_flask_exporter import PrometheusMetrics
from prometheus_client import Gauge

app = Flask(__name__)
metrics = PrometheusMetrics(app)

# Menambahkan custom metrics
metrics.info('app_info', 'Application info', version='1.0.0')
prediction_counter = metrics.counter(
    'prediction_requests_total', 'Number of prediction requests',
    labels={'status': lambda resp: resp.status_code}
)
prediction_latency = metrics.histogram(
    'prediction_latency_seconds', 'Time spent processing prediction requests',
    labels={'status': lambda resp: resp.status_code}
)

# Definisikan gauge menggunakan prometheus_client langsung
prediction_value_gauge = Gauge('last_prediction_value', 'Last prediction value')

# Load model
MODEL_DIR = os.path.join('kstarid-pipeline', 'Pusher', 'pushed_model', '12')

# Konstanta dari apple_transform.py
NUMERIC_FEATURES = [
    'A_id', 'Size', 'Weight', 'Sweetness', 
    'Crunchiness', 'Juiciness', 'Ripeness', 
    'Acidity'
]

def transformed_name(key):
    return f"{key}_xf"

# Coba load model jika ada
try:
    model = tf.saved_model.load(MODEL_DIR)
    print(f"Model loaded successfully from {MODEL_DIR}")
    model_loaded = True
except Exception as e:
    print(f"Warning: Could not load model: {e}")
    model_loaded = False

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction_result = None
    probability = None
    
    if request.method == 'POST':
        # Collect data from form
        try:
            data = {
                'A_id': float(request.form.get('A_id', 1)),
                'Size': float(request.form.get('Size', 0.5)),
                'Weight': float(request.form.get('Weight', 0.5)),
                'Sweetness': float(request.form.get('Sweetness', 0.5)),
                'Crunchiness': float(request.form.get('Crunchiness', 0.5)),
                'Juiciness': float(request.form.get('Juiciness', 0.5)),
                'Ripeness': float(request.form.get('Ripeness', 0.5)),
                'Acidity': float(request.form.get('Acidity', 0.5))
            }
            
            # Since model loading might fail, use dummy prediction for Railway deployment
            if model_loaded:
                # Process data for model prediction - skipped for simplicity
                # Implement your actual prediction logic here
                
                # For now, use a simple rule-based prediction as a placeholder
                sweetness_weight = 0.3
                crunch_weight = 0.2
                juice_weight = 0.2
                ripe_weight = 0.2
                acid_weight = 0.1
                
                score = (data['Sweetness'] * sweetness_weight + 
                         data['Crunchiness'] * crunch_weight + 
                         data['Juiciness'] * juice_weight + 
                         data['Ripeness'] * ripe_weight - 
                         data['Acidity'] * acid_weight)
                
                # Normalize to 0-1
                normalized_score = min(max(score, 0), 1)
                
                prediction_value = normalized_score
            else:
                # Without model, use simple heuristic based on sweetness and juiciness
                prediction_value = (data['Sweetness'] * 0.4 + data['Juiciness'] * 0.3 + 
                                    data['Crunchiness'] * 0.3 - data['Acidity'] * 0.1)
                prediction_value = min(max(prediction_value, 0), 1)
            
            # Map to good/bad
            prediction_result = "GOOD" if prediction_value > 0.5 else "BAD"
            probability = round(max(prediction_value, 1 - prediction_value) * 100, 1)
            
            # Track prediction quality for monitoring - gunakan gauge langsung
            prediction_value_gauge.set(prediction_value)
            
        except Exception as e:
            print(f"Error in prediction: {e}")
            return f"Error processing request: {e}", 400
    
    return render_template('index.html', prediction=prediction_result, probability=probability)

@app.route('/health')
def health():
    return jsonify({
        "status": "healthy",
        "app_version": "1.0.0",
        "model_loaded": model_loaded
    })

# Tambahkan endpoint untuk monitoring
@app.route('/monitoring')
def monitoring_page():
    return render_template('monitoring.html')

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port, debug=False)