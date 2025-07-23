FROM tensorflow/serving:2.8.0

# Copy model ke direktori model di container dengan struktur yang benar
COPY kstarid-pipeline/Pusher/pushed_model/12 /models/apple_quality_model/1

# Set environment variable
ENV MODEL_NAME=apple_quality_model
ENV PORT=8501

# Expose port
EXPOSE 8501
EXPOSE 8500

# Start TensorFlow Serving
CMD tensorflow_model_server --rest_api_port=${PORT} --model_name=${MODEL_NAME} --model_base_path=/models/${MODEL_NAME} 