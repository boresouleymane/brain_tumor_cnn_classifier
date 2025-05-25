import os
from flask import Flask, request, render_template
from PIL import Image
import numpy as np
import torch
import tensorflow as tf
import torchvision.transforms as transforms

# Flask app
app = Flask(__name__)

# Classes à prédire
class_names = ['glioma', 'meningioma', 'notumor', 'pituitary']

# ====== Chargement modèles ======
# ✅ TensorFlow
from models.cnn_tf import get_pretrained_model as get_tf_model
tf_model = get_tf_model(num_classes=4)
tf_model.load_weights("models/souleymane_bore_model.keras")  # Format .keras
print("✅ Modèle TensorFlow chargé.")

# ✅ PyTorch
from models.cnn_torch import get_pretrained_model as get_torch_model
torch_model = get_torch_model()
torch_model.load_state_dict(torch.load("models/souleymane_bore_model.torch", map_location=torch.device('cpu')))
torch_model.eval()
print("✅ Modèle PyTorch chargé.")

# ====== Prétraitements ======
def preprocess_image_tf(img):
    img = img.resize((224, 224))
    img_array = np.array(img) / 127.5 - 1.0
    return np.expand_dims(img_array, axis=0)

def preprocess_image_torch(img):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])
    return transform(img).unsqueeze(0)

# ====== Routes ======
@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None

    if request.method == "POST":
        model_choice = request.form.get("model")
        file = request.files.get("image")

        if file:
            img = Image.open(file).convert("RGB")

            if model_choice == "tensorflow":
                img_array = preprocess_image_tf(img)
                preds = tf_model.predict(img_array)
                predicted_class = class_names[np.argmax(preds)]

            elif model_choice == "torch":
                img_tensor = preprocess_image_torch(img)
                with torch.no_grad():
                    preds = torch_model(img_tensor)
                    predicted_class = class_names[torch.argmax(preds).item()]

            prediction = f"Classe prédite : {predicted_class}"

    return render_template("index.html", prediction=prediction)

# Lancement
if __name__ == "__main__":
    app.run(debug=True)
