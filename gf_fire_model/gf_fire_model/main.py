


# import tensorflow as tf
# from tensorflow.keras.models import load_model
# import numpy as np
# import matplotlib.pyplot as plt
# import os
# from PIL import Image
# import warnings
# warnings.filterwarnings('ignore')

# print("✅ All libraries loaded successfully!")

# # %%
# # Load your CustomCNN model
# model_path = 'CustomCNN_best_model.h5'

# if os.path.exists(model_path):
#     custom_cnn_model = load_model(model_path)
#     print(f"✅ CustomCNN model loaded from {model_path}")
#     print(f"Model input shape: {custom_cnn_model.input_shape}")
# else:
#     print(f"❌ Model file not found: {model_path}")
#     print("Please upload CustomCNN_best_model.h5 to /content/ directory")
#     custom_cnn_model = None

# # %%
# # 🔥 CORRECTED Fire Detection System
# class FixedFireDetector:
#     def __init__(self, model):
#         self.model = model
#         self.img_size = (224, 224)
#         self.classes = ['No Fire', 'Fire']

#     def preprocess_image(self, image_path):
#         """Preprocess image for model prediction"""
#         img = tf.keras.preprocessing.image.load_img(image_path, target_size=self.img_size)
#         img_array = tf.keras.preprocessing.image.img_to_array(img)
#         img_array = img_array / 255.0  # Normalize to 0-1
#         img_array = np.expand_dims(img_array, axis=0)
#         return img_array

#     def predict(self, image_path, threshold=0.5):
#         """CORRECTED prediction logic - fire images will show as FIRE"""

#         # Preprocess image
#         processed_img = self.preprocess_image(image_path)

#         # Get model prediction
#         raw_prediction = self.model.predict(processed_img, verbose=0)[0][0]

#         # 🔧 CORRECTED LOGIC: Flip the prediction since classes are inverted
#         # If model gives high score, it means "No Fire" in the original training
#         # If model gives low score, it means "Fire" in the original training

#         predicted_class = 0 if raw_prediction > threshold else 1  # FLIPPED

#         # Calculate confidence correctly
#         if predicted_class == 1:  # Fire
#             confidence = 1 - raw_prediction  # Higher confidence when raw score is lower
#         else:  # No Fire
#             confidence = raw_prediction  # Higher confidence when raw score is higher

#         result = {
#             'raw_score': float(raw_prediction),
#             'predicted_class': int(predicted_class),
#             'class_name': self.classes[predicted_class],
#             'confidence': float(confidence),
#             'is_fire': bool(predicted_class == 1),
#             'threshold': threshold
#         }

#         return result

#     def visualize_prediction(self, image_path, threshold=0.5, figsize=(10, 6)):
#         """Show image with corrected prediction"""

#         # Make prediction
#         result = self.predict(image_path, threshold)

#         # Load and display image
#         img = Image.open(image_path)

#         plt.figure(figsize=figsize)
#         plt.imshow(img)
#         plt.axis('off')

#         # Style based on corrected prediction
#         if result['is_fire']:
#             color = 'red'
#             emoji = '🔥'
#             alert = 'FIRE DETECTED!'
#             border_color = 'red'
#         else:
#             color = 'green'
#             emoji = '✅'
#             alert = 'NO FIRE'
#             border_color = 'green'

#         title = (
#             f"{emoji} {alert}\n" +
#             f"Confidence: {result['confidence']:.1%}\n" +
#             f"Raw Score: {result['raw_score']:.3f} (Corrected)"
#         )

#         plt.title(title, color=color, fontsize=16, fontweight='bold', pad=20)

#         # Add colored border
#         ax = plt.gca()
#         for spine in ax.spines.values():
#             spine.set_edgecolor(border_color)
#             spine.set_linewidth(5)

#         plt.tight_layout()
#         plt.show()

#         # Print detailed results
#         print(f"\n{'='*60}")
#         print(f"🔍 CORRECTED FIRE DETECTION RESULTS")
#         print(f"{'='*60}")
#         print(f"🎯 Final Result: {result['class_name']}")
#         print(f"📊 Confidence: {result['confidence']:.1%}")
#         print(f"🔢 Raw Model Score: {result['raw_score']:.4f}")
#         print(f"⚙️ Threshold Used: {result['threshold']}")
#         print(f"🔥 Fire Detected: {'YES' if result['is_fire'] else 'NO'}")
#         print(f"🔧 Logic: CORRECTED (Inverted from original model)")

#         if result['is_fire']:
#             if result['confidence'] > 0.8:
#                 print("🚨 HIGH CONFIDENCE FIRE DETECTION! Take immediate action if real scenario.")
#             else:
#                 print("⚠️ Fire detected - verify if needed.")
#         else:
#             print("✅ SAFE: No fire detected in the image.")

#         return result

# # Initialize the corrected fire detector
# if custom_cnn_model is not None:
#     fire_detector = FixedFireDetector(custom_cnn_model)
#     print("🔥 CORRECTED Fire Detector Ready!")
#     print("✅ Fire images will now correctly show as FIRE")
#     print("✅ No-fire images will now correctly show as NO FIRE")
# else:
#     fire_detector = None

# # %%
# import tkinter as tk
# from tkinter import filedialog
# import os

# def upload_and_detect_fire():
#     """Open GUI with a button to select an image and run fire detection"""

#     if fire_detector is None:
#         print("❌ Fire detector not available. Please load the model first.")
#         return

#     def select_file():
#         filename = filedialog.askopenfilename(
#             title="Select an image",
#             filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp")]
#         )
#         if filename:
#             status_label.config(text=f"🔍 Analyzing: {os.path.basename(filename)}")
#             try:
#                 fire_detector.visualize_prediction(filename)
#                 status_label.config(text=f"✅ Analysis complete: {os.path.basename(filename)}")
#             except Exception as e:
#                 status_label.config(text=f"❌ Error: {e}")
#         else:
#             status_label.config(text="❌ No file selected")

#     # GUI Window
#     root = tk.Tk()
#     root.title("🔥 Fire Detection System")

#     # Button
#     btn = tk.Button(root, text="Choose File", command=select_file, font=("Arial", 14), padx=10, pady=5)
#     btn.pack(pady=20)

#     # Status Label
#     status_label = tk.Label(root, text="Please select an image file...", font=("Arial", 12))
#     status_label.pack(pady=10)

#     root.mainloop()

# print("🚀 READY TO DETECT FIRE WITH CORRECTED RESULTS!")
# upload_and_detect_fire()


# # %%



import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
import os
from PIL import Image
import warnings

warnings.filterwarnings('ignore')

print("✅ All libraries loaded successfully!")

# ================================
# Load your CustomCNN model
# ================================
MODEL_PATH = 'CustomCNN_best_model.h5'

if os.path.exists(MODEL_PATH):
    custom_cnn_model = load_model(MODEL_PATH)
    print(f"✅ CustomCNN model loaded from {MODEL_PATH}")
    print(f"Model input shape: {custom_cnn_model.input_shape}")
else:
    print(f"❌ Model file not found: {MODEL_PATH}")
    print("Please place 'CustomCNN_best_model.h5' in the working directory.")
    custom_cnn_model = None


# ================================
# Fire Detection Class
# ================================
class FixedFireDetector:
    def __init__(self, model):
        self.model = model
        self.img_size = (224, 224)
        self.classes = ['No Fire', 'Fire']

    def preprocess_image(self, image_path):
        """Preprocess image for model prediction"""
        img = tf.keras.preprocessing.image.load_img(image_path, target_size=self.img_size)
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        img_array = img_array / 255.0  # Normalize to 0-1
        img_array = np.expand_dims(img_array, axis=0)
        return img_array

    def predict(self, image_path, threshold=0.5):
        """Predict fire or no fire from an image path"""
        processed_img = self.preprocess_image(image_path)

        raw_prediction = self.model.predict(processed_img, verbose=0)[0][0]

        # Correct inverted logic
        predicted_class = 0 if raw_prediction > threshold else 1

        if predicted_class == 1:  # Fire
            confidence = 1 - raw_prediction
        else:  # No Fire
            confidence = raw_prediction

        return {
            'raw_score': float(raw_prediction),
            'predicted_class': int(predicted_class),
            'class_name': self.classes[predicted_class],
            'confidence': float(confidence),
            'is_fire': bool(predicted_class == 1),
            'threshold': threshold
        }


# ================================
# Initialize detector for API use
# ================================
fire_detector = None
if custom_cnn_model is not None:
    fire_detector = FixedFireDetector(custom_cnn_model)
    print("🔥 Fire Detector API Ready!")
else:
    print("❌ Fire Detector not initialized (model missing).")
