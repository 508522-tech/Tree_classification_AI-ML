import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
from PIL import Image as PILImage

# ✅ Load model
model = load_model("tree_classifier_mobilenetv2.keras")

# ✅ Class labels (must match your training folders!)
class_labels = {
    0: 'oak',
    1: 'maple',
    2: 'pine',
    3: 'birch',
    # ... add ALL classes here, total must match your dataset!
}

st.title("🌳 Tree Species Classifier")

# ✅ Upload & predict
uploaded_file = st.file_uploader("Upload a tree image...", type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
    img = PILImage.open(uploaded_file)
    st.image(img, caption='Uploaded Image', use_column_width=True)

    if st.button("Predict"):
        img = img.resize((224, 224))
        img_array = image.img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        pred = model.predict(img_array)
        class_index = np.argmax(pred)
        confidence = np.max(pred)

        st.write(f"**Prediction:** {class_labels[class_index]}")
        st.write(f"**Confidence:** {confidence:.2f}")

# ✅ Confusion Matrix
st.header("📊 Live Confusion Matrix")

if st.button("Generate Confusion Matrix"):
    # ⚡ Load val data
    datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

    val_data = datagen.flow_from_directory(
        "Tree_Species_Dataset",
        target_size=(224, 224),
        batch_size=32,
        class_mode='categorical',
        subset='validation',
        shuffle=False
    )

    # ⚡ Predict all
    Y_pred = model.predict(val_data)
    y_pred = np.argmax(Y_pred, axis=1)
    y_true = val_data.classes

    class_names = list(val_data.class_indices.keys())
    full_label_indices = list(range(len(class_names)))

    # ✅ Safe confusion matrix with full labels
    cm = confusion_matrix(y_true, y_pred, labels=full_label_indices)

    st.write("### Confusion Matrix")
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names,
                yticklabels=class_names,
                ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    st.pyplot(fig)

    st.write("### Classification Report")
    report = classification_report(
        y_true, y_pred,
        target_names=class_names,
        labels=full_label_indices,
        zero_division=0
    )
    st.text(report)
