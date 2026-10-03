import streamlit as st
import tensorflow as tf
import numpy as np

# --- Professional PlantGuard AI UI ---
st.set_page_config(
    page_title="PlantGuard AI | Plant Disease Detection",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
.stApp {
    background:
        radial-gradient(circle at 8% 8%, rgba(76,175,80,.10), transparent 28%),
        radial-gradient(circle at 92% 18%, rgba(255,193,7,.08), transparent 24%),
        linear-gradient(135deg,#f7fbf5 0%,#ffffff 50%,#f2f8f1 100%);
}
#MainMenu {visibility:hidden;}
footer {visibility:hidden;}

section[data-testid="stSidebar"] {
    background: linear-gradient(180deg,#0d5c3b 0%,#0a4a31 100%);
}
section[data-testid="stSidebar"] * {color:white !important;}
section[data-testid="stSidebar"] [data-baseweb="select"] > div {
    background:rgba(255,255,255,.12);
    border:1px solid rgba(255,255,255,.20);
    border-radius:12px;
}
.brand {text-align:center;padding:8px 4px 18px;}
.brand-icon {font-size:42px;}
.brand-title {font-size:24px;font-weight:800;}
.brand-subtitle {font-size:12px;opacity:.78;}
.sidebar-note {
    margin-top:24px;padding:14px;border-radius:14px;
    background:rgba(255,255,255,.10);
    border:1px solid rgba(255,255,255,.12);
    font-size:12px;line-height:1.5;
}
.page-title {
    font-size:clamp(30px,4vw,48px);font-weight:850;
    line-height:1.05;color:#123b2a;margin-bottom:10px;
}
.page-subtitle {font-size:17px;color:#557064;line-height:1.6;}
.hero {
    padding:30px 34px;border-radius:26px;margin-bottom:24px;
    background:linear-gradient(135deg,#e7f6e9 0%,#f9fcf7 55%,#fff8e5 100%);
    border:1px solid rgba(35,111,72,.12);
    box-shadow:0 12px 35px rgba(27,74,49,.08);
}
.hero-badge {
    display:inline-block;padding:7px 12px;border-radius:999px;
    background:#d8f1dc;color:#17633f;font-weight:700;font-size:12px;
    margin-bottom:12px;
}
.hero-image {border-radius:22px;overflow:hidden;}
.feature-card,.result-card,.info-card {
    background:rgba(255,255,255,.92);
    border:1px solid #e4eee6;border-radius:18px;padding:20px;
    box-shadow:0 8px 25px rgba(24,71,47,.06);
}
.feature-card {height:100%;}
.feature-icon {font-size:28px;}
.feature-title {font-size:17px;font-weight:750;color:#164d35;}
.feature-text {color:#64766c;font-size:14px;line-height:1.5;}
div[data-testid="stFileUploader"] {
    background:#fff;border:2px dashed #9bc7a7;
    border-radius:18px;padding:8px;
}
.stButton > button {
    width:100%;border:none;border-radius:12px;padding:12px 18px;
    font-weight:750;background:linear-gradient(135deg,#16834f,#0d633c);
    color:white;box-shadow:0 7px 18px rgba(13,99,60,.18);
}
.prediction-heading {font-size:26px;font-weight:800;color:#123b2a;}
.confidence-number {font-size:30px;font-weight:850;color:#17633f;}
.section-label {
    font-size:13px;font-weight:800;color:#547063;
    text-transform:uppercase;letter-spacing:.7px;
}
</style>
""", unsafe_allow_html=True)


#Tenserflow Model Prediction
def model_prediction(test_image):
    model  = tf.keras.models.load_model('trained_model.h5')
    image = tf.keras.preprocessing.image.load_img(test_image,target_size=(128, 128))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr]) #Convert single image to a batch
    prediction = model.predict(input_arr)
    result_index = np.argmax(prediction)
    return result_index

# Sidebar
with st.sidebar:
    st.markdown("""
    <div class="brand">
        <div class="brand-icon">🌿</div>
        <div class="brand-title">PlantGuard AI</div>
        <div class="brand-subtitle">Plant Disease Detection</div>
    </div>
    """, unsafe_allow_html=True)

    app_mode = st.selectbox(
        "Navigate",
        ["Home", "About", "Disease Recognition"],
        format_func=lambda x: {
            "Home": "🏠  Home",
            "About": "📖  About",
            "Disease Recognition": "🔬  Disease Recognition",
        }[x],
    )

    st.markdown("""
    <div class="sidebar-note">
        <b>🧠 CNN-Based Analysis</b><br>
        Upload a clear leaf image to get a prediction, confidence score,
        treatment guidance and prevention tips.
    </div>
    """, unsafe_allow_html=True)

# Home Page
if(app_mode=="Home"):
    left, right = st.columns([1.05, 1], gap="large")

    with left:
        st.markdown("""
        <div style="padding:35px 5px 10px;">
            <div class="hero-badge">🌱 AI-POWERED PLANT HEALTH</div>
            <div class="page-title">Detect Plant Diseases<br>with AI</div>
            <div class="page-subtitle">
                Upload a plant leaf image and our CNN-based system will analyze it
                and provide a predicted plant condition with confidence,
                treatment guidance and prevention tips.
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.info("🔬 Select **Disease Recognition** from the sidebar to start.")

    with right:
        st.markdown('<div class="hero-image">', unsafe_allow_html=True)
        st.image("home_page.jpeg", width="stretch")
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3, gap="medium")

    with c1:
        st.markdown("""
        <div class="feature-card">
            <div class="feature-icon">📤</div>
            <div class="feature-title">Easy Upload</div>
            <div class="feature-text">Upload a clear JPG, JPEG or PNG leaf image.</div>
        </div>
        """, unsafe_allow_html=True)

    with c2:
        st.markdown("""
        <div class="feature-card">
            <div class="feature-icon">🧠</div>
            <div class="feature-title">CNN Analysis</div>
            <div class="feature-text">The trained model analyzes visual leaf patterns.</div>
        </div>
        """, unsafe_allow_html=True)

    with c3:
        st.markdown("""
        <div class="feature-card">
            <div class="feature-icon">💡</div>
            <div class="feature-title">Useful Results</div>
            <div class="feature-text">Get prediction, confidence, treatment and prevention guidance.</div>
        </div>
        """, unsafe_allow_html=True)

# About Page
elif(app_mode=="About"):
    st.header("About")

    st.subheader("About Dataset")

    st.write(
        "This dataset is recreated using offline augmentation from the original dataset. "
        "The original dataset can be found on this GitHub repository. "
        "The dataset consists of about 87K RGB images of healthy and diseased crop leaves, "
        "categorized into 38 different classes."
    )

    st.subheader("Content")

    st.write("1. Train — 70,295 images")
    st.write("2. Valid — 17,572 images")
    st.write("3. Test — 33 images")

    st.subheader("Future Scope")

    st.markdown("""
    - **More Crop & Disease Classes:** Expand the dataset to cover more crops and diseases.
    - **Mobile Application:** Develop a mobile version for easier field use.
    - **Real-Time Detection:** Enable real-time detection using a phone camera.
    """)

#Prediction Page
# Prediction Page
elif(app_mode=="Disease Recognition"):

    st.markdown("""
    <div class="hero">
        <div class="hero-badge">🔬 CNN IMAGE ANALYSIS</div>
        <div class="page-title" style="font-size:38px;">Plant Disease Recognition</div>
        <div class="page-subtitle">
            Upload a clear leaf image to analyze it and receive an AI-assisted prediction.
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### 📤 Upload a Leaf Image")
    test_image = st.file_uploader(
        "Choose a clear leaf image",
        type=["jpg", "jpeg", "png"],
        label_visibility="collapsed"
    )

    # Class names
    class_name = [
        'Apple___Apple_scab',
        'Apple___Black_rot',
        'Apple___Cedar_apple_rust',
        'Apple___healthy',
        'Blueberry___healthy',
        'Cherry_(including_sour)___Powdery_mildew',
        'Cherry_(including_sour)___healthy',
        'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
        'Corn_(maize)___Common_rust_',
        'Corn_(maize)___Northern_Leaf_Blight',
        'Corn_(maize)___healthy',
        'Grape___Black_rot',
        'Grape___Esca_(Black_Measles)',
        'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
        'Grape___healthy',
        'Orange___Haunglongbing_(Citrus_greening)',
        'Peach___Bacterial_spot',
        'Peach___healthy',
        'Pepper,_bell___Bacterial_spot',
        'Pepper,_bell___healthy',
        'Potato___Early_blight',
        'Potato___Late_blight',
        'Potato___healthy',
        'Raspberry___healthy',
        'Soybean___healthy',
        'Squash___Powdery_mildew',
        'Strawberry___Leaf_scorch',
        'Strawberry___healthy',
        'Tomato___Bacterial_spot',
        'Tomato___Early_blight',
        'Tomato___Late_blight',
        'Tomato___Leaf_Mold',
        'Tomato___Septoria_leaf_spot',
        'Tomato___Spider_mites Two-spotted_spider_mite',
        'Tomato___Target_Spot',
        'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
        'Tomato___Tomato_mosaic_virus',
        'Tomato___healthy'
    ]

    if test_image is not None:

        st.markdown(
            '<div class="info-card"><b>Preview</b><br>'
            'Your uploaded leaf image is ready for analysis.</div>',
            unsafe_allow_html=True
        )

        # Show uploaded image
        st.image(
            test_image,
            caption="Uploaded Leaf Image",
            width="stretch"
        )

        # Prediction button
        if st.button("🔍 Predict Disease"):

            with st.spinner("Analyzing the leaf image..."):

                # Load the working H5 model
                model = tf.keras.models.load_model("trained_model.h5")

                # Prepare image
                image = tf.keras.preprocessing.image.load_img(
                    test_image,
                    target_size=(128, 128)
                )

                input_arr = tf.keras.preprocessing.image.img_to_array(image)

                # Convert image into batch
                input_arr = np.expand_dims(input_arr, axis=0)

                # Model prediction
                prediction = model.predict(input_arr, verbose=0)

                # Get predicted class
                result_index = np.argmax(prediction[0])

                # Get confidence
                confidence = float(prediction[0][result_index]) * 100

                result = class_name[result_index]

                # Separate plant and disease
                if "___" in result:
                    plant_name, disease_name = result.split("___", 1)
                else:
                    plant_name = result
                    disease_name = result

                # Make names easier to read
                plant_name = plant_name.replace("_", " ").replace(",", "")
                disease_name = disease_name.replace("_", " ")

                # Check healthy/diseased
                if "healthy" in result.lower():
                    health_status = "Healthy"
                else:
                    health_status = "Diseased"

                # Treatment and prevention information
                disease_info = {

                    "Apple scab": {
                        "treatment": "Remove infected leaves and fruit. Apply an appropriate fungicide according to local agricultural recommendations.",
                        "prevention": "Remove fallen infected leaves and maintain good airflow around the plant."
                    },

                    "Apple Black rot": {
                        "treatment": "Remove infected fruits and branches and use an appropriate fungicide if recommended.",
                        "prevention": "Keep the orchard clean and remove dead or infected plant material."
                    },

                    "Apple Cedar apple rust": {
                        "treatment": "Remove severely infected leaves and follow recommended fungicide treatment.",
                        "prevention": "Improve airflow and avoid planting near known rust hosts where possible."
                    },

                    "Powdery mildew": {
                        "treatment": "Remove heavily infected leaves and use an appropriate fungicide if necessary.",
                        "prevention": "Provide good air circulation and avoid excessive humidity."
                    },

                    "Common rust": {
                        "treatment": "Remove severely affected leaves and use an appropriate fungicide when recommended.",
                        "prevention": "Maintain proper plant spacing and good field sanitation."
                    },

                    "Northern Leaf Blight": {
                        "treatment": "Remove severely affected plant material and use recommended fungicide where appropriate.",
                        "prevention": "Use resistant varieties, crop rotation, and good field sanitation."
                    },

                    "Cercospora leaf spot Gray leaf spot": {
                        "treatment": "Remove affected leaves and follow recommended fungicide treatment.",
                        "prevention": "Use crop rotation and maintain good field sanitation."
                    },

                    "Black rot": {
                        "treatment": "Remove infected plant parts and use an appropriate fungicide if recommended.",
                        "prevention": "Keep plants dry when possible and remove infected plant material."
                    },

                    "Esca (Black Measles)": {
                        "treatment": "Remove severely affected plant material and consult a local agricultural expert.",
                        "prevention": "Use healthy planting material and maintain good vineyard sanitation."
                    },

                    "Leaf blight (Isariopsis Leaf Spot)": {
                        "treatment": "Remove infected leaves and use an appropriate fungicide if recommended.",
                        "prevention": "Improve airflow and avoid prolonged leaf wetness."
                    },

                    "Haunglongbing (Citrus greening)": {
                        "treatment": "There is no simple cure for infected trees. Remove severely affected trees according to local agricultural guidance and control insect vectors.",
                        "prevention": "Use healthy planting material and manage the insect vectors that spread the disease."
                    },

                    "Bacterial spot": {
                        "treatment": "Remove severely infected leaves and fruits and follow local recommendations for bacterial disease management.",
                        "prevention": "Use clean planting material and avoid unnecessary leaf wetness."
                    },

                    "Early blight": {
                        "treatment": "Remove infected leaves and use an appropriate fungicide when recommended.",
                        "prevention": "Use crop rotation, good spacing, and remove infected plant debris."
                    },

                    "Late blight": {
                        "treatment": "Remove infected plant material and seek prompt disease-management treatment.",
                        "prevention": "Avoid prolonged leaf wetness and maintain good airflow."
                    },

                    "Leaf Mold": {
                        "treatment": "Remove infected leaves and improve ventilation. Use an appropriate fungicide if recommended.",
                        "prevention": "Improve airflow and reduce excessive humidity."
                    },

                    "Septoria leaf spot": {
                        "treatment": "Remove affected leaves and use an appropriate fungicide if recommended.",
                        "prevention": "Avoid overhead watering and remove infected plant debris."
                    },

                    "Spider mites Two-spotted spider mite": {
                        "treatment": "Remove heavily affected leaves and use an appropriate mite-control treatment when necessary.",
                        "prevention": "Monitor plants regularly and reduce dusty or overly dry conditions."
                    },

                    "Target Spot": {
                        "treatment": "Remove infected leaves and use an appropriate fungicide if recommended.",
                        "prevention": "Maintain good airflow and avoid prolonged leaf wetness."
                    },

                    "Tomato Yellow Leaf Curl Virus": {
                        "treatment": "There is no direct cure for infected plants. Remove severely infected plants and control whitefly vectors.",
                        "prevention": "Use healthy seedlings and manage whiteflies to reduce virus spread."
                    },

                    "Tomato mosaic virus": {
                        "treatment": "There is no direct cure. Remove infected plants and sanitize tools.",
                        "prevention": "Use healthy planting material and disinfect tools after handling infected plants."
                    },

                    "Leaf scorch": {
                        "treatment": "Remove severely affected leaves and provide suitable growing conditions.",
                        "prevention": "Maintain proper watering and avoid environmental stress."
                    }
                }

                # Find information
                if health_status == "Healthy":

                    treatment = "No disease treatment is required. Continue normal plant care."

                    prevention = "Maintain proper watering, nutrition, sunlight, and regular monitoring."

                else:

                    # Try exact disease name
                    info = disease_info.get(disease_name)

                    # General advice if exact disease is not listed
                    if info is None:

                        treatment = (
                            "Remove severely affected plant parts and consult a local "
                            "agricultural expert for the appropriate treatment."
                        )

                        prevention = (
                            "Maintain good plant hygiene, proper spacing, adequate "
                            "nutrition, and avoid prolonged leaf wetness."
                        )

                    else:

                        treatment = info["treatment"]
                        prevention = info["prevention"]

                # Display result
                st.markdown("---")
                st.markdown(
                    '<div class="prediction-heading">🌱 Prediction Result</div>',
                    unsafe_allow_html=True
                )

                r1, r2, r3 = st.columns(3, gap="medium")

                with r1:
                    if health_status == "Healthy":
                        st.success("🌿 PLANT CONDITION: HEALTHY")
                    else:
                        st.error("🦠 PLANT CONDITION: DISEASE DETECTED")

                with r2:
                    st.markdown(
                        f'<div class="result-card">'
                        f'<div class="section-label">Plant</div>'
                        f'<div style="font-size:22px;font-weight:800;color:#164d35;">'
                        f'{plant_name}</div></div>',
                        unsafe_allow_html=True
                    )

                with r3:
                    st.markdown(
                        f'<div class="result-card">'
                        f'<div class="section-label">Confidence</div>'
                        f'<div class="confidence-number">{confidence:.2f}%</div>'
                        f'</div>',
                        unsafe_allow_html=True
                    )

                st.progress(min(int(confidence), 100))

                d1, d2 = st.columns(2, gap="medium")

                with d1:
                    st.markdown("### 🦠 Detected Disease")
                    st.markdown(
                        f'<div class="result-card"><b>{disease_name}</b></div>',
                        unsafe_allow_html=True
                    )

                with d2:
                    st.markdown("### 📊 Model Confidence")
                    st.markdown(
                        f'<div class="result-card">'
                        f'The model assigned <b>{confidence:.2f}%</b> confidence to this prediction.'
                        f'</div>',
                        unsafe_allow_html=True
                    )

                g1, g2 = st.columns(2, gap="medium")

                with g1:
                    st.markdown("### 💊 Recommended Treatment")
                    st.info(treatment)

                with g2:
                    st.markdown("### 🛡️ Prevention Tips")
                    st.info(prevention)

                st.caption(
                    "Note: This prediction is generated by a machine-learning model "
                    "and should be used as an initial screening. For serious crop "
                    "problems, consult an agricultural expert."
                )