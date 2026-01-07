"""
TrueSight - AI Image Detection Platform
Single-File Standalone Version (No import issues)

Run with: streamlit run truesight_standalone.py
"""

import streamlit as st
import time
import io
import re
import tempfile
import numpy as np
from pathlib import Path
from PIL import Image, ImageOps, ImageChops
from scipy.ndimage import median_filter, maximum_filter

# ============================================================================
# DEPENDENCIES CHECK
# ============================================================================
try:
    import pillow_heif
    pillow_heif.register_heif_opener()
    HEIC_AVAILABLE = True
except ImportError:
    HEIC_AVAILABLE = False

try:
    from transformers import pipeline
    TRANSFORMERS_AVAILABLE = True
    classifier = pipeline("image-classification", model="dima806/ai_vs_real_image_detection")
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    classifier = None

# ============================================================================
# IMAGE UTILITIES
# ============================================================================

def validate_image_format(uploaded_file):
    """Validates image format"""
    allowed_extensions = {'.jpg', '.jpeg', '.png', '.heic', '.heif'}
    file_ext = Path(uploaded_file.name).suffix.lower()
    
    if file_ext not in allowed_extensions:
        return False, f"Unsupported format '{file_ext}'. Please upload JPG, PNG, or HEIC."
    
    if file_ext in ['.heic', '.heif'] and not HEIC_AVAILABLE:
        return False, "HEIC format detected but pillow-heif is not installed."
    
    return True, ""


def convert_heic_to_pil(uploaded_file):
    """Converts HEIC to PIL Image"""
    try:
        heic_bytes = uploaded_file.read()
        image = Image.open(io.BytesIO(heic_bytes))
        image = ImageOps.exif_transpose(image)
        
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        return image
    except Exception as e:
        st.error(f"HEIC Conversion Error: {e}")
        return None


def load_and_prepare_image(uploaded_file):
    """Loads standard formats"""
    try:
        image_bytes = uploaded_file.read()
        image = Image.open(io.BytesIO(image_bytes))
        image = ImageOps.exif_transpose(image)
        
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        return image
    except Exception as e:
        st.error(f"Image Load Error: {e}")
        return None


def detect_snapchat_origin(uploaded_file, image):
    """Detects Snapchat images"""
    filename = uploaded_file.name.lower()
    
    # Layer 1: Filename patterns
    snapchat_patterns = [
        'snapchat', 'snap_', 'snap-', '_snap', '-snap',
        'sc_', 'sc-', '_sc_', r'snap\d+', r'sc\d+', r'memories_\d+'
    ]
    
    for pattern in snapchat_patterns:
        if re.search(pattern, filename):
            return True, f"Filename pattern detected: '{pattern}'"
    
    # Layer 2: EXIF metadata
    try:
        exif_data = image.getexif()
        
        if exif_data:
            software = exif_data.get(305, "").lower()
            make = exif_data.get(271, "").lower()
            model = exif_data.get(272, "").lower()
            
            snapchat_identifiers = ['snapchat', 'snap inc', 'snap camera']
            
            if any(identifier in software for identifier in snapchat_identifiers):
                return True, f"EXIF Software tag: '{software}'"
            
            if 'snap' in make or 'spectacles' in model:
                return True, f"Snapchat hardware detected: '{model}'"
    except:
        pass
    
    return False, ""

# ============================================================================
# MODEL INFERENCE
# ============================================================================

def analyze_ela(image):
    """Error Level Analysis"""
    try:
        if image.width > 1024:
            img = image.resize((1024, 1024), Image.Resampling.LANCZOS)
        else:
            img = image
        
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=True) as tmp:
            img.save(tmp.name, "JPEG", quality=90)
            resaved = Image.open(tmp.name).convert('RGB')
            ela_img = ImageChops.difference(img, resaved)
            ela_gray = np.array(ela_img.convert('L'))
            return float(np.std(ela_gray))
    except:
        return 50.0


def analyze_noise(image):
    """Noise variance analysis"""
    try:
        img = image.resize((512, 512), Image.Resampling.LANCZOS)
        gray = np.array(img.convert('L'))
        denoised = median_filter(gray, size=3)
        noise_map = gray.astype(float) - denoised.astype(float)
        return float(np.var(noise_map))
    except:
        return 10.0


def analyze_frequency(image):
    """Frequency domain analysis"""
    try:
        gray = image.resize((512, 512), Image.Resampling.LANCZOS).convert('L')
        f = np.fft.fft2(gray)
        fshift = np.fft.fftshift(f)
        magnitude = 20 * np.log(np.abs(fshift) + 1)
        
        rows, cols = 512, 512
        crow, ccol = rows // 2, cols // 2
        magnitude[crow-5:crow+5, ccol-5:ccol+5] = 0
        
        local_max = maximum_filter(magnitude, size=10)
        peaks = (magnitude == local_max)
        threshold = np.mean(magnitude) + (4.0 * np.std(magnitude))
        num_peaks = np.sum(peaks & (magnitude > threshold))
        
        return num_peaks > 20
    except:
        return False


def check_camera_metadata(image):
    """Checks for camera metadata"""
    try:
        exif = image.getexif()
        if not exif:
            return False
        
        make = exif.get(271, "")
        model = exif.get(272, "")
        
        camera_brands = [
            'canon', 'nikon', 'sony', 'apple', 'samsung',
            'google', 'pixel', 'iphone', 'galaxy'
        ]
        
        make_model = f"{make} {model}".lower()
        return any(brand in make_model for brand in camera_brands)
    except:
        return False


def predict_image(image):
    """Main prediction function"""
    
    # Layer 1: Neural Network
    nn_score = 50.0
    
    if TRANSFORMERS_AVAILABLE and classifier:
        try:
            predictions = classifier(image)
            for pred in predictions:
                label = pred['label'].lower()
                score = pred['score']
                
                if label in ['fake', 'ai', 'artificial']:
                    nn_score = score * 100
                elif label in ['real', 'human']:
                    nn_score = (1 - score) * 100
        except:
            pass
    
    # Layer 2: Forensics
    ela_std = analyze_ela(image)
    noise_var = analyze_noise(image)
    has_ai_grid = analyze_frequency(image)
    has_camera_metadata = check_camera_metadata(image)
    
    # Layer 3: Fusion Logic
    final_score = nn_score
    
    if has_camera_metadata and not has_ai_grid:
        final_score = min(final_score, 25)
    
    if has_ai_grid and ela_std < 15:
        final_score = max(final_score, 90)
    
    if noise_var < 2.0 and not has_camera_metadata:
        final_score += 15
    
    final_score = max(0, min(100, final_score))
    
    # Generate verdict
    if final_score > 65:
        prediction = "AI Generated"
    else:
        prediction = "Real"
    
    return prediction, final_score

# ============================================================================
# STREAMLIT APP
# ============================================================================

st.set_page_config(
    page_title="TrueSight - AI Image Detection",
    page_icon="🔍",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Custom CSS
st.markdown("""
<style>
    .stApp {
        background: linear-gradient(135deg, #0a0e27 0%, #1a1f3a 100%);
        color: #e0e0e0;
    }
    
    .hero-container {
        text-align: center;
        padding: 2rem 0 3rem 0;
        background: linear-gradient(180deg, rgba(26, 31, 58, 0.8) 0%, rgba(10, 14, 39, 0) 100%);
        border-radius: 12px;
        margin-bottom: 2rem;
    }
    
    .hero-title {
        font-size: 3.5rem;
        font-weight: 800;
        background: linear-gradient(135deg, #00d4ff 0%, #0099ff 50%, #0066ff 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    
    .hero-subtitle {
        font-size: 1.2rem;
        color: #a0aec0;
        font-weight: 300;
    }
    
    .verdict-card {
        padding: 2rem;
        border-radius: 16px;
        margin: 2rem 0;
        text-align: center;
        border: 2px solid;
    }
    
    .verdict-ai {
        background: linear-gradient(135deg, rgba(220, 38, 38, 0.15) 0%, rgba(185, 28, 28, 0.1) 100%);
        border-color: #ef4444;
    }
    
    .verdict-real {
        background: linear-gradient(135deg, rgba(34, 197, 94, 0.15) 0%, rgba(22, 163, 74, 0.1) 100%);
        border-color: #22c55e;
    }
    
    .verdict-title {
        font-size: 2.5rem;
        font-weight: 800;
        margin-bottom: 1rem;
    }
    
    .warning-box {
        background: linear-gradient(135deg, rgba(245, 158, 11, 0.15) 0%, rgba(217, 119, 6, 0.1) 100%);
        border: 2px solid #f59e0b;
        border-radius: 16px;
        padding: 2rem;
        margin: 2rem 0;
        text-align: center;
    }
    
    .warning-title {
        font-size: 1.5rem;
        font-weight: 700;
        color: #fbbf24;
    }
    
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# Hero Section
st.markdown("""
<div class="hero-container">
    <div class="hero-title">🔍 TrueSight</div>
    <div class="hero-subtitle">AI Image Authenticity Verification</div>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div style="text-align: center; margin: 2rem 0 1rem 0;">
    <h3 style="color: #00d4ff;">Upload Image for Analysis</h3>
    <p style="color: #a0aec0;">Supported: JPG, PNG, HEIC</p>
</div>
""", unsafe_allow_html=True)

# Upload Widget
uploaded_file = st.file_uploader(
    "Choose an image file",
    type=['jpg', 'jpeg', 'png', 'heic', 'heif'],
    label_visibility="collapsed"
)

# Processing Pipeline
if uploaded_file is not None:
    # Step 1: Validate
    is_valid, error_msg = validate_image_format(uploaded_file)
    
    if not is_valid:
        st.error(f"❌ {error_msg}")
    else:
        # Step 2: Convert HEIC if needed
        file_ext = Path(uploaded_file.name).suffix.lower()
        
        if file_ext in ['.heic', '.heif']:
            with st.spinner('Converting HEIC format...'):
                image = convert_heic_to_pil(uploaded_file)
        else:
            image = load_and_prepare_image(uploaded_file)
        
        if image is None:
            st.error("❌ Failed to load image.")
        else:
            # Step 3: Display image
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                st.image(image, caption="Uploaded Image", use_container_width=True)
            
            # Step 4: Check Snapchat
            is_snapchat, snapchat_reason = detect_snapchat_origin(uploaded_file, image)
            
            if is_snapchat:
                st.markdown(f"""
                <div class="warning-box">
                    <div style="font-size: 3rem;">🔒</div>
                    <div class="warning-title">Snapchat Images Not Supported</div>
                    <p style="color: #d1d5db; margin-top: 1rem;">
                        This image appears to be from Snapchat.<br>
                        Please upload an original image or a non-Snapchat source.<br><br>
                        <strong>Reason:</strong> {snapchat_reason}
                    </p>
                </div>
                """, unsafe_allow_html=True)
            else:
                # Step 5: Run Inference
                st.markdown("---")
                
                with st.spinner('🔬 Analyzing image authenticity...'):
                    progress_bar = st.progress(0)
                    for percent in range(100):
                        time.sleep(0.01)
                        progress_bar.progress(percent + 1)
                    
                    prediction, confidence = predict_image(image)
                    progress_bar.empty()
                
                # Step 6: Display Results
                is_ai = prediction.lower() in ['ai generated', 'fake', 'ai']
                
                if is_ai:
                    st.markdown(f"""
                    <div class="verdict-card verdict-ai">
                        <div class="verdict-title" style="color: #ef4444;">🔴 AI-GENERATED IMAGE</div>
                        <div style="font-size: 1.8rem; color: #ef4444;">{confidence:.1f}% Confidence</div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    st.progress(confidence / 100)
                    
                    st.info("""
                    **Analysis Summary:**  
                    This image exhibits visual patterns and frequency artifacts commonly observed in AI-generated content.
                    """)
                else:
                    st.markdown(f"""
                    <div class="verdict-card verdict-real">
                        <div class="verdict-title" style="color: #22c55e;">🟢 AUTHENTIC IMAGE</div>
                        <div style="font-size: 1.8rem; color: #22c55e;">{confidence:.1f}% Confidence</div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    st.progress(confidence / 100)
                    
                    st.success("""
                    **Analysis Summary:**  
                    This image demonstrates characteristics consistent with authentic photography.
                    """)
                
                # Reset button
                st.markdown("<br>", unsafe_allow_html=True)
                col1, col2, col3 = st.columns([1, 1, 1])
                with col2:
                    if st.button("🔄 Analyze Another Image", use_container_width=True):
                        st.rerun()

# Footer
st.markdown("""
<div style="margin-top: 4rem; padding: 2rem; text-align: center; background: rgba(10, 14, 39, 0.5); border-radius: 12px;">
    <p style="color: #718096; font-size: 0.85rem;">
        <strong>⚠️ Platform Restrictions:</strong> Snapchat images not supported due to heavy post-processing<br><br>
        TrueSight v1.0 | Powered by Advanced Computer Vision & Neural Networks
    </p>
</div>
""", unsafe_allow_html=True)
