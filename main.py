import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import numpy as np
import cv2
import json
from pathlib import Path
import time

# Page config - MUST BE FIRST
st.set_page_config(
    page_title="🌿 Plant Disease Detection",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #2E7D32;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #558B2F;
        text-align: center;
        margin-bottom: 2rem;
    }
    .prediction-box {
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
    .high-confidence {
        background-color: #C8E6C9;
        border: 2px solid #4CAF50;
    }
    .medium-confidence {
        background-color: #FFF9C4;
        border: 2px solid #FFC107;
    }
    .low-confidence {
        background-color: #FFCDD2;
        border: 2px solid #F44336;
    }
    .stProgress > div > div > div > div {
        background-color: #4CAF50;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model(checkpoint_path, device='cpu'):
    """Load trained model - cached for performance"""
    try:
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        
        model_name = checkpoint.get('model_name', 'mobilenet_v2')
        num_classes = checkpoint.get('num_classes', 38)
        
        # Create model architecture
        if 'efficientnet' in model_name.lower():
            if 'b3' in model_name.lower():
                model = models.efficientnet_b3()
                num_ftrs = model.classifier[1].in_features
            else:
                model = models.efficientnet_b0()
                num_ftrs = model.classifier[1].in_features
            model.classifier[1] = nn.Linear(num_ftrs, num_classes)
            
        elif 'mobilenet' in model_name.lower():
            model = models.mobilenet_v2()
            num_ftrs = model.classifier[1].in_features
            model.classifier[1] = nn.Linear(num_ftrs, num_classes)
            
        elif 'resnet' in model_name.lower():
            model = models.resnet50()
            num_ftrs = model.fc.in_features
            model.fc = nn.Linear(num_ftrs, num_classes)
        else:
            st.error(f"Unknown model architecture: {model_name}")
            return None, None, None
        
        # Load weights
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        model.eval()
        
        class_names = checkpoint.get('class_names', [f"Class_{i}" for i in range(num_classes)])
        val_acc = checkpoint.get('val_acc', 0)
        
        return model, class_names, val_acc
    
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None, None, None

def get_image_transform():
    """Get image preprocessing transform"""
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

def preprocess_image(image):
    """Preprocess uploaded image"""
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    transform = get_image_transform()
    img_tensor = transform(image).unsqueeze(0)
    
    return img_tensor

def predict(model, image_tensor, device, top_k=5):
    """Make prediction"""
    with torch.no_grad():
        image_tensor = image_tensor.to(device)
        outputs = model(image_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        
        top_probs, top_indices = probabilities.topk(top_k)
        
        return top_probs[0].cpu().numpy(), top_indices[0].cpu().numpy()

def format_class_name(class_name):
    """Format class name for display"""
    # Remove underscores and format
    name = class_name.replace('___', ' - ').replace('_', ' ')
    return name.title()

def get_confidence_info(confidence):
    """Get confidence level info"""
    if confidence >= 0.9:
        return "🟢", "High Confidence", "high-confidence", "#4CAF50"
    elif confidence >= 0.7:
        return "🟡", "Moderate Confidence", "medium-confidence", "#FFC107"
    else:
        return "🔴", "Low Confidence", "low-confidence", "#F44336"

def generate_gradcam(model, image_tensor, target_class, device, model_name):
    """Generate Grad-CAM heatmap (simplified version)"""
    try:
        # Get target layer
        if 'efficientnet' in model_name.lower():
            target_layer = model.features[-1]
        elif 'mobilenet' in model_name.lower():
            target_layer = model.features[-1]
        elif 'resnet' in model_name.lower():
            target_layer = model.layer4[-1]
        else:
            return None
        
        gradients = []
        activations = []
        
        def forward_hook(module, input, output):
            activations.append(output)
        
        def backward_hook(module, grad_input, grad_output):
            gradients.append(grad_output[0])
        
        forward_handle = target_layer.register_forward_hook(forward_hook)
        backward_handle = target_layer.register_full_backward_hook(backward_hook)
        
        # Forward pass
        image_tensor = image_tensor.to(device)
        image_tensor.requires_grad = True
        output = model(image_tensor)
        
        # Backward pass
        model.zero_grad()
        output[0, target_class].backward()
        
        # Remove hooks
        forward_handle.remove()
        backward_handle.remove()
        
        # Generate CAM
        gradient = gradients[0].cpu().data.numpy()[0]
        activation = activations[0].cpu().data.numpy()[0]
        
        weights = np.mean(gradient, axis=(1, 2))
        cam = np.sum(weights[:, np.newaxis, np.newaxis] * activation, axis=0)
        
        # Normalize
        cam = np.maximum(cam, 0)
        cam = cam / (cam.max() + 1e-8)
        
        return cam
    
    except Exception as e:
        st.warning(f"Grad-CAM generation failed: {str(e)}")
        return None

def create_heatmap_overlay(original_image, cam):
    """Create heatmap overlay on original image"""
    # Resize CAM to image size
    cam_resized = cv2.resize(cam, (original_image.width, original_image.height))
    
    # Create heatmap
    heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    
    # Overlay
    original_array = np.array(original_image)
    overlay = cv2.addWeighted(original_array, 0.6, heatmap, 0.4, 0)
    
    return Image.fromarray(overlay), Image.fromarray(heatmap)

def main():
    # Header
    st.markdown('<h1 class="main-header">🌿 Plant Disease Detection System</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">AI-powered disease identification from leaf images using Deep Learning</p>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Model path
        model_path = st.text_input(
            "Model Path",
            value="checkpoints/best_model_fast.pth",
            help="Path to trained model checkpoint"
        )
        
        # Device selection
        device_options = ['cpu']
        if torch.cuda.is_available():
            device_options.insert(0, 'cuda')
        if torch.backends.mps.is_available():
            device_options.insert(1, 'mps')
        
        device = st.selectbox("Device", device_options, help="Processing device")
        
        # Confidence threshold
        confidence_threshold = st.slider(
            "Confidence Threshold",
            min_value=0.0,
            max_value=1.0,
            value=0.7,
            step=0.05,
            help="Minimum confidence for reliable prediction"
        )
        
        # Show Grad-CAM
        show_gradcam = st.checkbox("Show Grad-CAM Visualization", value=True)
        
        # Show top-k
        top_k = st.slider("Show Top-K Predictions", 1, 10, 5)
        
        st.markdown("---")
        
        # About
        st.markdown("### 📊 Model Info")
        
        # Try to load model info
        if Path(model_path).exists():
            try:
                checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
                st.info(f"""
                **Model:** {checkpoint.get('model_name', 'Unknown')}  
                **Classes:** {checkpoint.get('num_classes', 'Unknown')}  
                **Accuracy:** {checkpoint.get('val_acc', 0):.2f}%
                """)
            except:
                st.warning("Could not load model info")
        
        st.markdown("---")
        
        st.markdown("""
        ### 💡 Tips for Best Results
        - Use clear, well-lit images
        - Focus on the diseased area
        - Avoid blurry photos
        - Single leaf works best
        """)
    
    # Main content
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("### 📤 Upload Leaf Image")
        
        uploaded_file = st.file_uploader(
            "Choose an image...",
            type=['jpg', 'jpeg', 'png'],
            help="Upload a clear image of a plant leaf"
        )
        
        if uploaded_file is not None:
            # Display uploaded image
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_container_width=True)
            
            # Image info
            st.caption(f"Image size: {image.width}×{image.height} pixels")
    
    with col2:
        st.markdown("### 🔍 Prediction Results")
        
        if uploaded_file is not None:
            # Load model
            if not Path(model_path).exists():
                st.error(f"❌ Model not found at: {model_path}")
                st.info("Please train the model first or update the model path in the sidebar.")
                return
            
            with st.spinner("Loading model..."):
                model, class_names, val_acc = load_model(model_path, device)
            
            if model is None:
                st.error("Failed to load model. Please check the model file.")
                return
            
            st.success("✓ Model loaded successfully")
            
            # Make prediction
            with st.spinner("Analyzing image..."):
                start_time = time.time()
                image_tensor = preprocess_image(image)
                probabilities, indices = predict(model, image_tensor, device, top_k)
                inference_time = time.time() - start_time
            
            # Top prediction
            top_prob = probabilities[0]
            top_class = class_names[indices[0]]
            top_class_formatted = format_class_name(top_class)
            
            emoji, conf_level, conf_class, conf_color = get_confidence_info(top_prob)
            
            # Display prediction
            st.markdown(f"""
            <div class="prediction-box {conf_class}">
                <h2 style="margin:0; color: #1B5E20;">{emoji} {top_class_formatted}</h2>
                <p style="font-size: 1.5rem; margin: 10px 0; color: {conf_color};">
                    <strong>Confidence: {top_prob*100:.1f}%</strong>
                </p>
                <p style="margin:0; color: #424242;">
                    {conf_level} | Inference: {inference_time*1000:.0f}ms
                </p>
            </div>
            """, unsafe_allow_html=True)
            
            # Confidence-based message
            if top_prob >= confidence_threshold:
                if 'healthy' in top_class.lower():
                    st.success("✅ **Plant appears to be healthy!** No disease detected.")
                else:
                    st.warning("⚠️ **Disease detected!** Consider treatment or consult an expert.")
                    
                    # Add treatment suggestion
                    with st.expander("💊 Treatment Suggestions"):
                        st.info("""
                        **General Recommendations:**
                        - Remove infected leaves immediately
                        - Improve air circulation around plants
                        - Avoid overhead watering
                        - Apply appropriate fungicide/treatment
                        - Consult local agricultural extension for specific advice
                        
                        **Note:** Specific treatment depends on disease type and severity.
                        """)
            else:
                st.error(f"""
                🔍 **Low Confidence Prediction**
                
                The model is uncertain about this diagnosis (confidence: {top_prob*100:.1f}%).
                This may be due to:
                - Poor image quality or lighting
                - Disease not in training dataset
                - Unusual leaf condition
                
                **Recommendation:** Try taking another photo with better lighting and focus.
                """)
            
            # Top-K predictions
            st.markdown("---")
            st.markdown(f"#### 📊 Top {top_k} Predictions")
            
            for i in range(min(top_k, len(probabilities))):
                prob = probabilities[i]
                class_name = format_class_name(class_names[indices[i]])
                
                col_a, col_b = st.columns([3, 1])
                with col_a:
                    st.markdown(f"**{i+1}. {class_name}**")
                with col_b:
                    st.markdown(f"**{prob*100:.1f}%**")
                
                st.progress(float(prob))
        
        else:
            # Placeholder
            st.info("""
            👆 **Upload an image to get started**
            
            Upload a clear photo of a plant leaf to receive:
            - Disease classification
            - Confidence score
            - Treatment recommendations
            - Visual explanation (Grad-CAM)
            """)
    
    # Grad-CAM visualization
    if uploaded_file is not None and show_gradcam and model is not None:
        st.markdown("---")
        st.markdown("### 🔥 Grad-CAM Visualization")
        st.markdown("*Heatmap showing which regions influenced the prediction*")
        
        with st.spinner("Generating Grad-CAM..."):
            try:
                cam = generate_gradcam(
                    model, 
                    image_tensor, 
                    indices[0], 
                    device,
                    checkpoint.get('model_name', 'mobilenet_v2')
                )
                
                if cam is not None:
                    overlay, heatmap = create_heatmap_overlay(image, cam)
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.markdown("**Original**")
                        st.image(image, use_container_width=True)
                    
                    with col2:
                        st.markdown("**Heatmap**")
                        st.image(heatmap, use_container_width=True)
                    
                    with col3:
                        st.markdown("**Overlay**")
                        st.image(overlay, use_container_width=True)
                    
                    st.info("""
                    🔥 **How to interpret:** 
                    - Red/Yellow areas: High influence on prediction
                    - Blue/Purple areas: Low influence
                    - Model focuses on diseased regions for accurate diagnosis
                    """)
                else:
                    st.warning("Grad-CAM visualization not available for this model")
            
            except Exception as e:
                st.error(f"Error generating Grad-CAM: {str(e)}")
    
    # Footer
    # st.markdown("---")
    # st.markdown("""
    # <div style="text-align: center; color: #757575; padding: 20px;">
    #     <p>🌿 <strong>Plant Disease Detection System</strong> | Built with PyTorch & Streamlit</p>
    #     <p>Model trained on PlantVillage Dataset | 38 disease classes | 95%+ accuracy</p>
    #     <p style="font-size: 0.8rem;">⚠️ This tool is for educational purposes. Always consult agricultural experts for critical decisions.</p>
    # </div>
    # """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()