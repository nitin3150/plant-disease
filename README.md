# 🌿 Plant Disease Detection Using Deep Learning

AI-powered system that detects plant diseases from leaf images with **97% accuracy**. Just upload a photo and get instant diagnosis!

**🌐 Live Demo**: [Your App URL Here]  
**📊 Accuracy**: 97.45%  
**⚡ Speed**: <1 second  

![Demo](demo.gif)

---

## 🎯 What Does This Do?

Upload a photo of a plant leaf → Get instant disease diagnosis with confidence score and treatment suggestions.

**Detects 38 diseases** across 14 plant types:
🍎 Apple | 🍅 Tomato | 🌽 Corn | 🍇 Grape | 🥔 Potato | 🍓 Strawberry | and more!

---

## ✨ Features

- ✅ **97% accurate** disease detection
- ⚡ **Instant results** (<1 second)
- 🔥 **Visual explanation** showing diseased areas
- 💊 **Treatment suggestions** for identified diseases
- 📱 **Works on phone** - take photo and upload
- 🌐 **Free to use** - deployed online

---

## 🚀 Quick Start

### Run the App (5 minutes)

```bash
# 1. Install dependencies
pip install streamlit torch torchvision pillow opencv-python-headless numpy

# 2. Download the trained model
# Get best_model_full.pth from releases
# Put it in checkpoints/ folder

# 3. Run the app
streamlit run app.py

# 4. Opens in browser at: http://localhost:8501
```

That's it! Upload a leaf image and test it.

---

## 📊 Dataset

**PlantVillage Dataset**: 54,303 images of healthy and diseased plant leaves

**14 Plants**: Apple, Blueberry, Cherry, Corn, Grape, Orange, Peach, Pepper, Potato, Raspberry, Soybean, Squash, Strawberry, Tomato

**38 Categories**: 26 diseases + 12 healthy classes

**Download**: [PlantVillage on Kaggle](https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset)

---

## 🏗️ How It Works

### Architecture
```
Input Image (224x224) 
    ↓
EfficientNet-B3 (Pre-trained on ImageNet)
    ↓
Custom Classifier (38 classes)
    ↓
Disease Prediction + Confidence Score
```

### Training
- **Model**: EfficientNet-B3 (12M parameters)
- **Training Time**: 24 hours on GPU
- **Accuracy**: 97.45%
- **Platform**: Google Colab / Kaggle

---

## 📁 Project Files

```
plant-disease-detection/
├── app.py                  # Streamlit web app
├── train.py               # Training script
├── evaluate.py            # Testing & metrics
├── requirements.txt       # Dependencies
├── checkpoints/
│   └── best_model_full.pth   # Trained model (download separately)
└── notebooks/
    ├── colab_training.ipynb  # Train on Google Colab
    └── kaggle_fast_training.ipynb  # Fast 1-hour training
```

---

## 🎓 Train Your Own Model

### Option 1: Fast Training (1 hour, 85-90% accuracy)

Perfect for testing or demos:

1. Open Kaggle: https://www.kaggle.com/code
2. Create new notebook, select **GPU P100**
3. Add PlantVillage dataset (click "+ Add Data")
4. Copy code from `notebooks/kaggle_fast_training.ipynb`
5. Run all cells → Done in 1 hour!

### Option 2: Full Training (24 hours, 97-99% accuracy)

For production quality:

1. Use same Kaggle setup as above
2. Copy code from `notebooks/kaggle_full_training.ipynb`
3. Run all cells → Done in 24 hours
4. Download model from Output tab

---

## 📱 Using the App

### Web Interface

1. **Upload Image**: Click "Browse files" and select a plant leaf photo
2. **Get Prediction**: See disease name and confidence score
3. **View Details**: Check top 5 predictions
4. **See Explanation**: Grad-CAM shows diseased areas in red/yellow
5. **Read Suggestions**: Get treatment recommendations

### Example Results

```
🟢 Tomato Late Blight
Confidence: 94.3% (High Confidence)

Top 5 Predictions:
1. Tomato Late Blight     94.3% ████████████████████
2. Tomato Early Blight     3.2% ██
3. Tomato Septoria Leaf    1.5% █
4. Tomato Bacterial Spot   0.8% 
5. Tomato Healthy          0.2% 
```

---

## 🛠️ Requirements

**Software**:
- Python 3.8+
- PyTorch 2.0+
- Streamlit 1.28+

**Hardware** (for training):
- GPU recommended (Google Colab free tier works!)
- 16GB RAM
- 10GB disk space

**Hardware** (for running app):
- Any computer (CPU is fine)
- 4GB RAM
- Works on M1 Mac, Windows, Linux

---

## 🎨 Customization

### Change Model Path

In `app.py`, update sidebar:
```python
model_path = st.text_input(
    "Model Path",
    value="checkpoints/your_model.pth"  # Change this
)
```

### Adjust Confidence Threshold

```python
confidence_threshold = st.slider(
    "Confidence Threshold",
    value=0.7  # Change this (0.0 to 1.0)
)
```

---

## 🐛 Troubleshooting

**App won't start?**
```bash
pip install --upgrade streamlit torch torchvision
```

**Model not found?**
```bash
# Check model is in right place
ls checkpoints/

# Update path in app if needed
```

**Low confidence predictions?**
- Use clear, well-lit images
- Single leaf works best
- Avoid blurry photos
- Try images from PlantVillage test set

**Slow on M1 Mac?**
- Select "mps" device in sidebar (5x faster!)

---

## 🔮 Future Improvements

- [ ] Add 100+ more disease classes (expand to 271 classes)
- [ ] Train on real-world field images (better smartphone photo performance)
- [ ] Disease severity assessment (Mild/Moderate/Severe)
- [ ] Multi-disease detection per image
- [ ] Mobile app (iOS/Android)
- [ ] Treatment database integration
- [ ] Multi-language support

---

## 📚 References

1. **Dataset**: Hughes, D. P., & Salathe, M. (2015). PlantVillage Dataset. arXiv:1511.08060
2. **EfficientNet**: Tan, M., & Le, Q. (2019). EfficientNet: Rethinking Model Scaling for CNNs. arXiv:1905.11946
3. **Grad-CAM**: Selvaraju et al. (2017). Grad-CAM: Visual Explanations from Deep Networks. arXiv:1610.02391

---

## 👨‍💻 Author

**Nitin** - Master of Science in Applied Machine Intelligence, Northeastern University

📧 Contact: [Your Email]  
🔗 LinkedIn: [Your LinkedIn]  
💼 GitHub: [Your GitHub]  

---

## 📄 License

This project is licensed under the MIT License - free to use for research and commercial purposes.

---

## 🙏 Acknowledgments

- PlantVillage dataset creators
- PyTorch and torchvision teams
- Streamlit for easy deployment
- Northeastern University

---

## 🌟 Star This Project!

If you found this helpful, please ⭐ star the repository!

---

**Built with ❤️ for sustainable agriculture and AI research**
