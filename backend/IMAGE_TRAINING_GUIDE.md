# 📸 NeRF Training Guide: Images & Training

## 🎯 **Quick Start: Upload Images & Train**

### **Step 1: Create a Project**
```bash
curl -X POST http://localhost:8000/api/v1/projects \
  -H "Content-Type: application/json" \
  -d '{"name": "My NeRF Scene", "description": "3D reconstruction project"}'
```

### **Step 2: Upload Images**
```bash
# Replace PROJECT_ID with your actual project ID
PROJECT_ID="your-project-id-here"

# Upload multiple images
curl -X POST http://localhost:8000/api/v1/projects/$PROJECT_ID/upload_images \
  -F "files=@image1.jpg" \
  -F "files=@image2.jpg" \
  -F "files=@image3.jpg"
```

### **Step 3: Start Training**
```bash
curl -X POST http://localhost:8000/api/v1/projects/$PROJECT_ID/start_training \
  -H "Content-Type: application/json" \
  -d '{
    "num_epochs": 1000,
    "learning_rate": 0.001,
    "batch_size": 1024,
    "img_wh": [400, 400]
  }'
```

## 📸 **Where to Get Images**

### **1. Free Sample Datasets**
- **NeRF Official Datasets**: https://drive.google.com/drive/folders/128yBriW1IG_3NJ5Rp7APSTZsJqdJdfc1
  - `lego`, `drums`, `ficus`, `materials`, `mic`, `ship`, `chair`, `hotdog`
- **LLFF Dataset**: https://github.com/Fyusion/LLFF
  - Real-world scenes with camera poses
- **Tanks & Temples**: https://www.tanksandtemples.org/
  - Large-scale outdoor scenes

### **2. Capture Your Own Images**
**Requirements:**
- 20-50 photos minimum
- Object/scene in center of frame
- Vary camera angles (not just horizontal rotation)
- Good lighting, avoid shadows
- Sharp images (no motion blur)
- 30-50% overlap between consecutive images

**Camera Tips:**
- Use phone camera or DSLR
- Manual focus if possible
- Consistent exposure settings
- Take photos in a circle around the object
- Include some overhead and low-angle shots

### **3. Download Sample Images**
```bash
# Create a sample images directory
mkdir -p sample_images
cd sample_images

# Download some sample images (you'll need to replace with actual URLs)
# Example: Download from Unsplash or other free sources
curl -o image1.jpg "https://images.unsplash.com/photo-..."
curl -o image2.jpg "https://images.unsplash.com/photo-..."
# ... repeat for more images
```

## 🚀 **Complete Training Workflow**

### **1. Prepare Your Images**
```bash
# Create a directory for your images
mkdir my_nerf_images
# Copy your images here
cp /path/to/your/images/*.jpg my_nerf_images/
```

### **2. Create Project & Upload**
```bash
# Create project
PROJECT_RESPONSE=$(curl -s -X POST http://localhost:8000/api/v1/projects \
  -H "Content-Type: application/json" \
  -d '{"name": "My Scene", "description": "3D reconstruction"}')

# Extract project ID
PROJECT_ID=$(echo $PROJECT_RESPONSE | jq -r '.id')
echo "Project ID: $PROJECT_ID"

# Upload images
curl -X POST http://localhost:8000/api/v1/projects/$PROJECT_ID/upload_images \
  -F "files=@my_nerf_images/image1.jpg" \
  -F "files=@my_nerf_images/image2.jpg" \
  -F "files=@my_nerf_images/image3.jpg"
```

### **3. Estimate Camera Poses (if needed)**
```bash
# If you don't have camera poses, estimate them
curl -X POST http://localhost:8000/api/v1/projects/$PROJECT_ID/estimate_poses
```

### **4. Start Training**
```bash
curl -X POST http://localhost:8000/api/v1/projects/$PROJECT_ID/start_training \
  -H "Content-Type: application/json" \
  -d '{
    "num_epochs": 2000,
    "learning_rate": 0.001,
    "batch_size": 1024,
    "img_wh": [400, 400],
    "near": 2.0,
    "far": 6.0,
    "pos_freq_bands": 10,
    "view_freq_bands": 4,
    "hidden_dim": 256,
    "num_layers": 8
  }'
```

### **5. Monitor Training**
```bash
# Get training jobs for the project
curl http://localhost:8000/api/v1/projects/$PROJECT_ID/jobs

# Get specific job status (replace JOB_ID)
curl http://localhost:8000/api/v1/jobs/JOB_ID
```

### **6. Render Novel Views**
```bash
# Render a new view once training is complete
curl -X POST http://localhost:8000/api/v1/projects/$PROJECT_ID/render \
  -H "Content-Type: application/json" \
  -d '{
    "pose": [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 2.5, 0, 0, 0, 1],
    "resolution": 400
  }'
```

## 📊 **Training Parameters Explained**

| Parameter | Description | Recommended Value |
|-----------|-------------|-------------------|
| `num_epochs` | Training iterations | 1000-5000 |
| `learning_rate` | Learning rate | 0.001 |
| `batch_size` | Rays per batch | 1024-4096 |
| `img_wh` | Image resolution | [400, 400] |
| `near/far` | Scene bounds | 2.0/6.0 |
| `pos_freq_bands` | Positional encoding | 10 |
| `view_freq_bands` | View encoding | 4 |
| `hidden_dim` | Network width | 256 |
| `num_layers` | Network depth | 8 |

## 🎨 **Image Requirements**

### **Best Practices:**
- **Resolution**: 400x400 to 800x800 pixels
- **Format**: JPG or PNG
- **Count**: 20-100 images
- **Coverage**: 360° around object/scene
- **Lighting**: Consistent, avoid harsh shadows
- **Focus**: Sharp images, no motion blur

### **Avoid:**
- ❌ Blurry images
- ❌ Inconsistent lighting
- ❌ Too few images (< 20)
- ❌ Only horizontal rotation
- ❌ Moving objects in scene
- ❌ Reflective surfaces (if possible)

## 🔧 **Troubleshooting**

### **Common Issues:**
1. **"No images found"**: Make sure images are uploaded to the project
2. **"Pose estimation failed"**: Try with more images or better coverage
3. **"Training failed"**: Check GPU memory, reduce batch size
4. **"Poor quality results"**: Use more images, better lighting, sharper photos

### **Performance Tips:**
- Use GPU if available
- Start with smaller resolution (400x400)
- Increase batch size if you have more GPU memory
- Use more images for better quality

## 📱 **Using the Web Interface**

1. **Open**: http://localhost:8000/docs
2. **Create Project**: Use the `/projects` POST endpoint
3. **Upload Images**: Use the `/projects/{id}/upload_images` endpoint
4. **Start Training**: Use the `/projects/{id}/start_training` endpoint
5. **Monitor**: Check job status and metrics

## 🎯 **Next Steps**

1. **Get some sample images** from the provided sources
2. **Create a test project** using the API
3. **Upload your images** 
4. **Start training** with default parameters
5. **Monitor progress** and adjust parameters as needed
6. **Render novel views** once training completes

**Happy NeRF training! 🚀** 