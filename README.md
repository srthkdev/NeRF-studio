# NeRF Studio

A production-grade Neural Radiance Fields platform that enables users to upload photos and generate interactive 3D scene reconstructions.

## 🚀 Features

### Core NeRF Implementation
- **Real NeRF Training**: Complete PyTorch-based NeRF implementation with hierarchical sampling
- **Volume Rendering**: Advanced volume rendering engine with ray marching and density integration
- **Positional Encoding**: Fourier feature positional encoding for high-frequency details
- **Multi-scale Training**: Coarse and fine network training for efficient sampling

### Training Pipeline
- **Real-time Training**: Live training progress with WebSocket streaming
- **Metrics Tracking**: Comprehensive loss, PSNR, and performance metrics
- **Checkpointing**: Automatic model checkpointing and recovery
- **Job Management**: Background training jobs with status tracking
- **Training Controls**: Start, stop, pause, and resume training operations

### Advanced 3D Visualization
- **Interactive 3D Viewer**: Three.js-based 3D scene visualization
- **Advanced NeRF Viewer**: WebGL volume rendering with real-time NeRF visualization
- **Level-of-Detail System**: Adaptive quality rendering for performance optimization
- **Camera Frustum Display**: Visual representation of camera positions and orientations
- **Coordinate Axes**: 3D coordinate system visualization
- **Real-time Updates**: Live updates during training and pose estimation
- **Quality Controls**: Adjustable rendering quality and sampling parameters

### Advanced Mesh Extraction & Export
- **Multi-format Export**: Support for GLTF, OBJ, PLY, USD, FBX, STL formats
- **USD Format Support**: Pixar USD format for professional workflows
- **Texture Baking**: Advanced texture generation from neural radiance fields
- **Mesh Optimization**: Automatic mesh decimation and optimization
- **Batch Export**: Export multiple formats simultaneously
- **Export Progress Tracking**: Real-time export progress with detailed status
- **Compression Support**: Automatic file compression for efficient storage
- **Quality Settings**: Configurable export quality levels (low, medium, high)
- **Download Management**: Direct download links for exported files

### Camera Pose Management
- **COLMAP Integration**: Automatic camera pose estimation from image collections
- **Manual Pose Upload**: Support for custom camera pose files
- **Pose Validation**: Automatic validation of camera pose consistency
- **Circular Path Generation**: Automatic generation of circular camera paths
- **Pose Visualization**: 3D visualization of camera positions and orientations

### Fast Inference
- **Novel View Synthesis**: Real-time rendering of novel viewpoints
- **View Frustum Culling**: Efficient rendering with view frustum optimization
- **Adaptive Sampling**: Intelligent sampling based on scene complexity
- **Chunked Rendering**: Memory-efficient rendering for large scenes
- **Performance Tracking**: Real-time rendering performance metrics

### Advanced Performance Monitoring
- **Real-time System Metrics**: Comprehensive CPU, memory, GPU, and disk monitoring
- **Training Performance Analytics**: Detailed training speed and efficiency metrics
- **Performance Alerts**: Configurable alerts for system and training performance
- **Performance Baselines**: Automatic baseline establishment and regression detection
- **Performance Optimization**: Automated optimization recommendations
- **Performance Testing**: Automated performance regression testing
- **Resource Management**: GPU memory management and optimization
- **Horizontal Scaling**: Support for distributed training and inference
- **System Health Monitoring**: Comprehensive system health and stability tracking

## Development Setup

### Prerequisites

- Python 3.11+
- Node.js 18+
- Docker and Docker Compose
- CUDA-compatible GPU (recommended)

### Quick Start with Docker

1. Clone the repository:
```bash
git clone <repository-url>
cd nerf-studio
```

2. Start the development environment:
```bash
docker-compose up --build
```

3. Access the application:
- Frontend: http://localhost:5173
- Backend API: http://localhost:8000
- API Documentation: http://localhost:8000/docs

### Local Development

#### Backend Setup

1. Create a virtual environment:
```bash
cd backend
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Start the backend server:
```bash
uvicorn app.main:app --reload
```

#### Frontend Setup

1. Install dependencies:
```bash
cd frontend
npm install
```

2. Start the development server:
```bash
npm run dev
```

## Project Structure

```
nerf-studio/
├── backend/                 # FastAPI backend
│   ├── app/
│   │   ├── api/            # API routes
│   │   ├── core/           # Configuration
│   │   ├── ml/             # ML pipeline
│   │   ├── models/         # Data models
│   │   └── services/       # Business logic
│   └── requirements.txt
├── frontend/               # React frontend
│   ├── src/
│   │   ├── components/     # React components
│   │   └── ...
│   └── package.json
├── docker-compose.yml      # Development environment
└── .github/workflows/      # CI/CD pipelines
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests: `pytest` (backend) and `npm test` (frontend)
5. Submit a pull request

## License

MIT License