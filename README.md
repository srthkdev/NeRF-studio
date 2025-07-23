# NeRF Studio

A production-grade Neural Radiance Fields platform that enables users to upload photos and generate interactive 3D scene reconstructions.

## Features

- Upload collections of photos for 3D reconstruction
- Automatic camera pose estimation using COLMAP
- Real-time NeRF training with progress monitoring
- Interactive 3D viewer with WebGL rendering
- Export trained models to standard 3D formats

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