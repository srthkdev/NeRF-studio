# 🚀 NeRF Studio Deployment Guide

This guide covers deploying NeRF Studio to various platforms, with a focus on Render.com.

## 📋 Prerequisites

- Git repository with your NeRF Studio code
- Render.com account (free tier available)
- Basic understanding of Docker and web services

## 🎯 Quick Deploy to Render

### Option 1: Using render.yaml (Recommended)

1. **Fork/Clone the Repository**
   ```bash
   git clone https://github.com/your-username/NeRF-studio.git
   cd NeRF-studio
   ```

2. **Connect to Render**
   - Go to [render.com](https://render.com)
   - Sign up/Login with your GitHub account
   - Click "New +" → "Blueprint"

3. **Deploy from Repository**
   - Connect your GitHub repository
   - Render will automatically detect the `render.yaml` file
   - Click "Apply" to deploy both services

4. **Configure Environment Variables**
   - Backend service: No additional configuration needed (uses SQLite)
   - Frontend service: Update `VITE_API_URL` to point to your backend URL

### Option 2: Manual Deployment

#### Backend Service

1. **Create Web Service**
   - Name: `nerf-studio-backend`
   - Environment: `Python`
   - Build Command: `cd backend && pip install -r requirements.txt`
   - Start Command: `cd backend && python -m uvicorn app.main:app --host 0.0.0.0 --port $PORT`

2. **Environment Variables**
   ```
   PYTHON_VERSION=3.11.0
   DATABASE_URL=sqlite+aiosqlite:///./nerf_studio.db
   DEBUG=false
   MAX_CONCURRENT_JOBS=2
   UPLOAD_DIR=./uploads
   MODEL_DIR=./models
   ```

#### Frontend Service

1. **Create Static Site**
   - Name: `nerf-studio-frontend`
   - Build Command: `cd frontend && npm install && npm run build`
   - Publish Directory: `frontend/dist`

2. **Environment Variables**
   ```
   VITE_API_URL=https://your-backend-service.onrender.com/api/v1
   ```

## 🐳 Docker Deployment

### Local Docker Testing

```bash
# Build and run with Docker Compose
docker-compose up --build

# Access the application
# Frontend: http://localhost
# Backend: http://localhost:8000
```

### Production Docker Deployment

```bash
# Build production images
docker build -f Dockerfile.backend -t nerf-studio-backend .
docker build -f Dockerfile.frontend -t nerf-studio-frontend .

# Run with production settings
docker run -d -p 8000:8000 --name nerf-backend nerf-studio-backend
docker run -d -p 80:80 --name nerf-frontend nerf-studio-frontend
```

## 🔧 Configuration

### Environment Variables

#### Backend (.env)
```env
# Database
DATABASE_URL=sqlite+aiosqlite:///./nerf_studio.db

# Development
DEBUG=false

# Training
MAX_CONCURRENT_JOBS=2

# Storage
UPLOAD_DIR=./uploads
MODEL_DIR=./models
```

#### Frontend (.env)
```env
# API Configuration
VITE_API_URL=https://your-backend-url.com/api/v1
```

### CORS Configuration

The backend automatically configures CORS based on the `DEBUG` environment variable:

- **Development** (`DEBUG=true`): Allows all origins
- **Production** (`DEBUG=false`): Uses specific origins from `BACKEND_CORS_ORIGINS`

## 📊 Monitoring & Health Checks

### Health Check Endpoint
- URL: `/api/v1/health`
- Returns: `{"status": "healthy", "service": "nerf-studio-backend"}`

### Logs
- Backend logs are available in Render dashboard
- Frontend logs are minimal (static site)

## 🛠️ Troubleshooting

### Common Issues

1. **Build Failures**
   ```bash
   # Check Python version compatibility
   python --version  # Should be 3.11+
   
   # Check Node.js version
   node --version    # Should be 18+
   ```

2. **Database Issues**
   ```bash
   # SQLite database is created automatically
   # No manual setup required
   ```

3. **CORS Errors**
   - Ensure `VITE_API_URL` is correctly set
   - Check backend CORS configuration
   - Verify HTTPS/HTTP protocol matching

4. **Memory Issues**
   - Reduce `MAX_CONCURRENT_JOBS` if needed
   - Consider upgrading Render plan for more resources

### Performance Optimization

1. **Backend**
   - Use Render's paid plans for better performance
   - Optimize image processing with smaller batch sizes
   - Enable caching where possible

2. **Frontend**
   - Static assets are automatically cached
   - Consider CDN for global distribution
   - Optimize bundle size with code splitting

## 🔒 Security Considerations

### Production Security
- HTTPS is automatically enabled on Render
- CORS is properly configured for production
- No sensitive data in environment variables
- SQLite database is file-based and secure

### Additional Security
- Consider adding authentication if needed
- Implement rate limiting for API endpoints
- Regular security updates for dependencies

## 📈 Scaling

### Render Scaling Options
- **Free Tier**: Limited resources, good for testing
- **Starter Plan**: Better performance, suitable for small projects
- **Standard Plan**: More resources, suitable for production use

### Horizontal Scaling
- Backend can be scaled horizontally
- Frontend is static and can be served from CDN
- Database remains SQLite (consider PostgreSQL for larger scale)

## 🔄 Continuous Deployment

### GitHub Integration
1. Connect your GitHub repository to Render
2. Enable automatic deployments
3. Deploy on every push to main branch

### Manual Deployment
```bash
# Push changes to trigger deployment
git add .
git commit -m "Update deployment"
git push origin main
```

## 📞 Support

### Render Support
- [Render Documentation](https://render.com/docs)
- [Render Community](https://community.render.com)

### Application Support
- Check logs in Render dashboard
- Monitor health check endpoint
- Review error messages in browser console

## 🎉 Success!

Once deployed, your NeRF Studio will be available at:
- **Frontend**: `https://your-frontend-service.onrender.com`
- **Backend**: `https://your-backend-service.onrender.com`
- **API Docs**: `https://your-backend-service.onrender.com/docs`

---

**Note**: The free tier of Render has limitations on compute resources and may not be suitable for heavy NeRF training. Consider upgrading to a paid plan for production use. 

## 🚀 Static Site Deployment on Render

### Frontend Static Site Configuration

#### **Build Commands for Render**

```bash
# Production build command (used by Render)
cd frontend
npm ci
npm run build
```

#### **Environment Variables for Static Site**

```bash
# Required Environment Variables
NODE_VERSION=18.17.0
VITE_API_URL=https://nerf-studio.onrender.com
VITE_APP_NAME=NeRF Studio
VITE_APP_VERSION=1.0.0
VITE_ENVIRONMENT=production
```

#### **Manual Deployment Commands**

```bash
# Local build for testing
cd frontend
npm install
npm run build

# Preview built site locally
npm run preview

# Deploy to Render (via Git push)
git add .
git commit -m "Deploy frontend static site"
git push origin main
```

#### **Render Configuration (render.yaml)**

```yaml
# Frontend Static Site
- type: web
  name: nerf-studio-frontend
  env: static
  plan: starter
  buildCommand: |
    cd frontend
    npm ci
    npm run build
  staticPublishPath: ./frontend/dist
  envVars:
    - key: NODE_VERSION
      value: 18.17.0
    - key: VITE_API_URL
      value: https://nerf-studio.onrender.com
    - key: VITE_APP_NAME
      value: NeRF Studio
    - key: VITE_APP_VERSION
      value: 1.0.0
    - key: VITE_ENVIRONMENT
      value: production
  routes:
    - type: rewrite
      source: /*
      destination: /index.html
  headers:
    - path: /*
      name: Cache-Control
      value: public, max-age=31536000, immutable
    - path: /assets/*
      name: Cache-Control
      value: public, max-age=31536000, immutable
    - path: /api/*
      name: Cache-Control
      value: no-cache, no-store, must-revalidate
```

### **Build Optimization Features**

1. **Code Splitting**: Vendor, Three.js, and Chart.js libraries are split into separate chunks
2. **Minification**: Terser minification for optimal bundle size
3. **Caching**: Long-term caching for static assets
4. **SPA Routing**: All routes redirect to index.html for client-side routing

### **Performance Optimizations**

- **Bundle Analysis**: Use `npm run build -- --analyze` to analyze bundle size
- **Image Optimization**: All images in `/public` are optimized for web delivery
- **CDN Ready**: Static assets are configured for CDN deployment
- **Gzip Compression**: Enabled by default on Render

### **Environment-Specific Configurations**

#### **Development Environment**
```bash
# Local development
npm run dev
# API URL: http://localhost:8000
```

#### **Staging Environment**
```bash
# Staging build
VITE_API_URL=https://staging-nerf-backend.onrender.com
npm run build
```

#### **Production Environment**
```bash
# Production build
VITE_API_URL=https://nerf-studio.onrender.com
VITE_ENVIRONMENT=production
npm run build
```

### **Deployment Checklist**

- [ ] Backend API is deployed and accessible
- [ ] Environment variables are configured in Render dashboard
- [ ] Build command completes successfully
- [ ] Static files are generated in `frontend/dist/`
- [ ] SPA routing works correctly
- [ ] API calls are working from frontend
- [ ] Assets are loading properly
- [ ] Performance metrics are acceptable

### **Troubleshooting**

#### **Build Failures**
```bash
# Clear node_modules and reinstall
rm -rf node_modules package-lock.json
npm install

# Check for TypeScript errors
npm run lint

# Verify build locally
npm run build
```

#### **Runtime Issues**
```bash
# Check environment variables
echo $VITE_API_URL

# Verify API connectivity
curl https://nerf-studio.onrender.com/api/v1/health

# Check browser console for errors
```

#### **Performance Issues**
```bash
# Analyze bundle size
npm run build -- --analyze

# Check asset sizes
ls -la frontend/dist/assets/

# Verify caching headers
curl -I https://your-frontend.onrender.com/assets/main.js
```

--- 