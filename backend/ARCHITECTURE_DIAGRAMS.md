# 🏗️ NeRF Studio Architecture Diagrams

## 📊 **System Architecture Overview**

```mermaid
graph TB
    subgraph "Frontend"
        UI[Web UI / React]
        WS[WebSocket Client]
    end
    
    subgraph "Backend API Layer"
        API[FastAPI Server]
        Router[API Router]
        Middleware[CORS, Auth]
    end
    
    subgraph "Service Layer"
        PS[Project Service]
        JS[Job Service]
        WSManager[WebSocket Manager]
    end
    
    subgraph "ML Pipeline"
        Trainer[NeRF Trainer]
        Model[Hierarchical NeRF]
        Inference[Fast NeRF Inference]
        Export[Advanced Export]
    end
    
    subgraph "Data Layer"
        DB[(SQLite Database)]
        FileSystem[File Storage]
        Checkpoints[Model Checkpoints]
    end
    
    subgraph "External Services"
        COLMAP[COLMAP Pose Estimation]
        Redis[(Redis Cache)]
    end
    
    UI --> API
    WS --> WSManager
    API --> Router
    Router --> PS
    Router --> JS
    PS --> DB
    JS --> DB
    PS --> FileSystem
    JS --> Trainer
    Trainer --> Model
    Trainer --> Checkpoints
    Trainer --> Inference
    Inference --> Export
    PS --> COLMAP
    WSManager --> Redis
```

## 🔄 **API Request Flow**

```mermaid
sequenceDiagram
    participant Client
    participant API as FastAPI
    participant Service as Project Service
    participant DB as SQLite DB
    participant FileSystem
    participant Trainer as NeRF Trainer
    participant WS as WebSocket Manager
    
    Client->>API: POST /projects (Create Project)
    API->>Service: create_project()
    Service->>DB: Insert project record
    Service->>FileSystem: Create project directory
    Service->>DB: Update project data
    API->>Client: Return project data
    
    Client->>API: POST /projects/{id}/upload_images
    API->>Service: get_project()
    Service->>DB: Fetch project
    API->>FileSystem: Save uploaded images
    API->>Service: update_project_data()
    Service->>DB: Update project metadata
    API->>Client: Return upload status
    
    Client->>API: POST /projects/{id}/start_training
    API->>Service: get_project()
    API->>Service: create_job()
    Service->>DB: Insert training job
    API->>Trainer: Start training (background)
    Trainer->>WS: Send progress updates
    WS->>Client: Real-time training progress
    API->>Client: Return job ID
```

## 🎯 **NeRF Training Pipeline**

```mermaid
flowchart TD
    A[Input Images] --> B[Image Preprocessing]
    B --> C{Has Camera Poses?}
    C -->|No| D[COLMAP Pose Estimation]
    C -->|Yes| E[Load Poses]
    D --> E
    E --> F[Generate Rays]
    F --> G[Initialize NeRF Model]
    G --> H[Training Loop]
    
    H --> I[Sample Rays]
    I --> J[Volume Rendering]
    J --> K[Compute Loss]
    K --> L[Backpropagation]
    L --> M[Update Model]
    M --> N{Converged?}
    N -->|No| I
    N -->|Yes| O[Save Checkpoint]
    
    O --> P[Model Evaluation]
    P --> Q[Novel View Rendering]
    Q --> R[Export Options]
    R --> S[GLTF/OBJ Export]
    R --> T[Texture Baking]
    R --> U[Mesh Optimization]
```

## 🗄️ **Database Schema**

```mermaid
erDiagram
    PROJECTS {
        string id PK
        string name
        string description
        datetime created_at
        datetime updated_at
        json data
        json config
        json checkpoints
    }
    
    TRAINING_JOBS {
        string id PK
        string project_id FK
        string status
        datetime created_at
        datetime started_at
        datetime completed_at
        json metrics
        float progress
    }
    
    PROJECTS ||--o{ TRAINING_JOBS : "has"
```

## 📁 **File System Structure**

```mermaid
graph TD
    A[NeRF Studio Backend] --> B[data/]
    A --> C[app/]
    A --> D[nerf_studio.db]
    
    B --> E[projects/]
    E --> F[project-id-1/]
    E --> G[project-id-2/]
    E --> H[project-id-n/]
    
    F --> I[images/]
    F --> J[checkpoints/]
    F --> K[exports/]
    F --> L[poses.json]
    F --> M[project_meta.json]
    
    I --> N[image1.jpg]
    I --> O[image2.jpg]
    I --> P[imageN.jpg]
    
    J --> Q[nerf_step_1000.pth]
    J --> R[nerf_step_2000.pth]
    J --> S[nerf_final.pth]
    
    K --> T[model.gltf]
    K --> U[model.obj]
    K --> V[textures/]
```

## 🔧 **Service Architecture**

```mermaid
graph LR
    subgraph "API Layer"
        A1[Project API]
        A2[Training API]
        A3[Export API]
        A4[System API]
    end
    
    subgraph "Service Layer"
        S1[Project Service]
        S2[Job Service]
        S3[Export Service]
        S4[Performance Monitor]
    end
    
    subgraph "ML Components"
        M1[NeRF Trainer]
        M2[Model Manager]
        M3[Inference Engine]
        M4[Export Pipeline]
    end
    
    subgraph "Data Access"
        D1[Database Manager]
        D2[File Manager]
        D3[Cache Manager]
    end
    
    A1 --> S1
    A2 --> S2
    A3 --> S3
    A4 --> S4
    
    S1 --> D1
    S1 --> D2
    S2 --> M1
    S3 --> M4
    
    M1 --> M2
    M1 --> D2
    M4 --> D2
    S4 --> D3
```

## 🌐 **WebSocket Communication Flow**

```mermaid
sequenceDiagram
    participant Client
    participant WS as WebSocket Manager
    participant Trainer as NeRF Trainer
    participant Monitor as Performance Monitor
    
    Client->>WS: Connect to /ws/jobs/{job_id}
    WS->>WS: Register client connection
    
    loop Training Progress
        Trainer->>Monitor: Update metrics
        Monitor->>WS: Send progress update
        WS->>Client: Broadcast progress
    end
    
    Client->>WS: Disconnect
    WS->>WS: Remove client connection
```

## 🚀 **Deployment Architecture**

```mermaid
graph TB
    subgraph "Client Layer"
        Web[Web Browser]
        Mobile[Mobile App]
        API_Client[API Client]
    end
    
    subgraph "Load Balancer"
        LB[NGINX/HAProxy]
    end
    
    subgraph "Application Layer"
        API1[FastAPI Instance 1]
        API2[FastAPI Instance 2]
        API3[FastAPI Instance N]
    end
    
    subgraph "Data Layer"
        DB1[(SQLite Primary)]
        DB2[(SQLite Replica)]
        FileStorage[File Storage]
    end
    
    subgraph "ML Infrastructure"
        GPU1[GPU Node 1]
        GPU2[GPU Node 2]
        Queue[Job Queue]
    end
    
    Web --> LB
    Mobile --> LB
    API_Client --> LB
    
    LB --> API1
    LB --> API2
    LB --> API3
    
    API1 --> DB1
    API2 --> DB1
    API3 --> DB1
    
    API1 --> FileStorage
    API2 --> FileStorage
    API3 --> FileStorage
    
    API1 --> Queue
    API2 --> Queue
    API3 --> Queue
    
    Queue --> GPU1
    Queue --> GPU2
```

## 🔄 **Complete User Journey**

```mermaid
journey
    title NeRF Studio User Journey
    section Project Creation
      Upload Images: 5: User
      Create Project: 5: User
      Configure Settings: 4: User
    section Training
      Start Training: 5: User
      Monitor Progress: 4: User
      Wait for Completion: 3: User
    section Results
      View Rendered Images: 5: User
      Export 3D Model: 4: User
      Share Results: 3: User
```

## 📊 **Performance Monitoring**

```mermaid
graph TD
    A[System Metrics] --> B[CPU Usage]
    A --> C[GPU Usage]
    A --> D[Memory Usage]
    A --> E[Disk I/O]
    
    F[Training Metrics] --> G[Loss Function]
    F --> H[PSNR]
    F --> I[SSIM]
    F --> J[Training Time]
    
    K[Performance Monitor] --> A
    K --> F
    K --> L[Alert System]
    
    L --> M[High GPU Usage]
    L --> N[Low Training Progress]
    L --> O[System Errors]
```

## 🔐 **Security Architecture**

```mermaid
graph TD
    A[Client Request] --> B[Load Balancer]
    B --> C[Rate Limiting]
    C --> D[Authentication]
    D --> E[Authorization]
    E --> F[Input Validation]
    F --> G[API Endpoint]
    G --> H[Service Layer]
    H --> I[Database]
    
    J[Security Headers] --> B
    K[CORS Policy] --> G
    L[SQL Injection Protection] --> I
    M[File Upload Validation] --> F
```

## 📈 **Scalability Architecture**

```mermaid
graph TB
    subgraph "Horizontal Scaling"
        API1[API Instance 1]
        API2[API Instance 2]
        API3[API Instance N]
    end
    
    subgraph "Vertical Scaling"
        GPU1[GPU Node 1]
        GPU2[GPU Node 2]
        GPU3[GPU Node N]
    end
    
    subgraph "Data Scaling"
        DB1[(Database 1)]
        DB2[(Database 2)]
        Cache[(Redis Cache)]
    end
    
    subgraph "Storage Scaling"
        Storage1[File Storage 1]
        Storage2[File Storage 2]
        CDN[CDN]
    end
    
    API1 --> GPU1
    API2 --> GPU2
    API3 --> GPU3
    
    API1 --> DB1
    API2 --> DB2
    API1 --> Cache
    API2 --> Cache
    
    API1 --> Storage1
    API2 --> Storage2
    Storage1 --> CDN
    Storage2 --> CDN
```

## 🎯 **Key Components Summary**

| Component | Purpose | Technology |
|-----------|---------|------------|
| **FastAPI** | Web framework | Python |
| **SQLite** | Database | File-based |
| **NeRF Model** | 3D reconstruction | PyTorch |
| **WebSocket** | Real-time updates | Python |
| **COLMAP** | Pose estimation | C++/Python |
| **File Storage** | Image/model storage | Local filesystem |
| **Performance Monitor** | System monitoring | Python |

## 🔧 **Configuration Management**

```mermaid
graph LR
    A[Environment Variables] --> B[Settings Class]
    B --> C[Database Config]
    B --> D[API Config]
    B --> E[ML Config]
    B --> F[File Storage Config]
    
    G[.env File] --> A
    H[Docker Environment] --> A
    I[Kubernetes Config] --> A
```

---

**📝 Usage Instructions:**
1. Copy any diagram code block
2. Paste into Mermaid Live Editor: https://mermaid.live/
3. Customize colors, styling, and layout
4. Export as PNG, SVG, or PDF
5. Use in documentation, presentations, or architecture docs 