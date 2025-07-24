# 🎯 Specialized NeRF Studio Diagrams

## 🔄 **Image Upload & Processing Pipeline**

```mermaid
flowchart TD
    A[User Uploads Images] --> B[File Validation]
    B --> C{Valid Images?}
    C -->|No| D[Return Error]
    C -->|Yes| E[Save to Project Directory]
    E --> F[Update Project Metadata]
    F --> G[Image Preprocessing]
    G --> H[Resize Images]
    H --> I[Convert to RGB]
    I --> J[Generate Thumbnails]
    J --> K[Store Image List]
    K --> L[Ready for Training]
    
    style A fill:#e1f5fe
    style L fill:#c8e6c9
    style D fill:#ffcdd2
```

## 🎨 **NeRF Model Architecture**

```mermaid
graph TB
    subgraph "Input Processing"
        A[Camera Rays] --> B[Positional Encoding]
        A --> C[View Direction]
        C --> D[View Encoding]
    end
    
    subgraph "Coarse Network"
        E[Coarse MLP] --> F[Coarse Density]
        E --> G[Coarse Color]
        F --> H[Coarse Sampling]
    end
    
    subgraph "Fine Network"
        I[Fine MLP] --> J[Fine Density]
        I --> K[Fine Color]
        J --> L[Fine Sampling]
    end
    
    subgraph "Volume Rendering"
        M[Ray Marching] --> N[Alpha Compositing]
        N --> O[Final Color]
    end
    
    B --> E
    D --> E
    H --> I
    K --> M
    G --> M
    M --> O
    
    style A fill:#e3f2fd
    style O fill:#c8e6c9
```

## 📊 **Training Metrics & Monitoring**

```mermaid
graph LR
    subgraph "Real-time Metrics"
        A[Loss Function] --> B[PSNR]
        A --> C[SSIM]
        A --> D[LPIPS]
    end
    
    subgraph "System Metrics"
        E[GPU Memory] --> F[GPU Utilization]
        E --> G[Training Speed]
        H[CPU Usage] --> I[Memory Usage]
    end
    
    subgraph "Progress Tracking"
        J[Epoch Progress] --> K[Time Remaining]
        J --> L[Convergence Status]
    end
    
    subgraph "Alerts"
        M[High GPU Usage] --> N[Alert System]
        O[Low Progress] --> N
        P[System Errors] --> N
    end
    
    B --> Q[WebSocket Updates]
    C --> Q
    D --> Q
    F --> Q
    G --> Q
    K --> Q
    L --> Q
    N --> Q
    
    style Q fill:#fff3e0
```

## 🔧 **Export Pipeline Architecture**

```mermaid
flowchart TD
    A[Trained NeRF Model] --> B[Volume Sampling]
    B --> C[Mesh Extraction]
    C --> D[Marching Cubes]
    D --> E[Raw Mesh]
    
    E --> F[Mesh Optimization]
    F --> G[Texture Generation]
    G --> H[UV Mapping]
    H --> I[Texture Baking]
    
    I --> J{Export Format}
    J -->|GLTF| K[GLTF Export]
    J -->|OBJ| L[OBJ Export]
    J -->|PLY| M[PLY Export]
    J -->|FBX| N[FBX Export]
    
    K --> O[Compression]
    L --> O
    M --> O
    N --> O
    
    O --> P[Final Export]
    
    style A fill:#e8f5e8
    style P fill:#c8e6c9
```

## 🌐 **WebSocket Real-time Communication**

```mermaid
sequenceDiagram
    participant U as User Browser
    participant WS as WebSocket Server
    participant T as Training Process
    participant M as Metrics Collector
    participant DB as Database
    
    U->>WS: Connect to /ws/jobs/{job_id}
    WS->>DB: Register connection
    WS->>U: Connection established
    
    loop Training Progress
        T->>M: Update training metrics
        M->>WS: Send progress data
        WS->>U: Broadcast progress update
        U->>WS: Acknowledge receipt
    end
    
    loop System Monitoring
        M->>WS: Send system metrics
        WS->>U: Broadcast system status
    end
    
    U->>WS: Disconnect
    WS->>DB: Remove connection
    WS->>U: Connection closed
```

## 🗂️ **Project Lifecycle Management**

```mermaid
stateDiagram-v2
    [*] --> Created
    Created --> ImagesUploaded: Upload Images
    ImagesUploaded --> PosesEstimated: Estimate Poses
    PosesEstimated --> TrainingStarted: Start Training
    TrainingStarted --> TrainingInProgress: Begin Training
    TrainingInProgress --> TrainingCompleted: Training Done
    TrainingCompleted --> ModelExported: Export Model
    ModelExported --> [*]
    
    TrainingInProgress --> TrainingFailed: Error
    TrainingFailed --> TrainingStarted: Retry
    TrainingFailed --> [*]: Abandon
    
    ImagesUploaded --> TrainingStarted: Skip Pose Estimation
    PosesEstimated --> TrainingStarted: Manual Poses
```

## 🔄 **Background Job Processing**

```mermaid
graph TB
    subgraph "Job Queue"
        A[New Job] --> B[Job Queue]
        B --> C[Job Scheduler]
    end
    
    subgraph "Processing"
        C --> D[Resource Check]
        D --> E{Resources Available?}
        E -->|No| F[Wait in Queue]
        E -->|Yes| G[Start Processing]
        F --> D
    end
    
    subgraph "Execution"
        G --> H[Load Project Data]
        H --> I[Initialize Model]
        I --> J[Training Loop]
        J --> K[Save Checkpoints]
        K --> L[Update Progress]
        L --> M{Training Complete?}
        M -->|No| J
        M -->|Yes| N[Finalize Job]
    end
    
    subgraph "Completion"
        N --> O[Update Database]
        O --> P[Cleanup Resources]
        P --> Q[Job Complete]
    end
    
    style A fill:#e3f2fd
    style Q fill:#c8e6c9
    style F fill:#fff3e0
```

## 📱 **API Endpoint Structure**

```mermaid
graph TD
    A[FastAPI Router] --> B[Project Endpoints]
    A --> C[Training Endpoints]
    A --> D[Export Endpoints]
    A --> E[System Endpoints]
    A --> F[WebSocket Endpoints]
    
    B --> B1[POST /projects]
    B --> B2[GET /projects]
    B --> B3[GET /projects/{id}]
    B --> B4[DELETE /projects/{id}]
    B --> B5[POST /projects/{id}/upload_images]
    B --> B6[POST /projects/{id}/estimate_poses]
    
    C --> C1[POST /projects/{id}/start_training]
    C --> C2[GET /jobs/{id}]
    C --> C3[GET /projects/{id}/jobs]
    C --> C4[GET /jobs/{id}/metrics]
    
    D --> D1[POST /projects/{id}/export/advanced]
    D --> D2[GET /exports/{id}/status]
    D --> D3[GET /projects/{id}/download_export]
    
    E --> E1[GET /system/metrics]
    E --> E2[GET /docs]
    E --> E3[GET /redoc]
    
    F --> F1[WS /ws/jobs/{id}]
    
    style A fill:#e8f5e8
    style B1 fill:#e3f2fd
    style C1 fill:#e3f2fd
    style D1 fill:#e3f2fd
    style F1 fill:#e3f2fd
```

## 🔍 **Error Handling & Recovery**

```mermaid
flowchart TD
    A[API Request] --> B{Input Validation}
    B -->|Invalid| C[Return 400 Error]
    B -->|Valid| D[Process Request]
    
    D --> E{Database Operation}
    E -->|Success| F[Return Success]
    E -->|Failure| G[Database Error Handler]
    
    G --> H{Retry Possible?}
    H -->|Yes| I[Retry Operation]
    H -->|No| J[Return 500 Error]
    I --> E
    
    D --> K{File Operation}
    K -->|Success| L[Continue]
    K -->|Failure| M[File Error Handler]
    
    M --> N{File Exists?}
    N -->|Yes| O[Permission Error]
    N -->|No| P[File Not Found]
    
    O --> Q[Return 403 Error]
    P --> R[Return 404 Error]
    
    L --> S{ML Operation}
    S -->|Success| T[Return Result]
    S -->|Failure| U[ML Error Handler]
    
    U --> V{GPU Available?}
    V -->|Yes| W[GPU Error]
    V -->|No| X[CPU Fallback]
    
    W --> Y[Return 503 Error]
    X --> S
    
    style C fill:#ffcdd2
    style F fill:#c8e6c9
    style J fill:#ffcdd2
    style Q fill:#ffcdd2
    style R fill:#ffcdd2
    style T fill:#c8e6c9
    style Y fill:#ffcdd2
```

## 🎯 **Performance Optimization Flow**

```mermaid
graph LR
    subgraph "Input Optimization"
        A[Image Resizing] --> B[Batch Processing]
        B --> C[Memory Management]
    end
    
    subgraph "Model Optimization"
        D[Model Pruning] --> E[Quantization]
        E --> F[Mixed Precision]
    end
    
    subgraph "Training Optimization"
        G[Gradient Accumulation] --> H[Learning Rate Scheduling]
        H --> I[Early Stopping]
    end
    
    subgraph "Inference Optimization"
        J[Model Caching] --> K[Batch Inference]
        K --> L[GPU Memory Pinning]
    end
    
    C --> D
    F --> G
    I --> J
    L --> M[Optimized Performance]
    
    style M fill:#c8e6c9
```

## 🔐 **Security & Validation Flow**

```mermaid
flowchart TD
    A[Incoming Request] --> B[Rate Limiting]
    B --> C[Input Sanitization]
    C --> D[File Type Validation]
    D --> E[File Size Check]
    E --> F[Content Validation]
    
    F --> G{All Checks Pass?}
    G -->|No| H[Return Security Error]
    G -->|Yes| I[Process Request]
    
    I --> J[Database Query Sanitization]
    J --> K[Output Encoding]
    K --> L[Security Headers]
    L --> M[Return Response]
    
    style H fill:#ffcdd2
    style M fill:#c8e6c9
```

---

**🎨 Customization Tips:**

1. **Colors**: Use consistent color schemes for different types of components
   - Blue: Data/Storage components
   - Green: Success/Completion states
   - Red: Errors/Failures
   - Orange: Warnings/Processing
   - Purple: ML/AI components

2. **Styling**: Add `style` commands to highlight important nodes
3. **Layout**: Use different graph directions (TB, LR, TD) for different perspectives
4. **Grouping**: Use subgraphs to organize related components
5. **Flow**: Use different arrow styles to show different types of relationships

**📊 Export Options:**
- PNG: For presentations and documentation
- SVG: For web use and scaling
- PDF: For print and formal documentation 