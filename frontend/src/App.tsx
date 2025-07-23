// NOTE: Make sure to have 'react', 'react-dom', and '@types/react' installed in your project for proper TypeScript support.
import React, { useEffect, useState, useRef, ChangeEvent, DragEvent, FormEvent } from 'react';
import './App.css';
import { Canvas } from '@react-three/fiber';
import { OrbitControls, Text, Box } from '@react-three/drei';
import { Line } from 'react-chartjs-2';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend
} from 'chart.js';

// Register Chart.js components
ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend
);

const API_BASE = '/api/v1';

type Project = { id: string; name: string };
type Job = { id: string; status: string; progress?: number; config?: any };
type ImageWithPose = { filename: string; pose?: number[] };
type ProjectConfig = Record<string, any>;
type TrainingMetrics = {
  step: number;
  loss: number;
  psnr: number;
  lr: number;
  best_psnr: number;
  eta_seconds: number;
  status: string;
  progress: number;
};

function App() {
  const [projects, setProjects] = useState<Project[]>([]);
  const [selectedProject, setSelectedProject] = useState<Project | null>(null);
  const [newProjectName, setNewProjectName] = useState('');
  const [uploading, setUploading] = useState(false);
  const [images, setImages] = useState<ImageWithPose[]>([]);
  const fileInputRef = useRef<HTMLInputElement | null>(null);
  const [dragActive, setDragActive] = useState(false);
  const [job, setJob] = useState<Job | null>(null);
  const [jobStatus, setJobStatus] = useState<string | null>(null);
  const [jobProgress, setJobProgress] = useState<number>(0);
  const wsRef = useRef<WebSocket | null>(null);
  // New state for config and pose
  const [projectConfig, setProjectConfig] = useState<ProjectConfig>({});
  const [configEdit, setConfigEdit] = useState('');
  const [configEditing, setConfigEditing] = useState(false);
  const [poseInputs, setPoseInputs] = useState<Record<string, string>>({});
  const [poseUploading, setPoseUploading] = useState<Record<string, boolean>>({});
  const [cameraCenters, setCameraCenters] = useState<number[][]>([]);
  
  // New state for enhanced training dashboard
  const [trainingMetrics, setTrainingMetrics] = useState<TrainingMetrics | null>(null);
  const [metricsHistory, setMetricsHistory] = useState<any>(null);
  const [trainingConfig, setTrainingConfig] = useState({
    batch_size: 4096,
    lr: 5e-4,
    num_epochs: 200000,
    hidden_dim: 256,
    n_coarse: 64,
    n_fine: 128
  });
  const [showTrainingConfig, setShowTrainingConfig] = useState(false);
  const [meshExtracting, setMeshExtracting] = useState(false);
  const [meshFiles, setMeshFiles] = useState<{[key: string]: string}>({});
  const [systemMetrics, setSystemMetrics] = useState<any>(null);
  const [trainingSummary, setTrainingSummary] = useState<any>(null);

  // Chart.js data state
  const [chartData, setChartData] = useState<{
    labels: number[];
    datasets: {
      label: string;
      data: number[];
      borderColor: string;
      backgroundColor: string;
      tension: number;
      yAxisID?: string;
    }[];
  }>({
    labels: [],
    datasets: [
      {
        label: 'Loss',
        data: [],
        borderColor: 'rgb(255, 99, 132)',
        backgroundColor: 'rgba(255, 99, 132, 0.5)',
        tension: 0.1
      },
      {
        label: 'PSNR',
        data: [],
        borderColor: 'rgb(53, 162, 235)',
        backgroundColor: 'rgba(53, 162, 235, 0.5)',
        tension: 0.1,
        yAxisID: 'y1'
      }
    ]
  });

  const [chartOptions] = useState({
    responsive: true,
    interaction: {
      mode: 'index' as const,
      intersect: false,
    },
    scales: {
      x: {
        display: true,
        title: {
          display: true,
          text: 'Training Step'
        }
      },
      y: {
        type: 'linear' as const,
        display: true,
        position: 'left' as const,
        title: {
          display: true,
          text: 'Loss'
        }
      },
      y1: {
        type: 'linear' as const,
        display: true,
        position: 'right' as const,
        title: {
          display: true,
          text: 'PSNR'
        },
        grid: {
          drawOnChartArea: false,
        },
      },
    },
    plugins: {
      legend: {
        position: 'top' as const,
      },
      title: {
        display: true,
        text: 'Training Metrics'
      }
    }
  });

  // Fetch projects
  useEffect(() => {
    fetch(`${API_BASE}/projects`)
      .then(res => res.json())
      .then(data => setProjects(data.projects || []));
  }, []);

  // Fetch images and config for selected project
  useEffect(() => {
    if (selectedProject) {
      fetch(`${API_BASE}/projects/${selectedProject.id}/images`)
        .then(res => res.json())
        .then(data => setImages(data.images || []));
      fetch(`${API_BASE}/projects/${selectedProject.id}/config`)
        .then(res => res.json())
        .then(data => { setProjectConfig(data); setConfigEdit(JSON.stringify(data, null, 2)); });
      fetch(`${API_BASE}/projects/${selectedProject.id}/visualize_poses`)
        .then(res => res.json())
        .then(data => setCameraCenters(data.camera_centers || []));
    } else {
      setImages([]);
      setProjectConfig({});
      setConfigEdit('');
      setCameraCenters([]);
    }
  }, [selectedProject]);

  // Create new project
  const handleCreateProject = async (e: FormEvent<HTMLFormElement>) => {
    e.preventDefault();
    if (!newProjectName.trim()) return;
    const form = new FormData();
    form.append('name', newProjectName);
    const res = await fetch(`${API_BASE}/projects`, { method: 'POST', body: form });
    const data = await res.json();
    setProjects((p: Project[]) => [...p, { id: data.project_id, name: data.name }]);
    setNewProjectName('');
  };

  // Handle file upload
  const handleFiles = async (files: FileList | null) => {
    if (!selectedProject || !files || files.length === 0) return;
    setUploading(true);
    const form = new FormData();
    Array.from(files).forEach(file => {
      form.append('files', file);
    });
    await fetch(`${API_BASE}/projects/${selectedProject.id}/upload_images`, {
      method: 'POST',
      body: form,
    });
    // Refresh images
    fetch(`${API_BASE}/projects/${selectedProject.id}/images`)
      .then(res => res.json())
      .then(data => setImages(data.images || []));
    setUploading(false);
  };

  // Drag and drop handlers
  const handleDrag = (e: DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === 'dragenter' || e.type === 'dragover') setDragActive(true);
    else if (e.type === 'dragleave') setDragActive(false);
  };
  const handleDrop = (e: DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      handleFiles(e.dataTransfer.files);
    }
  };

  // Start training job with configuration
  const handleStartTraining = async () => {
    if (!selectedProject) return;
    
    const form = new FormData();
    form.append('project_id', selectedProject.id);
    form.append('config', JSON.stringify(trainingConfig));
    
    const res = await fetch(`${API_BASE}/jobs/submit`, { method: 'POST', body: form });
    const data = await res.json();
    
    setJob({ id: data.job_id, status: data.status, config: data.config });
    setJobStatus(data.status);
    setJobProgress(0);
    setTrainingMetrics(null);
    
    // Open WebSocket for real-time updates
    if (wsRef.current) wsRef.current.close();
    const ws = new WebSocket(`${window.location.protocol === 'https:' ? 'wss' : 'ws'}://${window.location.host}/api/v1/ws/jobs/${data.job_id}`);
    
    ws.onmessage = (event) => {
      const msg = JSON.parse(event.data);
      if (msg.status) setJobStatus(msg.status);
      if (typeof msg.progress === 'number') setJobProgress(msg.progress);
      
      // Update training metrics
      if (msg.loss !== undefined) {
        setTrainingMetrics({
          step: msg.step || 0,
          loss: msg.loss,
          psnr: msg.psnr,
          lr: msg.lr,
          best_psnr: msg.best_psnr,
          eta_seconds: msg.eta_seconds,
          status: msg.status,
          progress: msg.progress
        });
      }
    };
    
    ws.onclose = () => { wsRef.current = null; };
    wsRef.current = ws;
  };

  // Cancel training job
  const handleCancelTraining = async () => {
    if (!job) return;
    
    await fetch(`${API_BASE}/jobs/${job.id}/cancel`, { method: 'POST' });
    setJobStatus('cancelled');
  };

  // Fetch metrics history for plotting
  const fetchMetricsHistory = async () => {
    if (!job) return;
    
    try {
      const res = await fetch(`${API_BASE}/jobs/${job.id}/metrics_history`);
      const data = await res.json();
      setMetricsHistory(data);
    } catch (e) {
      console.error('Failed to fetch metrics history:', e);
    }
  };

  // Save config
  const handleSaveConfig = async () => {
    try {
      const parsed = JSON.parse(configEdit);
      await fetch(`${API_BASE}/projects/${selectedProject!.id}/config`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(parsed),
      });
      setProjectConfig(parsed);
      setConfigEditing(false);
    } catch (e) {
      alert('Invalid JSON');
    }
  };

  // Pose upload for an image
  const handlePoseInputChange = (filename: string, value: string) => {
    setPoseInputs(inputs => ({ ...inputs, [filename]: value }));
  };
  const handleUploadPose = async (filename: string) => {
    setPoseUploading(p => ({ ...p, [filename]: true }));
    try {
      const poseArr = JSON.parse(poseInputs[filename]);
      if (!Array.isArray(poseArr) || poseArr.length !== 16) throw new Error();
      await fetch(`${API_BASE}/projects/${selectedProject!.id}/poses`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ [filename]: poseArr }),
      });
      // Refresh images and camera centers
      fetch(`${API_BASE}/projects/${selectedProject!.id}/images`)
        .then(res => res.json())
        .then(data => setImages(data.images || []));
      fetch(`${API_BASE}/projects/${selectedProject!.id}/visualize_poses`)
        .then(res => res.json())
        .then(data => setCameraCenters(data.camera_centers || []));
      setPoseInputs(inputs => ({ ...inputs, [filename]: '' }));
    } catch {
      alert('Pose must be a JSON array of 16 numbers (4x4 matrix, row-major)');
    }
    setPoseUploading(p => ({ ...p, [filename]: false }));
  };

  // Update chart data when metrics change
  useEffect(() => {
    if (trainingMetrics && trainingMetrics.step > 0) {
      setChartData(prev => {
        const newLabels = [...prev.labels, trainingMetrics.step];
        const newLossData = [...prev.datasets[0].data, trainingMetrics.loss];
        const newPsnrData = [...prev.datasets[1].data, trainingMetrics.psnr];
        
        // Keep only last 100 points for performance
        const maxPoints = 100;
        if (newLabels.length > maxPoints) {
          return {
            labels: newLabels.slice(-maxPoints),
            datasets: [
              {
                ...prev.datasets[0],
                data: newLossData.slice(-maxPoints)
              },
              {
                ...prev.datasets[1],
                data: newPsnrData.slice(-maxPoints)
              }
            ]
          };
        }
        
        return {
          labels: newLabels,
          datasets: [
            {
              ...prev.datasets[0],
              data: newLossData
            },
            {
              ...prev.datasets[1],
              data: newPsnrData
            }
          ]
        };
      });
    }
  }, [trainingMetrics]);

  // Clean up WebSocket on unmount
  useEffect(() => {
    return () => { if (wsRef.current) wsRef.current.close(); };
  }, []);

  const extractMesh = async () => {
    if (!selectedProject) return;
    
    setMeshExtracting(true);
    try {
      const response = await fetch(`${API_BASE}/projects/${selectedProject.id}/extract-mesh`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          bounds: [-2, 2, -2, 2, -2, 2],
          resolution: 128,
          formats: ['gltf', 'obj', 'ply']
        })
      });
      
      if (response.ok) {
        const result = await response.json();
        setMeshFiles(result.files);
        alert('Mesh extracted successfully!');
      } else {
        const error = await response.json();
        alert(`Mesh extraction failed: ${error.detail}`);
      }
    } catch (error) {
      console.error('Mesh extraction error:', error);
      alert('Mesh extraction failed');
    } finally {
      setMeshExtracting(false);
    }
  };

  const downloadMesh = async (format: string) => {
    if (!selectedProject) return;
    
    try {
      const response = await fetch(`${API_BASE}/projects/${selectedProject.id}/download-mesh?format=${format}`);
      if (response.ok) {
        const blob = await response.blob();
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `nerf_mesh_${selectedProject.id}.${format}`;
        document.body.appendChild(a);
        a.click();
        window.URL.revokeObjectURL(url);
        document.body.removeChild(a);
      } else {
        alert(`Download failed for ${format} format`);
      }
    } catch (error) {
      console.error('Download error:', error);
      alert('Download failed');
    }
  };

  const downloadAllMeshes = async () => {
    if (!selectedProject) return;
    
    try {
      const response = await fetch(`${API_BASE}/projects/${selectedProject.id}/download-all-meshes`);
      if (response.ok) {
        const blob = await response.blob();
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `nerf_meshes_${selectedProject.id}.zip`;
        document.body.appendChild(a);
        a.click();
        window.URL.revokeObjectURL(url);
        document.body.removeChild(a);
      } else {
        alert('Download failed');
      }
    } catch (error) {
      console.error('Download error:', error);
      alert('Download failed');
    }
  };

  const fetchSystemMetrics = async () => {
    try {
      const response = await fetch(`${API_BASE}/system/metrics`);
      if (response.ok) {
        const metrics = await response.json();
        setSystemMetrics(metrics);
      }
    } catch (error) {
      console.error('Failed to fetch system metrics:', error);
    }
  };

  const fetchTrainingSummary = async () => {
    try {
      const response = await fetch(`${API_BASE}/training/summary`);
      if (response.ok) {
        const summary = await response.json();
        setTrainingSummary(summary);
      }
    } catch (error) {
      console.error('Failed to fetch training summary:', error);
    }
  };

  // Fetch system metrics periodically
  useEffect(() => {
    const interval = setInterval(fetchSystemMetrics, 2000);
    return () => clearInterval(interval);
  }, []);

  // Fetch training summary periodically when training
  useEffect(() => {
    if (job && job.status === 'running') {
      const interval = setInterval(fetchTrainingSummary, 5000);
      return () => clearInterval(interval);
    }
  }, [job]);

  // Enhanced 3D Scene Component
  const Scene3D = () => {
    return (
      <>
        <ambientLight intensity={0.5} />
        <pointLight position={[10, 10, 10]} />
        <directionalLight position={[-10, -10, -5]} intensity={0.3} />
        
        {/* Camera centers as spheres */}
        {cameraCenters.map((center, i) => (
          <group key={i} position={center as [number, number, number]}>
            <mesh>
              <sphereGeometry args={[0.05, 16, 16]} />
              <meshStandardMaterial color="red" />
            </mesh>
            {/* Camera frustum visualization */}
            <mesh position={[0, 0, 0.1]}>
              <coneGeometry args={[0.02, 0.1, 8]} />
              <meshStandardMaterial color="orange" transparent opacity={0.7} />
            </mesh>
          </group>
        ))}
        
        {/* Coordinate axes */}
        <group>
          <mesh position={[0, 0, 0]}>
            <boxGeometry args={[0.01, 0.01, 0.01]} />
            <meshStandardMaterial color="white" />
          </mesh>
          {/* X axis */}
          <mesh position={[0.5, 0, 0]}>
            <boxGeometry args={[1, 0.01, 0.01]} />
            <meshStandardMaterial color="red" />
          </mesh>
          {/* Y axis */}
          <mesh position={[0, 0.5, 0]}>
            <boxGeometry args={[0.01, 1, 0.01]} />
            <meshStandardMaterial color="green" />
          </mesh>
          {/* Z axis */}
          <mesh position={[0, 0, 0.5]}>
            <boxGeometry args={[0.01, 0.01, 1]} />
            <meshStandardMaterial color="blue" />
          </mesh>
        </group>
        
        {/* Grid helper */}
        <gridHelper args={[10, 10]} />
        
        <OrbitControls 
          enablePan={true}
          enableZoom={true}
          enableRotate={true}
          minDistance={0.1}
          maxDistance={20}
        />
      </>
    );
  };

  return (
    <div className="App min-h-screen bg-gray-50 flex flex-col items-center justify-start py-8">
      <div className="w-full max-w-6xl">
        <header className="mb-8 text-center">
          <h1 className="text-3xl font-bold mb-2">NeRF Studio</h1>
          <p className="text-gray-500">Minimal Project Management</p>
        </header>
        <section className="mb-8 bg-white rounded-lg shadow p-6">
          <h2 className="text-xl font-semibold mb-4">Projects</h2>
          <form onSubmit={handleCreateProject} className="flex gap-2 mb-4">
            <input
              className="flex-1 border rounded px-3 py-2"
              placeholder="New project name"
              value={newProjectName}
              onChange={e => setNewProjectName(e.target.value)}
            />
            <button type="submit" className="bg-black text-white px-4 py-2 rounded hover:bg-gray-800">Create</button>
          </form>
          <div className="flex flex-wrap gap-2">
            {projects.map(proj => (
              <button
                key={proj.id}
                className={`px-3 py-2 rounded border ${selectedProject && selectedProject.id === proj.id ? 'bg-black text-white' : 'bg-gray-100 hover:bg-gray-200'}`}
                onClick={() => setSelectedProject(proj)}
              >
                {proj.name}
              </button>
            ))}
          </div>
        </section>
        {selectedProject && (
          <section className="mb-8 bg-white rounded-lg shadow p-6">
            {/* Project config panel */}
            <div className="mb-8">
              <h3 className="font-semibold mb-2">Project Settings</h3>
              {configEditing ? (
                <>
                  <textarea
                    className="w-full border rounded p-2 font-mono text-xs"
                    rows={6}
                    value={configEdit}
                    onChange={e => setConfigEdit(e.target.value)}
                  />
                  <div className="flex gap-2 mt-2">
                    <button className="bg-blue-600 text-white px-3 py-1 rounded" onClick={handleSaveConfig}>Save</button>
                    <button className="bg-gray-300 px-3 py-1 rounded" onClick={() => { setConfigEdit(JSON.stringify(projectConfig, null, 2)); setConfigEditing(false); }}>Cancel</button>
                  </div>
                </>
              ) : (
                <>
                  <pre className="bg-gray-100 rounded p-2 text-xs overflow-x-auto">{configEdit}</pre>
                  <button className="mt-2 bg-blue-600 text-white px-3 py-1 rounded" onClick={() => setConfigEditing(true)}>Edit</button>
                </>
              )}
            </div>
            <h2 className="text-xl font-semibold mb-4">Upload Images to "{selectedProject.name}"</h2>
            <div
              className={`border-2 border-dashed rounded-lg p-8 text-center transition-colors ${dragActive ? 'border-black bg-gray-100' : 'border-gray-300 bg-gray-50'}`}
              onDragEnter={handleDrag}
              onDragOver={handleDrag}
              onDragLeave={handleDrag}
              onDrop={handleDrop}
              onClick={() => fileInputRef.current && fileInputRef.current.click()}
              style={{ cursor: 'pointer' }}
            >
              <input
                type="file"
                multiple
                ref={fileInputRef}
                className="hidden"
                onChange={(e: ChangeEvent<HTMLInputElement>) => handleFiles(e.target.files)}
              />
              <div className="text-gray-500 mb-2">Drag & drop images here, or <span className="underline">browse</span></div>
              {uploading && <div className="text-blue-600 mt-2">Uploading...</div>}
            </div>
            <div className="mt-4">
              <h3 className="font-semibold mb-2">Images</h3>
              <div className="flex flex-wrap gap-2">
                {images.map(img => (
                  <div key={img.filename} className="w-20 h-20 bg-gray-200 rounded flex items-center justify-center text-xs text-gray-600">
                    {img.filename}
                  </div>
                ))}
                {images.length === 0 && <div className="text-gray-400">No images uploaded yet.</div>}
              </div>
            </div>
            {/* Image gallery with pose upload and metadata */}
            <div className="mt-4">
              <h3 className="font-semibold mb-2">Images & Camera Poses</h3>
              <div className="flex flex-wrap gap-4">
                {images.map(img => (
                  <div key={img.filename} className="w-32 bg-gray-100 rounded p-2 flex flex-col items-center">
                    <div className="text-xs text-gray-700 break-all mb-1">{img.filename}</div>
                    {img.pose ? (
                      <pre className="text-[10px] bg-gray-200 rounded p-1 mb-1">{JSON.stringify(img.pose)}</pre>
                    ) : (
                      <>
                        <textarea
                          className="w-full text-[10px] border rounded p-1 mb-1"
                          rows={2}
                          placeholder="Paste 16-number pose JSON"
                          value={poseInputs[img.filename] || ''}
                          onChange={e => handlePoseInputChange(img.filename, e.target.value)}
                        />
                        <button
                          className="bg-green-600 text-white px-2 py-1 rounded text-xs"
                          onClick={() => handleUploadPose(img.filename)}
                          disabled={poseUploading[img.filename]}
                        >{poseUploading[img.filename] ? 'Uploading...' : 'Upload Pose'}</button>
                      </>
                    )}
                  </div>
                ))}
                {images.length === 0 && <div className="text-gray-400">No images uploaded yet.</div>}
              </div>
            </div>
            
            {/* Enhanced Training Dashboard */}
            <div className="mt-8">
              <div className="flex justify-between items-center mb-4">
                <h3 className="font-semibold text-lg">Training Dashboard</h3>
                <button
                  className="text-blue-600 text-sm"
                  onClick={() => setShowTrainingConfig(!showTrainingConfig)}
                >
                  {showTrainingConfig ? 'Hide' : 'Show'} Training Config
                </button>
              </div>
              
              {/* Training Configuration */}
              {showTrainingConfig && (
                <div className="mb-6 p-4 bg-gray-50 rounded">
                  <h4 className="font-medium mb-3">Training Configuration</h4>
                  <div className="grid grid-cols-2 gap-4">
                    <div>
                      <label className="block text-sm font-medium mb-1">Batch Size</label>
                      <input
                        type="number"
                        className="w-full border rounded px-2 py-1"
                        value={trainingConfig.batch_size}
                        onChange={e => setTrainingConfig(c => ({ ...c, batch_size: parseInt(e.target.value) }))}
                      />
                    </div>
                    <div>
                      <label className="block text-sm font-medium mb-1">Learning Rate</label>
                      <input
                        type="number"
                        step="0.0001"
                        className="w-full border rounded px-2 py-1"
                        value={trainingConfig.lr}
                        onChange={e => setTrainingConfig(c => ({ ...c, lr: parseFloat(e.target.value) }))}
                      />
                    </div>
                    <div>
                      <label className="block text-sm font-medium mb-1">Epochs</label>
                      <input
                        type="number"
                        className="w-full border rounded px-2 py-1"
                        value={trainingConfig.num_epochs}
                        onChange={e => setTrainingConfig(c => ({ ...c, num_epochs: parseInt(e.target.value) }))}
                      />
                    </div>
                    <div>
                      <label className="block text-sm font-medium mb-1">Hidden Dim</label>
                      <input
                        type="number"
                        className="w-full border rounded px-2 py-1"
                        value={trainingConfig.hidden_dim}
                        onChange={e => setTrainingConfig(c => ({ ...c, hidden_dim: parseInt(e.target.value) }))}
                      />
                    </div>
                  </div>
                </div>
              )}
              
              {/* Training Controls */}
              <div className="flex gap-2 mb-4">
                <button
                  className="bg-green-600 text-white px-4 py-2 rounded hover:bg-green-700 disabled:opacity-50"
                  onClick={handleStartTraining}
                  disabled={uploading || !images.length || !!(job && jobStatus !== 'completed' && jobStatus !== 'cancelled' && jobStatus !== 'failed')}
                >
                  {job && jobStatus && jobStatus !== 'completed' && jobStatus !== 'cancelled' && jobStatus !== 'failed' ? 'Training in Progress...' : 'Start Training'}
                </button>
                
                {job && jobStatus === 'training' && (
                  <button
                    className="bg-red-600 text-white px-4 py-2 rounded hover:bg-red-700"
                    onClick={handleCancelTraining}
                  >
                    Cancel Training
                  </button>
                )}
              </div>
              
              {/* Real-time Training Metrics */}
              {job && (
                <div className="space-y-4">
                  {/* Progress Bar */}
                  <div>
                    <div className="flex justify-between text-sm text-gray-600 mb-1">
                      <span>Progress: {jobProgress}%</span>
                      <span>Status: {jobStatus}</span>
                    </div>
                    <div className="w-full bg-gray-200 rounded h-4 overflow-hidden">
                      <div
                        className="bg-green-500 h-4 transition-all"
                        style={{ width: `${jobProgress}%` }}
                      />
                    </div>
                  </div>
                  
                  {/* Metrics Display */}
                  {trainingMetrics && (
                    <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                      <div className="bg-blue-50 p-3 rounded">
                        <div className="text-sm text-blue-600">Step</div>
                        <div className="text-lg font-semibold">{trainingMetrics.step}</div>
                      </div>
                      <div className="bg-red-50 p-3 rounded">
                        <div className="text-sm text-red-600">Loss</div>
                        <div className="text-lg font-semibold">{trainingMetrics.loss.toFixed(4)}</div>
                      </div>
                      <div className="bg-green-50 p-3 rounded">
                        <div className="text-sm text-green-600">PSNR</div>
                        <div className="text-lg font-semibold">{trainingMetrics.psnr.toFixed(2)}</div>
                      </div>
                      <div className="bg-purple-50 p-3 rounded">
                        <div className="text-sm text-purple-600">Best PSNR</div>
                        <div className="text-lg font-semibold">{trainingMetrics.best_psnr.toFixed(2)}</div>
                      </div>
                    </div>
                  )}
                  
                  {/* ETA */}
                  {trainingMetrics && trainingMetrics.eta_seconds > 0 && (
                    <div className="text-sm text-gray-600">
                      Estimated time remaining: {Math.floor(trainingMetrics.eta_seconds / 60)}m {trainingMetrics.eta_seconds % 60}s
                    </div>
                  )}
                  
                  {/* System Resource Monitoring */}
                  {systemMetrics && (
                    <div className="mt-6">
                      <h4 className="font-medium mb-3">System Resources</h4>
                      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                        <div className="bg-blue-50 p-3 rounded">
                          <div className="text-sm text-blue-600">CPU</div>
                          <div className="text-lg font-semibold">{systemMetrics.cpu_percent?.toFixed(1) || 'N/A'}%</div>
                        </div>
                        <div className="bg-green-50 p-3 rounded">
                          <div className="text-sm text-green-600">Memory</div>
                          <div className="text-lg font-semibold">{systemMetrics.memory_percent?.toFixed(1) || 'N/A'}%</div>
                          <div className="text-xs text-gray-500">
                            {systemMetrics.memory_used_gb?.toFixed(1) || 'N/A'} / {systemMetrics.memory_total_gb?.toFixed(1) || 'N/A'} GB
                          </div>
                        </div>
                        {systemMetrics.gpu_memory_used_gb && (
                          <div className="bg-purple-50 p-3 rounded">
                            <div className="text-sm text-purple-600">GPU Memory</div>
                            <div className="text-lg font-semibold">
                              {systemMetrics.gpu_memory_used_gb?.toFixed(1) || 'N/A'} / {systemMetrics.gpu_memory_total_gb?.toFixed(1) || 'N/A'} GB
                            </div>
                          </div>
                        )}
                        {systemMetrics.gpu_temperature && (
                          <div className="bg-orange-50 p-3 rounded">
                            <div className="text-sm text-orange-600">GPU Temp</div>
                            <div className="text-lg font-semibold">{systemMetrics.gpu_temperature?.toFixed(1) || 'N/A'}°C</div>
                          </div>
                        )}
                      </div>
                    </div>
                  )}

                  {/* Training Performance Summary */}
                  {trainingSummary && trainingSummary.total_steps > 0 && (
                    <div className="mt-6">
                      <h4 className="font-medium mb-3">Training Performance</h4>
                      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                        <div className="bg-gray-50 p-3 rounded">
                          <div className="text-sm text-gray-600">Total Steps</div>
                          <div className="text-lg font-semibold">{trainingSummary.total_steps}</div>
                        </div>
                        <div className="bg-gray-50 p-3 rounded">
                          <div className="text-sm text-gray-600">Training Time</div>
                          <div className="text-lg font-semibold">{Math.floor(trainingSummary.total_time_seconds / 60)}m {Math.floor(trainingSummary.total_time_seconds % 60)}s</div>
                        </div>
                        <div className="bg-gray-50 p-3 rounded">
                          <div className="text-sm text-gray-600">Speed</div>
                          <div className="text-lg font-semibold">{trainingSummary.current_speed?.toFixed(2) || 'N/A'} steps/s</div>
                        </div>
                        <div className="bg-gray-50 p-3 rounded">
                          <div className="text-sm text-gray-600">Avg Speed</div>
                          <div className="text-lg font-semibold">{trainingSummary.average_speed?.toFixed(2) || 'N/A'} steps/s</div>
                        </div>
                      </div>
                    </div>
                  )}

                  {/* Chart.js Loss Curve */}
                  {chartData.labels.length > 0 && (
                    <div className="mt-6">
                      <h4 className="font-medium mb-2">Training Metrics</h4>
                      <div className="h-64 bg-white rounded border">
                        <Line data={chartData} options={chartOptions} />
                      </div>
                    </div>
                  )}
                </div>
              )}
            </div>
            
            {/* Mesh Extraction Section */}
            <div className="mt-8">
              <h3 className="font-semibold mb-4">Mesh Extraction</h3>
              <div className="bg-gray-50 p-4 rounded-lg">
                <p className="text-sm text-gray-600 mb-4">
                  Extract 3D mesh from trained NeRF model. Supports GLTF, OBJ, and PLY formats.
                </p>
                
                <div className="flex gap-2 mb-4">
                  <button
                    onClick={extractMesh}
                    disabled={meshExtracting || !job || job.status !== 'completed'}
                    className="px-4 py-2 bg-blue-600 text-white rounded hover:bg-blue-700 disabled:bg-gray-400"
                  >
                    {meshExtracting ? 'Extracting...' : 'Extract Mesh'}
                  </button>
                  
                  {Object.keys(meshFiles).length > 0 && (
                    <button
                      onClick={downloadAllMeshes}
                      className="px-4 py-2 bg-green-600 text-white rounded hover:bg-green-700"
                    >
                      Download All Formats
                    </button>
                  )}
                </div>
                
                {Object.keys(meshFiles).length > 0 && (
                  <div className="space-y-2">
                    <h4 className="font-medium">Available Formats:</h4>
                    <div className="flex gap-2 flex-wrap">
                      {Object.entries(meshFiles).map(([format, path]) => (
                        <button
                          key={format}
                          onClick={() => downloadMesh(format)}
                          className="px-3 py-1 bg-gray-200 text-gray-800 rounded hover:bg-gray-300 text-sm"
                        >
                          Download {format.toUpperCase()}
                        </button>
                      ))}
                    </div>
                  </div>
                )}
                
                {(!job || job.status !== 'completed') && (
                  <p className="text-sm text-orange-600 mt-2">
                    ⚠️ Train the model first to extract mesh
                  </p>
                )}
              </div>
            </div>

            {/* Enhanced 3D Viewer */}
            <div className="mt-8">
              <h3 className="font-semibold mb-2">3D Scene Viewer</h3>
              <div className="h-96 border border-gray-200 rounded bg-gray-900">
                <Canvas camera={{ position: [3, 3, 3], fov: 50 }}>
                  <Scene3D />
                </Canvas>
              </div>
              <div className="mt-2 text-sm text-gray-600">
                {cameraCenters.length > 0 ? 
                  `${cameraCenters.length} camera positions loaded` : 
                  'No camera poses available. Upload poses to see 3D visualization.'
                }
              </div>
            </div>
          </section>
        )}
      </div>
    </div>
  );
}

export default App;