
import { useState, useEffect, useRef } from 'react';
import PerformanceDashboard from './components/PerformanceDashboard';
import AdvancedExportManager from './components/AdvancedExportManager';
import AdvancedNeRFViewer from './components/AdvancedNeRFViewer';
import './App.css';

const API_URL = 'http://localhost:8000/api/v1';

interface Project {
  id: string;
  name: string;
  status: string;
  created_at: string;
}

interface TrainingMetrics {
  step: number;
  loss: number;
  psnr: number;
  lr: number;
}

interface SystemMetrics {
  cpu_percent: number;
  memory_percent: number;
  gpu_utilization: number;
}

interface TrainingLogEntry {
  step?: number;
  loss?: number;
  psnr?: number;
  progress?: number;
  status?: string;
  message?: string;
}

function App() {
  const [projects, setProjects] = useState<Project[]>([]);
  const [selectedProject, setSelectedProject] = useState<string | null>(null);
  const [jobId, setJobId] = useState<string | null>(null);
  const [trainingLog, setTrainingLog] = useState<TrainingLogEntry[]>([]);
  const [trainingProgress, setTrainingProgress] = useState<number>(0);
  const [systemMetrics, setSystemMetrics] = useState<SystemMetrics | null>(null);
  const [trainingMetrics, setTrainingMetrics] = useState<TrainingMetrics | null>(null);
  const [renderedImage, setRenderedImage] = useState<string | null>(null);
  const ws = useRef<WebSocket | null>(null);

  useEffect(() => {
    fetchProjects();
    const interval = setInterval(() => {
      fetchSystemMetrics();
      fetchTrainingMetrics();
    }, 5000);
    return () => clearInterval(interval);
  }, []);

  useEffect(() => {
    if (jobId) {
      ws.current = new WebSocket(`${API_URL.replace('http', 'ws')}/ws/jobs/${jobId}`);
      ws.current.onmessage = (event: MessageEvent) => {
        const data = JSON.parse(event.data);
        setTrainingLog(prev => [...prev, data]);
        if (data.progress) {
          setTrainingProgress(data.progress);
        }
      };
      return () => {
        if (ws.current) {
          ws.current.close();
        }
      };
    }
  }, [jobId]);

  const fetchProjects = async () => {
    const response = await fetch(`${API_URL}/projects`);
    const data = await response.json();
    // If the response is an array, use as is. If it's an object with 'projects', use that.
    setProjects(Array.isArray(data) ? data : data.projects || []);
  };

  const createProject = async (name: string) => {
    const response = await fetch(`${API_URL}/projects`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ name, description: '' }),
    });
    if (response.ok) {
      // The backend now returns { project: { ... } }
      const result = await response.json();
      if (result && result.project) {
        setProjects(prev => [...prev, result.project]);
      } else {
        fetchProjects();
      }
    }
  };

  const uploadImages = async (projectId: string, files: File[]) => {
    const formData = new FormData();
    files.forEach(file => formData.append('files', file));
    await fetch(`${API_URL}/projects/${projectId}/upload_images`, {
      method: 'POST',
      body: formData,
    });
  };

  const startTraining = async (projectId: string) => {
    const response = await fetch(`${API_URL}/projects/${projectId}/start_training`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({}),
    });
    const data = await response.json();
    setJobId(data.id);
  };

  const fetchSystemMetrics = async () => {
    try {
      const response = await fetch(`${API_URL}/system/metrics`);
      const data = await response.json();
      setSystemMetrics(data);
    } catch (error) {
      console.error('Failed to fetch system metrics:', error);
    }
  };

  const fetchTrainingMetrics = async () => {
    if (jobId) {
      try {
        const response = await fetch(`${API_URL}/jobs/${jobId}/metrics`);
        const data = await response.json();
        setTrainingMetrics(data);
      } catch (error) {
        console.error('Failed to fetch training metrics:', error);
      }
    }
  };

  const renderNovelView = async (projectId: string) => {
    try {
      // Dummy pose for now - 4x4 transformation matrix
      const pose = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 2.5, 0, 0, 0, 1];
      const response = await fetch(`${API_URL}/projects/${projectId}/render`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ pose: pose, resolution: 400 }),
      });
      const data = await response.json();
      setRenderedImage(data.image);
    } catch (error) {
      console.error('Failed to render novel view:', error);
    }
  };

  return (
    <div className="min-h-screen bg-gray-50">
      <header className="bg-white shadow-sm border-b">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between items-center py-6">
            <h1 className="text-3xl font-bold text-gray-900">NeRF Studio</h1>
            <div className="text-sm text-gray-500">
              Neural Radiance Fields Platform
            </div>
          </div>
        </div>
      </header>
      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        <div className="project-section bg-white p-6 rounded-lg shadow-md mb-6">
          <h2 className="text-2xl font-bold mb-4">Projects</h2>
          <div className="mb-4">
            <input
              type="text"
              placeholder="Enter project name..."
              className="border border-gray-300 rounded px-3 py-2 mr-2"
              onKeyDown={(e) => {
                if (e.key === 'Enter') {
                  const target = e.target as HTMLInputElement;
                  if (target.value.trim()) {
                    createProject(target.value.trim());
                    target.value = '';
                  }
                }
              }}
            />
            <button
              onClick={() => {
                const input = document.querySelector('input[placeholder="Enter project name..."]') as HTMLInputElement;
                if (input && input.value.trim()) {
                  createProject(input.value.trim());
                  input.value = '';
                }
              }}
              className="bg-blue-500 text-white px-4 py-2 rounded hover:bg-blue-600"
            >
              Create Project
            </button>
          </div>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
            {projects.map(p => (
              <div
                key={p.id}
                onClick={() => setSelectedProject(p.id)}
                className={`p-4 border rounded-lg cursor-pointer transition-colors ${selectedProject === p.id
                  ? 'border-blue-500 bg-blue-50'
                  : 'border-gray-200 hover:border-gray-300'
                  }`}
              >
                <h3 className="font-semibold">{p.name}</h3>
                <p className="text-sm text-gray-600">Status: {p.status}</p>
                <p className="text-xs text-gray-500">Created: {new Date(p.created_at).toLocaleDateString()}</p>
              </div>
            ))}
          </div>
        </div>

        {selectedProject && (
          <div className="details-section bg-white p-6 rounded-lg shadow-md mb-6">
            <h2 className="text-2xl font-bold mb-4">Project: {projects.find(p => p.id === selectedProject)?.name}</h2>

            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              <div>
                <h3 className="text-lg font-semibold mb-3">Upload Images</h3>
                <input
                  type="file"
                  multiple
                  accept="image/*"
                  onChange={(e) => {
                    if (e.target.files && selectedProject) {
                      uploadImages(selectedProject, Array.from(e.target.files));
                    }
                  }}
                  className="block w-full text-sm text-gray-500 file:mr-4 file:py-2 file:px-4 file:rounded-full file:border-0 file:text-sm file:font-semibold file:bg-blue-50 file:text-blue-700 hover:file:bg-blue-100"
                />

                <div className="mt-4 space-x-2">
                  <button
                    onClick={() => startTraining(selectedProject)}
                    className="bg-green-500 text-white px-4 py-2 rounded hover:bg-green-600"
                    disabled={trainingProgress > 0 && trainingProgress < 100}
                  >
                    {trainingProgress > 0 && trainingProgress < 100 ? 'Training...' : 'Start Training'}
                  </button>
                  <button
                    onClick={() => renderNovelView(selectedProject)}
                    className="bg-purple-500 text-white px-4 py-2 rounded hover:bg-purple-600"
                  >
                    Render View
                  </button>
                </div>
              </div>

              <div>
                <h3 className="text-lg font-semibold mb-3">Training Progress: {trainingProgress}%</h3>
                <div className="w-full bg-gray-200 rounded-full h-2.5 mb-4">
                  <div
                    className="bg-blue-600 h-2.5 rounded-full transition-all duration-300"
                    style={{ width: `${trainingProgress}%` }}
                  ></div>
                </div>

                {renderedImage && (
                  <div>
                    <h4 className="font-semibold mb-2">Rendered View:</h4>
                    <img src={renderedImage} alt="Rendered NeRF" className="max-w-full h-auto rounded border" />
                  </div>
                )}
              </div>
            </div>

            {trainingLog.length > 0 && (
              <div className="mt-6">
                <h3 className="text-lg font-semibold mb-3">Training Log</h3>
                <div className="bg-gray-100 p-4 rounded max-h-64 overflow-y-auto">
                  {trainingLog.slice(-10).map((log, i) => (
                    <div key={i} className="text-sm mb-2 font-mono">
                      {JSON.stringify(log, null, 2)}
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        )}

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          <PerformanceDashboard systemMetrics={systemMetrics} trainingMetrics={trainingMetrics} />

          {selectedProject && (
            <AdvancedExportManager projectId={selectedProject} />
          )}
        </div>

        {selectedProject && (
          <div className="mt-6">
            <AdvancedNeRFViewer projectId={selectedProject} />
          </div>
        )}
      </main>
    </div>
  );
}

export default App;
