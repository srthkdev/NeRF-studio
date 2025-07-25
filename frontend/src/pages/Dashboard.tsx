import { useState, useEffect, useRef } from 'react';
import { motion } from 'framer-motion';
import { 
  Plus, 
  Upload, 
  Play, 
  Eye, 
  Settings, 
  BarChart3, 
  Activity,
  CheckCircle,
  Clock,
  AlertCircle,
  Image,
  FileText
} from 'lucide-react';
import PerformanceDashboard from '../components/PerformanceDashboard';
import AdvancedExportManager from '../components/AdvancedExportManager';
import AdvancedNeRFViewer from '../components/AdvancedNeRFViewer';
import TrainingView from '../components/TrainingView';

const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000/api/v1';

interface Project {
  id: string;
  name: string;
  status?: string;
  created_at: string;
  data?: {
    images?: string[];
    project_dir?: string;
  };
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

const Dashboard = () => {
  const [projects, setProjects] = useState<Project[]>([]);
  const [selectedProject, setSelectedProject] = useState<string | null>(null);
  const [jobId, setJobId] = useState<string | null>(null);
  const [trainingLog, setTrainingLog] = useState<TrainingLogEntry[]>([]);
  const [trainingProgress, setTrainingProgress] = useState<number>(0);
  const [systemMetrics, setSystemMetrics] = useState<SystemMetrics | null>(null);
  const [trainingMetrics, setTrainingMetrics] = useState<TrainingMetrics | null>(null);
  const [renderedImage, setRenderedImage] = useState<string | null>(null);
  const [newProjectName, setNewProjectName] = useState('');
  const [isCreatingProject, setIsCreatingProject] = useState(false);
  const [isLoadingProjects, setIsLoadingProjects] = useState(true);
  const [projectsError, setProjectsError] = useState<string | null>(null);
  const [isEstimatingPoses, setIsEstimatingPoses] = useState(false);

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
    setIsLoadingProjects(true);
    setProjectsError(null);
    try {
      const response = await fetch(`${API_URL}/projects`);
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      const data = await response.json();
      setProjects(Array.isArray(data) ? data : data.projects || []);
    } catch (error) {
      console.error('Failed to fetch projects:', error);
      setProjects([]); // Set empty array on error
      setProjectsError('Failed to load projects. Please try again.');
    } finally {
      setIsLoadingProjects(false);
    }
  };

  const createProject = async (name: string) => {
    if (!name.trim()) return;
    
    setIsCreatingProject(true);
    try {
      const response = await fetch(`${API_URL}/projects`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ name, description: '' }),
      });
      if (response.ok) {
        const result = await response.json();
        if (result && result.project) {
          setProjects(prev => [...prev, result.project]);
          setNewProjectName('');
        } else {
          fetchProjects();
        }
      }
    } catch (error) {
      console.error('Failed to create project:', error);
    } finally {
      setIsCreatingProject(false);
    }
  };

  const uploadImages = async (projectId: string, files: File[]) => {
    try {
      const formData = new FormData();
      files.forEach(file => formData.append('files', file));
      
      const response = await fetch(`${API_URL}/projects/${projectId}/upload_images`, {
        method: 'POST',
        body: formData,
      });
      
      if (response.ok) {
        const result = await response.json();
        console.log('Images uploaded successfully:', result);
        alert(`Successfully uploaded ${result.files.length} images!`);
        
        // Refresh the projects list to show updated data
        fetchProjects();
      } else {
        const errorData = await response.json();
        throw new Error(errorData.detail || 'Failed to upload images');
      }
    } catch (error) {
      console.error('Failed to upload images:', error);
      const errorMessage = error instanceof Error ? error.message : 'Unknown error occurred';
      alert(`Upload failed: ${errorMessage}`);
    }
  };

  const startTraining = async (projectId: string) => {
    try {
      const response = await fetch(`${API_URL}/projects/${projectId}/start_training`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({}),
      });
      
      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || 'Failed to start training');
      }
      
      const data = await response.json();
      setJobId(data.id);
    } catch (error) {
      console.error('Training start failed:', error);
      // You could add a toast notification here
      const errorMessage = error instanceof Error ? error.message : 'Unknown error occurred';
      alert(`Training failed to start: ${errorMessage}`);
    }
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

  const estimatePoses = async (projectId: string) => {
    setIsEstimatingPoses(true);
    try {
      const response = await fetch(`${API_URL}/projects/${projectId}/estimate_poses`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
      });
      
      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || 'Failed to estimate poses');
      }
      
      alert('Camera poses estimated successfully! You can now start training.');
      // Refresh projects to show updated status
      fetchProjects();
    } catch (error) {
      console.error('Pose estimation failed:', error);
      const errorMessage = error instanceof Error ? error.message : 'Unknown error occurred';
      alert(`Pose estimation failed: ${errorMessage}`);
    } finally {
      setIsEstimatingPoses(false);
    }
  };

  const renderNovelView = async (projectId: string) => {
    try {
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

  const getStatusIcon = (status: string | undefined | null) => {
    if (!status) {
      return <AlertCircle className="w-4 h-4 text-gray-500" />;
    }
    
    switch (status.toLowerCase()) {
      case 'completed':
        return <CheckCircle className="w-4 h-4 text-green-500" />;
      case 'training':
        return <Activity className="w-4 h-4 text-blue-500 animate-pulse" />;
      case 'pending':
        return <Clock className="w-4 h-4 text-yellow-500" />;
      default:
        return <AlertCircle className="w-4 h-4 text-gray-500" />;
    }
  };



  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
      {/* Hero Section */}
      <motion.div 
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        className="text-center mb-12"
      >
        <h1 className="text-4xl font-bold text-gray-900 mb-4">
          Welcome to NeRF Studio
        </h1>
        <p className="text-xl text-gray-600 max-w-2xl mx-auto">
          Create stunning 3D scene reconstructions using Neural Radiance Fields. 
          Upload your images and watch as AI generates interactive 3D environments.
        </p>
      </motion.div>

      {/* Projects Section */}
      <motion.div 
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.1 }}
        className="bg-white rounded-2xl shadow-lg p-8 mb-8"
      >
        <div className="flex justify-between items-center mb-6">
          <h2 className="text-2xl font-bold text-gray-900">Your Projects</h2>
          <div className="flex items-center space-x-4">
            <div className="flex-1">
              <input
                type="text"
                placeholder="Enter project name..."
                value={newProjectName}
                onChange={(e) => {
                  console.log('Input changed:', e.target.value);
                  setNewProjectName(e.target.value);
                }}
                onKeyDown={(e) => {
                  if (e.key === 'Enter') {
                    createProject(newProjectName);
                  }
                }}
                className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
              />
              {newProjectName && (
                <p className="text-sm text-gray-600 mt-1">
                  Project name: "{newProjectName}" - Button should be {!newProjectName.trim() ? 'disabled' : 'clickable'}
                </p>
              )}
            </div>
            <button
              onClick={() => {
                console.log('Button clicked!', { newProjectName, isCreatingProject });
                createProject(newProjectName);
              }}
              disabled={isCreatingProject || !newProjectName.trim()}
              className={`flex items-center space-x-2 px-4 py-2 rounded-lg transition-colors font-medium shadow-md hover:shadow-lg relative z-10 ${
                isCreatingProject || !newProjectName.trim()
                  ? 'bg-gray-400 text-gray-600 cursor-not-allowed opacity-50'
                  : 'bg-blue-600 text-white hover:bg-blue-700 cursor-pointer'
              }`}
            >
              <Plus size={18} />
              <span>{isCreatingProject ? 'Creating...' : 'Create Project'}</span>
            </button>
          </div>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          {isLoadingProjects ? (
            <div className="col-span-full flex items-center justify-center py-12">
              <div className="text-center">
                <div className="spinner w-8 h-8 mx-auto mb-4"></div>
                <p className="text-gray-600">Loading projects...</p>
              </div>
            </div>
          ) : projectsError ? (
            <div className="col-span-full flex items-center justify-center py-12">
              <div className="text-center">
                <AlertCircle className="w-12 h-12 text-red-500 mx-auto mb-4" />
                <p className="text-red-600 mb-4">{projectsError}</p>
                <button
                  onClick={fetchProjects}
                  className="btn-secondary"
                >
                  Try Again
                </button>
              </div>
            </div>
          ) : projects.length === 0 ? (
            <div className="col-span-full flex items-center justify-center py-12">
              <div className="text-center">
                <div className="w-16 h-16 bg-gray-100 rounded-full flex items-center justify-center mx-auto mb-4">
                  <Plus className="w-8 h-8 text-gray-400" />
                </div>
                <p className="text-gray-600 mb-2">No projects yet</p>
                <p className="text-gray-500 text-sm">Create your first project to get started</p>
              </div>
            </div>
          ) : (
            projects.map((project) => (
              <motion.div
                key={project.id}
                whileHover={{ scale: 1.02 }}
                whileTap={{ scale: 0.98 }}
                onClick={() => setSelectedProject(project.id)}
                className={`p-6 border-2 rounded-xl cursor-pointer transition-all duration-200 ${
                  selectedProject === project.id
                    ? 'border-blue-500 bg-blue-50 shadow-lg'
                    : 'border-gray-200 hover:border-blue-300 hover:shadow-md'
                }`}
              >
                <div className="flex items-center justify-between mb-3">
                  <h3 className="font-semibold text-lg text-gray-900">{project.name || 'Unnamed Project'}</h3>
                  {getStatusIcon(project.status)}
                </div>
                <p className="text-sm text-gray-600 mb-2">Status: {project.status || 'Unknown'}</p>
                <p className="text-xs text-gray-500">
                  Created: {project.created_at ? new Date(project.created_at).toLocaleDateString() : 'Unknown'}
                </p>
              </motion.div>
            ))
          )}
        </div>
      </motion.div>

      {/* Project Details Section */}
      {selectedProject && (
        <motion.div 
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.2 }}
          className="bg-white rounded-2xl shadow-lg p-8 mb-8"
        >
          <div className="flex items-center justify-between mb-6">
            <h2 className="text-2xl font-bold text-gray-900">
              Project: {projects.find(p => p.id === selectedProject)?.name}
            </h2>
            <div className="flex items-center space-x-3">
              <button
                onClick={() => estimatePoses(selectedProject)}
                disabled={isEstimatingPoses}
                className="flex items-center space-x-2 px-4 py-2 bg-orange-600 text-white rounded-lg hover:bg-orange-700 disabled:opacity-50 transition-colors"
              >
                <Settings size={18} />
                <span>{isEstimatingPoses ? 'Estimating...' : 'Estimate Poses'}</span>
              </button>
              <button
                onClick={() => startTraining(selectedProject)}
                disabled={trainingProgress > 0 && trainingProgress < 100}
                className="flex items-center space-x-2 px-4 py-2 bg-green-600 text-white rounded-lg hover:bg-green-700 disabled:opacity-50 transition-colors"
              >
                <Play size={18} />
                <span>{trainingProgress > 0 && trainingProgress < 100 ? 'Training...' : 'Start Training'}</span>
              </button>
              <button
                onClick={() => renderNovelView(selectedProject)}
                className="flex items-center space-x-2 px-4 py-2 bg-purple-600 text-white rounded-lg hover:bg-purple-700 transition-colors"
              >
                <Eye size={18} />
                <span>Render View</span>
              </button>
              <button
                onClick={() => window.open('https://github.com/srthkdev/NeRF-studio/blob/main/test_results.md', '_blank')}
                className="flex items-center space-x-2 px-4 py-2 bg-indigo-600 text-white rounded-lg hover:bg-indigo-700 transition-colors"
              >
                <FileText size={18} />
                <span>View Test Cases</span>
              </button>
            </div>
          </div>

          <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
                          {/* Upload Section */}
              <div className="space-y-6">
                <div>
                  <h3 className="text-lg font-semibold mb-3 flex items-center space-x-2">
                    <Upload size={20} />
                    <span>Upload Images</span>
                  </h3>
                  <div className="border-2 border-dashed border-gray-300 rounded-lg p-6 text-center hover:border-blue-400 transition-colors">
                    <input
                      type="file"
                      multiple
                      accept="image/*"
                      onChange={(e) => {
                        if (e.target.files && selectedProject) {
                          uploadImages(selectedProject, Array.from(e.target.files));
                        }
                      }}
                      className="hidden"
                      id="image-upload"
                    />
                    <label htmlFor="image-upload" className="cursor-pointer">
                      <Upload size={32} className="mx-auto text-gray-400 mb-2" />
                      <p className="text-gray-600">Click to upload images or drag and drop</p>
                      <p className="text-sm text-gray-500 mt-1">Supports JPG, PNG, JPEG</p>
                    </label>
                  </div>
                  
                  {/* Training Prerequisites */}
                  <div className="mt-4 bg-blue-50 border border-blue-200 rounded-lg p-4">
                    <h4 className="font-semibold text-blue-800 mb-2 flex items-center">
                      <AlertCircle className="w-4 h-4 mr-2" />
                      Before Training
                    </h4>
                    <ul className="text-blue-700 text-sm space-y-1">
                      <li>• Upload 20-100 high-quality images from different angles</li>
                      <li>• Images should have good overlap between consecutive shots</li>
                      <li>• Maintain consistent lighting throughout</li>
                      <li>• Click "Estimate Poses" after uploading images</li>
                      <li>• Then click "Start Training" to begin NeRF training</li>
                    </ul>
                  </div>

                  {/* Uploaded Images Display */}
                  {selectedProject && projects.find(p => p.id === selectedProject)?.data?.images && (
                    <div className="mt-4">
                      <h4 className="font-semibold text-gray-800 mb-2 flex items-center">
                        <Image className="w-4 h-4 mr-2" />
                        Uploaded Images ({projects.find(p => p.id === selectedProject)?.data?.images?.length || 0})
                      </h4>
                      <div className="grid grid-cols-4 gap-2 max-h-32 overflow-y-auto">
                        {projects.find(p => p.id === selectedProject)?.data?.images?.map((image: string, index: number) => (
                          <div key={index} className="bg-gray-100 rounded p-1 text-xs text-center truncate">
                            {image}
                          </div>
                        ))}
                      </div>
                    </div>
                  )}
                </div>

              {/* Training Progress */}
              <div>
                <h3 className="text-lg font-semibold mb-3 flex items-center space-x-2">
                  <BarChart3 size={20} />
                  <span>Training Progress</span>
                </h3>
                <div className="space-y-3">
                  <div className="flex justify-between text-sm">
                    <span className="text-gray-600">Progress</span>
                    <span className="font-semibold">{trainingProgress}%</span>
                  </div>
                  <div className="w-full bg-gray-200 rounded-full h-3">
                    <motion.div
                      className="bg-gradient-to-r from-blue-500 to-purple-600 h-3 rounded-full"
                      initial={{ width: 0 }}
                      animate={{ width: `${trainingProgress}%` }}
                      transition={{ duration: 0.5 }}
                    />
                  </div>
                </div>
              </div>
            </div>

            {/* Rendered Image */}
            <div>
              <h3 className="text-lg font-semibold mb-3 flex items-center space-x-2">
                <Eye size={20} />
                <span>Rendered View</span>
              </h3>
              {renderedImage ? (
                <div className="border rounded-lg overflow-hidden">
                  <img 
                    src={renderedImage} 
                    alt="Rendered NeRF" 
                    className="w-full h-auto"
                  />
                </div>
              ) : (
                <div className="border-2 border-dashed border-gray-300 rounded-lg p-8 text-center">
                  <Eye size={48} className="mx-auto text-gray-400 mb-4" />
                  <p className="text-gray-600">No rendered view available</p>
                  <p className="text-sm text-gray-500 mt-1">Click "Render View" to generate</p>
                </div>
              )}
            </div>
          </div>
        </motion.div>
      )}

      {/* Training View - Now prominently displayed */}
      {selectedProject && (trainingProgress > 0 || trainingLog.length > 0) && (
        <TrainingView 
          trainingProgress={trainingProgress}
          trainingLog={trainingLog}
          trainingMetrics={trainingMetrics}
        />
      )}

      {/* Performance and Export Sections */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 mb-8">
        <PerformanceDashboard systemMetrics={systemMetrics} trainingMetrics={trainingMetrics} />
        {selectedProject && <AdvancedExportManager projectId={selectedProject} />}
      </div>

      {/* NeRF Demo GIF */}
      <motion.div 
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.2 }}
        className="bg-white rounded-2xl shadow-lg p-8 mb-8"
      >
        <h2 className="text-2xl font-bold text-gray-900 mb-6">🎬 NeRF Demo Preview</h2>
        <div className="flex justify-center">
          <img 
            src="https://bmild.github.io/nerf/fern_200k_256w.gif" 
            alt="NeRF Demo - Fern Scene" 
            className="rounded-lg shadow-lg max-w-full h-auto"
            style={{ maxHeight: '400px' }}
          />
        </div>
        <p className="text-center text-gray-600 mt-4 text-sm">
          Example of NeRF novel view synthesis - rotating around a fern scene
        </p>
      </motion.div>

      {/* 3D Viewer */}
      {selectedProject && (
        <motion.div 
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.3 }}
          className="bg-white rounded-2xl shadow-lg p-8"
        >
          <h2 className="text-2xl font-bold text-gray-900 mb-6">3D Scene Viewer</h2>
          <AdvancedNeRFViewer projectId={selectedProject} />
        </motion.div>
      )}
    </div>
  );
};

export default Dashboard; 