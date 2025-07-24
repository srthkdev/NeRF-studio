import { useState } from 'react';
import { useParams, Link } from 'react-router-dom';
import { motion } from 'framer-motion';
import { 
  BookOpen, 
  Code, 
  Play, 
  Settings, 
  Download, 
  Eye, 
  Upload,
  Lightbulb,
  AlertTriangle,
  CheckCircle,
  ExternalLink,
  ChevronRight,
  ChevronDown
} from 'lucide-react';

const Documentation = () => {
  const { section } = useParams();
  const [expandedSections, setExpandedSections] = useState<string[]>(['getting-started']);

  const toggleSection = (sectionId: string) => {
    setExpandedSections(prev => 
      prev.includes(sectionId) 
        ? prev.filter(id => id !== sectionId)
        : [...prev, sectionId]
    );
  };

  const sections = [
    {
      id: 'getting-started',
      title: 'Getting Started',
      icon: Play,
      content: (
        <div className="space-y-6">
          <div>
            <h3 className="text-xl font-semibold mb-3">Quick Start Guide</h3>
            <div className="bg-gray-50 rounded-lg p-4 space-y-3">
              <div className="flex items-start space-x-3">
                <div className="w-6 h-6 bg-blue-500 text-white rounded-full flex items-center justify-center text-sm font-bold">1</div>
                <div>
                  <p className="font-medium">Create a New Project</p>
                  <p className="text-gray-600 text-sm">Click "Create Project" and give your project a name</p>
                </div>
              </div>
              <div className="flex items-start space-x-3">
                <div className="w-6 h-6 bg-blue-500 text-white rounded-full flex items-center justify-center text-sm font-bold">2</div>
                <div>
                  <p className="font-medium">Upload Images</p>
                  <p className="text-gray-600 text-sm">Upload 20-100 high-quality images of your scene from different angles</p>
                </div>
              </div>
              <div className="flex items-start space-x-3">
                <div className="w-6 h-6 bg-blue-500 text-white rounded-full flex items-center justify-center text-sm font-bold">3</div>
                <div>
                  <p className="font-medium">Start Training</p>
                  <p className="text-gray-600 text-sm">Click "Start Training" and monitor progress in real-time</p>
                </div>
              </div>
              <div className="flex items-start space-x-3">
                <div className="w-6 h-6 bg-blue-500 text-white rounded-full flex items-center justify-center text-sm font-bold">4</div>
                <div>
                  <p className="font-medium">Explore & Export</p>
                  <p className="text-gray-600 text-sm">View your 3D scene and export in various formats</p>
                </div>
              </div>
            </div>
          </div>

          <div>
            <h3 className="text-xl font-semibold mb-3">System Requirements</h3>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div className="bg-green-50 border border-green-200 rounded-lg p-4">
                <h4 className="font-semibold text-green-800 mb-2">Minimum Requirements</h4>
                <ul className="text-sm text-green-700 space-y-1">
                  <li>• 8GB RAM</li>
                  <li>• 4GB VRAM (GPU)</li>
                  <li>• 10GB free disk space</li>
                  <li>• Modern web browser</li>
                </ul>
              </div>
              <div className="bg-blue-50 border border-blue-200 rounded-lg p-4">
                <h4 className="font-semibold text-blue-800 mb-2">Recommended</h4>
                <ul className="text-sm text-blue-700 space-y-1">
                  <li>• 16GB+ RAM</li>
                  <li>• 8GB+ VRAM (GPU)</li>
                  <li>• 50GB+ free disk space</li>
                  <li>• CUDA-compatible GPU</li>
                </ul>
              </div>
            </div>
          </div>
        </div>
      )
    },
    {
      id: 'nerf-theory',
      title: 'NeRF Theory',
      icon: BookOpen,
      content: (
        <div className="space-y-6">
          <div>
            <h3 className="text-xl font-semibold mb-3">What are Neural Radiance Fields?</h3>
            <p className="text-gray-700 mb-4">
              Neural Radiance Fields (NeRF) represent a scene as a continuous 5D function that outputs the volume density 
              and view-dependent emitted radiance at any point in 3D space. This allows for photorealistic novel view 
              synthesis from a sparse set of input views.
            </p>
            <div className="bg-blue-50 border border-blue-200 rounded-lg p-4">
              <h4 className="font-semibold text-blue-800 mb-2">Key Concepts</h4>
              <ul className="text-blue-700 space-y-2">
                <li><strong>Volume Rendering:</strong> Uses classical volume rendering techniques to project 3D densities and colors into 2D images</li>
                <li><strong>Positional Encoding:</strong> Fourier feature encoding enables the network to represent high-frequency functions</li>
                <li><strong>Hierarchical Sampling:</strong> Two-stage sampling strategy for efficient rendering</li>
                <li><strong>View Dependence:</strong> Models view-dependent effects like specular reflections</li>
              </ul>
            </div>
          </div>

          <div>
            <h3 className="text-xl font-semibold mb-3">Mathematical Foundation</h3>
            <div className="bg-gray-50 rounded-lg p-4 font-mono text-sm">
              <p className="mb-2">The NeRF function F maps 5D coordinates to density and color:</p>
              <p className="text-blue-600 mb-4">F: (x, y, z, θ, φ) → (σ, c)</p>
              <p className="mb-2">Where:</p>
              <ul className="space-y-1 text-gray-700">
                <li>• (x, y, z) = 3D spatial location</li>
                <li>• (θ, φ) = 2D viewing direction</li>
                <li>• σ = volume density</li>
                <li>• c = view-dependent emitted radiance</li>
              </ul>
            </div>
          </div>

          <div>
            <h3 className="text-xl font-semibold mb-3">Training Process</h3>
            <div className="space-y-4">
              <div className="flex items-start space-x-3">
                <div className="w-8 h-8 bg-purple-500 text-white rounded-full flex items-center justify-center text-sm font-bold">1</div>
                <div>
                  <p className="font-medium">Ray Sampling</p>
                  <p className="text-gray-600 text-sm">Sample points along camera rays through the scene</p>
                </div>
              </div>
              <div className="flex items-start space-x-3">
                <div className="w-8 h-8 bg-purple-500 text-white rounded-full flex items-center justify-center text-sm font-bold">2</div>
                <div>
                  <p className="font-medium">Neural Network Evaluation</p>
                  <p className="text-gray-600 text-sm">Query the NeRF network at each sampled point</p>
                </div>
              </div>
              <div className="flex items-start space-x-3">
                <div className="w-8 h-8 bg-purple-500 text-white rounded-full flex items-center justify-center text-sm font-bold">3</div>
                <div>
                  <p className="font-medium">Volume Rendering</p>
                  <p className="text-gray-600 text-sm">Integrate densities and colors along rays</p>
                </div>
              </div>
              <div className="flex items-start space-x-3">
                <div className="w-8 h-8 bg-purple-500 text-white rounded-full flex items-center justify-center text-sm font-bold">4</div>
                <div>
                  <p className="font-medium">Loss Computation</p>
                  <p className="text-gray-600 text-sm">Compare rendered images with ground truth</p>
                </div>
              </div>
            </div>
          </div>
        </div>
      )
    },
    {
      id: 'image-guidelines',
      title: 'Image Guidelines',
      icon: Upload,
      content: (
        <div className="space-y-6">
          <div>
            <h3 className="text-xl font-semibold mb-3">Best Practices for Image Capture</h3>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <div className="space-y-4">
                <div className="bg-green-50 border border-green-200 rounded-lg p-4">
                  <h4 className="font-semibold text-green-800 mb-2 flex items-center">
                    <CheckCircle className="w-5 h-5 mr-2" />
                    Do's
                  </h4>
                  <ul className="text-green-700 space-y-2 text-sm">
                    <li>• Capture 20-100 images from different angles</li>
                    <li>• Maintain consistent lighting throughout</li>
                    <li>• Ensure good overlap between consecutive images</li>
                    <li>• Use high-resolution images (1920x1080 or higher)</li>
                    <li>• Keep the subject in focus</li>
                    <li>• Capture images in a 360° circle around the subject</li>
                  </ul>
                </div>
              </div>
              
              <div className="space-y-4">
                <div className="bg-red-50 border border-red-200 rounded-lg p-4">
                  <h4 className="font-semibold text-red-800 mb-2 flex items-center">
                    <AlertTriangle className="w-5 h-5 mr-2" />
                    Don'ts
                  </h4>
                  <ul className="text-red-700 space-y-2 text-sm">
                    <li>• Avoid motion blur or camera shake</li>
                    <li>• Don't use images with moving subjects</li>
                    <li>• Avoid reflective or transparent surfaces</li>
                    <li>• Don't capture in low-light conditions</li>
                    <li>• Avoid images with extreme perspective changes</li>
                    <li>• Don't use heavily compressed images</li>
                  </ul>
                </div>
              </div>
            </div>
          </div>

          <div>
            <h3 className="text-xl font-semibold mb-3">Camera Setup Recommendations</h3>
            <div className="bg-blue-50 border border-blue-200 rounded-lg p-4">
              <h4 className="font-semibold text-blue-800 mb-3">Optimal Capture Pattern</h4>
              <div className="grid grid-cols-1 md:grid-cols-3 gap-4 text-sm">
                <div>
                  <p className="font-medium text-blue-800">Horizontal Circle</p>
                  <p className="text-blue-700">Capture images every 15-20 degrees around the subject</p>
                </div>
                <div>
                  <p className="font-medium text-blue-800">Vertical Angles</p>
                  <p className="text-blue-700">Include some images from above and below eye level</p>
                </div>
                <div>
                  <p className="font-medium text-blue-800">Distance</p>
                  <p className="text-blue-700">Maintain consistent distance from the subject</p>
                </div>
              </div>
            </div>
          </div>

          <div>
            <h3 className="text-xl font-semibold mb-3">Supported Formats</h3>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div className="bg-gray-50 rounded-lg p-4">
                <h4 className="font-semibold mb-2">Image Formats</h4>
                <ul className="space-y-1 text-sm">
                  <li>• JPEG (.jpg, .jpeg)</li>
                  <li>• PNG (.png)</li>
                  <li>• TIFF (.tiff, .tif)</li>
                  <li>• WebP (.webp)</li>
                </ul>
              </div>
              <div className="bg-gray-50 rounded-lg p-4">
                <h4 className="font-semibold mb-2">Recommended Settings</h4>
                <ul className="space-y-1 text-sm">
                  <li>• Resolution: 1920x1080 or higher</li>
                  <li>• Quality: High (low compression)</li>
                  <li>• Color space: sRGB</li>
                  <li>• File size: 1-10MB per image</li>
                </ul>
              </div>
            </div>
          </div>
        </div>
      )
    },
    {
      id: 'training-guide',
      title: 'Training Guide',
      icon: Settings,
      content: (
        <div className="space-y-6">
          <div>
            <h3 className="text-xl font-semibold mb-3">Training Process Overview</h3>
            <div className="space-y-4">
              <div className="bg-gradient-to-r from-blue-50 to-purple-50 border border-blue-200 rounded-lg p-6">
                <h4 className="font-semibold text-blue-800 mb-3">Training Stages</h4>
                <div className="space-y-3">
                  <div className="flex items-center space-x-3">
                    <div className="w-6 h-6 bg-blue-500 text-white rounded-full flex items-center justify-center text-sm">1</div>
                    <div>
                      <p className="font-medium">Initialization</p>
                      <p className="text-sm text-gray-600">Setting up the neural network and loading data</p>
                    </div>
                  </div>
                  <div className="flex items-center space-x-3">
                    <div className="w-6 h-6 bg-purple-500 text-white rounded-full flex items-center justify-center text-sm">2</div>
                    <div>
                      <p className="font-medium">Coarse Sampling</p>
                      <p className="text-sm text-gray-600">Initial ray sampling and density estimation</p>
                    </div>
                  </div>
                  <div className="flex items-center space-x-3">
                    <div className="w-6 h-6 bg-green-500 text-white rounded-full flex items-center justify-center text-sm">3</div>
                    <div>
                      <p className="font-medium">Fine Sampling</p>
                      <p className="text-sm text-gray-600">Detailed sampling based on coarse results</p>
                    </div>
                  </div>
                  <div className="flex items-center space-x-3">
                    <div className="w-6 h-6 bg-yellow-500 text-white rounded-full flex items-center justify-center text-sm">4</div>
                    <div>
                      <p className="font-medium">Optimization</p>
                      <p className="text-sm text-gray-600">Gradient descent to minimize rendering loss</p>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </div>

          <div>
            <h3 className="text-xl font-semibold mb-3">Training Metrics Explained</h3>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <div className="space-y-4">
                <div className="bg-white border border-gray-200 rounded-lg p-4">
                  <h4 className="font-semibold text-gray-800 mb-2">Loss Function</h4>
                  <p className="text-sm text-gray-600 mb-2">Measures the difference between rendered and ground truth images</p>
                  <div className="text-xs text-gray-500">
                    <p>• Lower values indicate better training</p>
                    <p>• Typically decreases over time</p>
                    <p>• Should stabilize near the end</p>
                  </div>
                </div>
                
                <div className="bg-white border border-gray-200 rounded-lg p-4">
                  <h4 className="font-semibold text-gray-800 mb-2">PSNR (Peak Signal-to-Noise Ratio)</h4>
                  <p className="text-sm text-gray-600 mb-2">Measures image quality and reconstruction accuracy</p>
                  <div className="text-xs text-gray-500">
                    <p>• Higher values indicate better quality</p>
                    <p>• Good: 20-30 dB</p>
                    <p>• Excellent: 30+ dB</p>
                  </div>
                </div>
              </div>
              
              <div className="space-y-4">
                <div className="bg-white border border-gray-200 rounded-lg p-4">
                  <h4 className="font-semibold text-gray-800 mb-2">Learning Rate</h4>
                  <p className="text-sm text-gray-600 mb-2">Controls how much the model updates in each step</p>
                  <div className="text-xs text-gray-500">
                    <p>• Starts high, decreases over time</p>
                    <p>• Affects training stability</p>
                    <p>• Too high: unstable training</p>
                  </div>
                </div>
                
                <div className="bg-white border border-gray-200 rounded-lg p-4">
                  <h4 className="font-semibold text-gray-800 mb-2">Training Steps</h4>
                  <p className="text-sm text-gray-600 mb-2">Number of optimization iterations completed</p>
                  <div className="text-xs text-gray-500">
                    <p>• More steps = more training time</p>
                    <p>• Typical: 1000-5000 steps</p>
                    <p>• Depends on scene complexity</p>
                  </div>
                </div>
              </div>
            </div>
          </div>

          <div>
            <h3 className="text-xl font-semibold mb-3">Training Tips</h3>
            <div className="bg-yellow-50 border border-yellow-200 rounded-lg p-4">
              <h4 className="font-semibold text-yellow-800 mb-3 flex items-center">
                <Lightbulb className="w-5 h-5 mr-2" />
                Optimization Tips
              </h4>
              <ul className="text-yellow-800 space-y-2 text-sm">
                <li>• <strong>Patience:</strong> Training can take 10-30 minutes depending on scene complexity</li>
                <li>• <strong>Monitor Progress:</strong> Watch the loss and PSNR metrics for convergence</li>
                <li>• <strong>Quality vs Speed:</strong> More training steps generally produce better results</li>
                <li>• <strong>Hardware:</strong> GPU acceleration significantly speeds up training</li>
                <li>• <strong>Interruption:</strong> You can stop and resume training at any time</li>
              </ul>
            </div>
          </div>
        </div>
      )
    },
    {
      id: 'export-formats',
      title: 'Export Formats',
      icon: Download,
      content: (
        <div className="space-y-6">
          <div>
            <h3 className="text-xl font-semibold mb-3">Supported Export Formats</h3>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <div className="space-y-4">
                <div className="bg-white border border-gray-200 rounded-lg p-4">
                  <h4 className="font-semibold text-gray-800 mb-2">3D Mesh Formats</h4>
                  <ul className="space-y-2 text-sm">
                    <li className="flex items-center justify-between">
                      <span>GLTF/GLB</span>
                      <span className="text-green-600 font-medium">Recommended</span>
                    </li>
                    <li className="flex items-center justify-between">
                      <span>OBJ</span>
                      <span className="text-blue-600 font-medium">Universal</span>
                    </li>
                    <li className="flex items-center justify-between">
                      <span>PLY</span>
                      <span className="text-blue-600 font-medium">Point Cloud</span>
                    </li>
                    <li className="flex items-center justify-between">
                      <span>USD</span>
                      <span className="text-purple-600 font-medium">Professional</span>
                    </li>
                    <li className="flex items-center justify-between">
                      <span>FBX</span>
                      <span className="text-blue-600 font-medium">Animation</span>
                    </li>
                    <li className="flex items-center justify-between">
                      <span>STL</span>
                      <span className="text-blue-600 font-medium">3D Printing</span>
                    </li>
                  </ul>
                </div>
              </div>
              
              <div className="space-y-4">
                <div className="bg-white border border-gray-200 rounded-lg p-4">
                  <h4 className="font-semibold text-gray-800 mb-2">Quality Settings</h4>
                  <div className="space-y-3">
                    <div>
                      <p className="font-medium text-sm">Low Quality</p>
                      <p className="text-xs text-gray-600">Fast export, smaller file size</p>
                    </div>
                    <div>
                      <p className="font-medium text-sm">Medium Quality</p>
                      <p className="text-xs text-gray-600">Balanced performance and quality</p>
                    </div>
                    <div>
                      <p className="font-medium text-sm">High Quality</p>
                      <p className="text-xs text-gray-600">Best visual quality, larger files</p>
                    </div>
                  </div>
                </div>
                
                <div className="bg-white border border-gray-200 rounded-lg p-4">
                  <h4 className="font-semibold text-gray-800 mb-2">Export Features</h4>
                  <ul className="space-y-2 text-sm">
                    <li>• Automatic mesh optimization</li>
                    <li>• Texture baking from NeRF</li>
                    <li>• Compression support</li>
                    <li>• Batch export capabilities</li>
                    <li>• Progress tracking</li>
                  </ul>
                </div>
              </div>
            </div>
          </div>

          <div>
            <h3 className="text-xl font-semibold mb-3">Format Comparison</h3>
            <div className="overflow-x-auto">
              <table className="w-full border-collapse border border-gray-200">
                <thead>
                  <tr className="bg-gray-50">
                    <th className="border border-gray-200 px-4 py-2 text-left">Format</th>
                    <th className="border border-gray-200 px-4 py-2 text-left">Use Case</th>
                    <th className="border border-gray-200 px-4 py-2 text-left">Pros</th>
                    <th className="border border-gray-200 px-4 py-2 text-left">Cons</th>
                  </tr>
                </thead>
                <tbody>
                  <tr>
                    <td className="border border-gray-200 px-4 py-2 font-medium">GLTF/GLB</td>
                    <td className="border border-gray-200 px-4 py-2">Web, Mobile, AR/VR</td>
                    <td className="border border-gray-200 px-4 py-2">Compact, efficient, widely supported</td>
                    <td className="border border-gray-200 px-4 py-2">Limited to modern platforms</td>
                  </tr>
                  <tr className="bg-gray-50">
                    <td className="border border-gray-200 px-4 py-2 font-medium">OBJ</td>
                    <td className="border border-gray-200 px-4 py-2">3D Software, Universal</td>
                    <td className="border border-gray-200 px-4 py-2">Universal compatibility, simple</td>
                    <td className="border border-gray-200 px-4 py-2">Large file sizes, no animation</td>
                  </tr>
                  <tr>
                    <td className="border border-gray-200 px-4 py-2 font-medium">USD</td>
                    <td className="border border-gray-200 px-4 py-2">Professional VFX</td>
                    <td className="border border-gray-200 px-4 py-2">Advanced features, industry standard</td>
                    <td className="border border-gray-200 px-4 py-2">Complex, limited software support</td>
                  </tr>
                  <tr className="bg-gray-50">
                    <td className="border border-gray-200 px-4 py-2 font-medium">STL</td>
                    <td className="border border-gray-200 px-4 py-2">3D Printing</td>
                    <td className="border border-gray-200 px-4 py-2">Perfect for 3D printing</td>
                    <td className="border border-gray-200 px-4 py-2">No color/texture, large files</td>
                  </tr>
                </tbody>
              </table>
            </div>
          </div>
        </div>
      )
    },
    {
      id: 'api-reference',
      title: 'API Reference',
      icon: Code,
      content: (
        <div className="space-y-6">
          <div>
            <h3 className="text-xl font-semibold mb-3">REST API Endpoints</h3>
            <div className="space-y-4">
              <div className="bg-gray-50 rounded-lg p-4">
                <h4 className="font-semibold mb-2">Base URL</h4>
                <code className="bg-gray-800 text-green-400 px-2 py-1 rounded text-sm">http://localhost:8000/api/v1</code>
              </div>
              
              <div className="space-y-3">
                <div className="border border-gray-200 rounded-lg p-4">
                  <div className="flex items-center space-x-2 mb-2">
                    <span className="bg-green-100 text-green-800 px-2 py-1 rounded text-xs font-medium">GET</span>
                    <code className="text-sm">/projects</code>
                  </div>
                  <p className="text-sm text-gray-600">Get all projects</p>
                </div>
                
                <div className="border border-gray-200 rounded-lg p-4">
                  <div className="flex items-center space-x-2 mb-2">
                    <span className="bg-blue-100 text-blue-800 px-2 py-1 rounded text-xs font-medium">POST</span>
                    <code className="text-sm">/projects</code>
                  </div>
                  <p className="text-sm text-gray-600">Create a new project</p>
                  <div className="mt-2 text-xs text-gray-500">
                    Body: {"{"}"name": "string", "description": "string"{"}"}
                  </div>
                </div>
                
                <div className="border border-gray-200 rounded-lg p-4">
                  <div className="flex items-center space-x-2 mb-2">
                    <span className="bg-blue-100 text-blue-800 px-2 py-1 rounded text-xs font-medium">POST</span>
                    <code className="text-sm">/projects/{'{id}'}/upload_images</code>
                  </div>
                  <p className="text-sm text-gray-600">Upload images to a project</p>
                  <div className="mt-2 text-xs text-gray-500">
                    Body: FormData with image files
                  </div>
                </div>
                
                <div className="border border-gray-200 rounded-lg p-4">
                  <div className="flex items-center space-x-2 mb-2">
                    <span className="bg-blue-100 text-blue-800 px-2 py-1 rounded text-xs font-medium">POST</span>
                    <code className="text-sm">/projects/{'{id}'}/start_training</code>
                  </div>
                  <p className="text-sm text-gray-600">Start NeRF training</p>
                </div>
                
                <div className="border border-gray-200 rounded-lg p-4">
                  <div className="flex items-center space-x-2 mb-2">
                    <span className="bg-blue-100 text-blue-800 px-2 py-1 rounded text-xs font-medium">POST</span>
                    <code className="text-sm">/projects/{'{id}'}/render</code>
                  </div>
                  <p className="text-sm text-gray-600">Render a novel view</p>
                  <div className="mt-2 text-xs text-gray-500">
                    Body: {"{"}"pose": [16], "resolution": number{"}"}
                  </div>
                </div>
              </div>
            </div>
          </div>

          <div>
            <h3 className="text-xl font-semibold mb-3">WebSocket API</h3>
            <div className="bg-gray-50 rounded-lg p-4">
              <h4 className="font-semibold mb-2">Training Progress</h4>
              <code className="bg-gray-800 text-green-400 px-2 py-1 rounded text-sm">ws://localhost:8000/api/v1/ws/jobs/{'{job_id}'}</code>
              <p className="text-sm text-gray-600 mt-2">Real-time training progress updates</p>
            </div>
          </div>

          <div>
            <h3 className="text-xl font-semibold mb-3">Response Formats</h3>
            <div className="space-y-4">
              <div className="bg-white border border-gray-200 rounded-lg p-4">
                <h4 className="font-semibold mb-2">Project Object</h4>
                <pre className="bg-gray-800 text-green-400 p-3 rounded text-xs overflow-x-auto">
{`{
  "id": "uuid",
  "name": "string",
  "status": "string",
  "created_at": "datetime",
  "updated_at": "datetime"
}`}
                </pre>
              </div>
              
              <div className="bg-white border border-gray-200 rounded-lg p-4">
                <h4 className="font-semibold mb-2">Training Metrics</h4>
                <pre className="bg-gray-800 text-green-400 p-3 rounded text-xs overflow-x-auto">
{`{
  "step": number,
  "loss": number,
  "psnr": number,
  "lr": number
}`}
                </pre>
              </div>
            </div>
          </div>
        </div>
      )
    }
  ];

  const currentSection = section || 'getting-started';
  const activeSection = sections.find(s => s.id === currentSection) || sections[0];

  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
      {/* Header */}
      <motion.div 
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        className="text-center mb-12"
      >
        <h1 className="text-4xl font-bold text-gray-900 mb-4">Documentation</h1>
        <p className="text-xl text-gray-600 max-w-3xl mx-auto">
          Complete guide to using NeRF Studio. Learn about Neural Radiance Fields, 
          best practices for image capture, training optimization, and more.
        </p>
      </motion.div>

      <div className="grid grid-cols-1 lg:grid-cols-4 gap-8">
        {/* Sidebar Navigation */}
        <div className="lg:col-span-1">
          <div className="bg-white rounded-2xl shadow-lg p-6 sticky top-24">
            <h2 className="text-lg font-semibold text-gray-900 mb-4">Contents</h2>
            <nav className="space-y-2">
              {sections.map((section) => {
                const Icon = section.icon;
                const isActive = section.id === currentSection;
                const isExpanded = expandedSections.includes(section.id);
                
                return (
                  <div key={section.id}>
                    <button
                      onClick={() => toggleSection(section.id)}
                      className={`w-full flex items-center justify-between p-3 rounded-lg transition-colors ${
                        isActive 
                          ? 'bg-blue-50 text-blue-600 border border-blue-200' 
                          : 'text-gray-600 hover:text-blue-600 hover:bg-gray-50'
                      }`}
                    >
                      <div className="flex items-center space-x-2">
                        <Icon size={16} />
                        <span className="font-medium">{section.title}</span>
                      </div>
                      {isExpanded ? <ChevronDown size={16} /> : <ChevronRight size={16} />}
                    </button>
                    
                    {isExpanded && (
                      <motion.div
                        initial={{ opacity: 0, height: 0 }}
                        animate={{ opacity: 1, height: 'auto' }}
                        exit={{ opacity: 0, height: 0 }}
                        className="ml-6 mt-2 space-y-1"
                      >
                        <Link
                          to={`/docs/${section.id}`}
                          className={`block p-2 rounded text-sm transition-colors ${
                            isActive 
                              ? 'text-blue-600 bg-blue-50' 
                              : 'text-gray-500 hover:text-blue-600 hover:bg-gray-50'
                          }`}
                        >
                          Overview
                        </Link>
                      </motion.div>
                    )}
                  </div>
                );
              })}
            </nav>
          </div>
        </div>

        {/* Main Content */}
        <div className="lg:col-span-3">
          <motion.div 
            key={activeSection.id}
            initial={{ opacity: 0, x: 20 }}
            animate={{ opacity: 1, x: 0 }}
            className="bg-white rounded-2xl shadow-lg p-8"
          >
            <div className="flex items-center space-x-3 mb-6">
              <div className="w-10 h-10 bg-gradient-to-r from-blue-500 to-purple-600 rounded-lg flex items-center justify-center">
                <activeSection.icon className="w-6 h-6 text-white" />
              </div>
              <div>
                <h1 className="text-3xl font-bold text-gray-900">{activeSection.title}</h1>
                <p className="text-gray-600">Complete guide and reference</p>
              </div>
            </div>

            {activeSection.content}
          </motion.div>
        </div>
      </div>
    </div>
  );
};

export default Documentation; 