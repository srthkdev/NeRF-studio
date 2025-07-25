import { useState } from 'react';
import { useParams, Link } from 'react-router-dom';
import { motion } from 'framer-motion';
import { 
  BookOpen, 
  Code, 
  Play, 
  Settings, 
  Download, 
  Upload,
  Lightbulb,
  AlertTriangle,
  CheckCircle,
  ChevronRight,
  ChevronDown,
  Activity
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
      title: 'NeRF Theory & Architecture',
      icon: BookOpen,
      content: (
        <div className="space-y-8">
          {/* Introduction */}
          <div>
            <h3 className="text-2xl font-bold mb-4 text-gray-900">🎯 What are Neural Radiance Fields (NeRF)?</h3>
            <p className="text-gray-700 mb-6 text-lg leading-relaxed">
              Neural Radiance Fields (NeRF) represent a revolutionary approach to 3D scene representation and novel view synthesis. 
              Instead of traditional explicit 3D representations like meshes or point clouds, NeRF models a scene as a continuous 
              5D function that outputs the volume density and view-dependent emitted radiance at any point in 3D space.
            </p>
            
            <div className="bg-gradient-to-r from-blue-50 to-purple-50 border border-blue-200 rounded-xl p-6">
              <h4 className="font-bold text-blue-900 mb-4 text-lg">🌟 Revolutionary Impact</h4>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div className="bg-white rounded-lg p-4 shadow-sm">
                  <h5 className="font-semibold text-blue-800 mb-2">Before NeRF</h5>
                  <ul className="text-blue-700 text-sm space-y-1">
                    <li>• Explicit 3D geometry required</li>
                    <li>• Complex mesh reconstruction</li>
                    <li>• Limited view synthesis quality</li>
                    <li>• Manual texture mapping</li>
                  </ul>
                </div>
                <div className="bg-white rounded-lg p-4 shadow-sm">
                  <h5 className="font-semibold text-purple-800 mb-2">With NeRF</h5>
                  <ul className="text-purple-700 text-sm space-y-1">
                    <li>• Implicit 3D representation</li>
                    <li>• Photorealistic novel views</li>
                    <li>• View-dependent effects</li>
                    <li>• Automatic texture learning</li>
                  </ul>
                </div>
              </div>
            </div>
          </div>

          {/* Mathematical Foundation */}
          <div>
            <h3 className="text-2xl font-bold mb-4 text-gray-900">🧮 Mathematical Foundation</h3>
            
            <div className="bg-gray-50 rounded-xl p-6 mb-6">
              <h4 className="font-bold text-gray-900 mb-4">Core NeRF Function</h4>
              <div className="bg-white rounded-lg p-4 font-mono text-sm border">
                <p className="mb-3 text-blue-600 font-semibold">F: (x, y, z, θ, φ) → (σ, c)</p>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  <div>
                    <p className="font-semibold text-gray-800 mb-2">Input Parameters:</p>
                    <ul className="text-gray-700 space-y-1">
                      <li>• <strong>(x, y, z):</strong> 3D spatial coordinates</li>
                      <li>• <strong>(θ, φ):</strong> 2D viewing direction (spherical)</li>
                    </ul>
                  </div>
                  <div>
                    <p className="font-semibold text-gray-800 mb-2">Output Values:</p>
                    <ul className="text-gray-700 space-y-1">
                      <li>• <strong>σ:</strong> Volume density (opacity)</li>
                      <li>• <strong>c:</strong> View-dependent radiance (RGB color)</li>
                    </ul>
                  </div>
                </div>
              </div>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <div className="bg-blue-50 border border-blue-200 rounded-lg p-4">
                <h4 className="font-semibold text-blue-800 mb-3">Volume Rendering Equation</h4>
                <div className="bg-white rounded p-3 font-mono text-xs">
                  <p className="text-blue-600 mb-2">C(r) = ∫ T(t) σ(r(t)) c(r(t), d) dt</p>
                  <p className="text-gray-600">Where T(t) = exp(-∫ σ(r(s)) ds)</p>
                </div>
                <p className="text-blue-700 text-sm mt-3">
                  This equation integrates the radiance along a ray, accounting for volume density and transparency.
                </p>
              </div>
              
              <div className="bg-purple-50 border border-purple-200 rounded-lg p-4">
                <h4 className="font-semibold text-purple-800 mb-3">Positional Encoding</h4>
                <div className="bg-white rounded p-3 font-mono text-xs">
                  <p className="text-purple-600 mb-2">γ(p) = [sin(2⁰πp), cos(2⁰πp), ..., sin(2^(L-1)πp), cos(2^(L-1)πp)]</p>
                </div>
                <p className="text-purple-700 text-sm mt-3">
                  Fourier feature encoding enables the network to represent high-frequency functions and fine details.
                </p>
              </div>
            </div>
          </div>

          {/* Architecture Details */}
          <div>
            <h3 className="text-2xl font-bold mb-4 text-gray-900">🏗️ NeRF Architecture Deep Dive</h3>
            
            <div className="space-y-6">
              <div className="bg-gradient-to-r from-green-50 to-blue-50 border border-green-200 rounded-xl p-6">
                <h4 className="font-bold text-green-900 mb-4">Neural Network Architecture</h4>
                <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
                  <div className="bg-white rounded-lg p-4 shadow-sm">
                    <h5 className="font-semibold text-green-800 mb-2">📊 Density Network (σ)</h5>
                    <ul className="text-green-700 text-sm space-y-1">
                      <li>• 8 fully connected layers</li>
                      <li>• 256 hidden units per layer</li>
                      <li>• ReLU activation functions</li>
                      <li>• Outputs volume density</li>
                    </ul>
                  </div>
                  <div className="bg-white rounded-lg p-4 shadow-sm">
                    <h5 className="font-semibold text-blue-800 mb-2">🎨 Color Network (c)</h5>
                    <ul className="text-blue-700 text-sm space-y-1">
                      <li>• 1 additional layer</li>
                      <li>• 128 hidden units</li>
                      <li>• Sigmoid activation</li>
                      <li>• Outputs RGB color</li>
                    </ul>
                  </div>
                  <div className="bg-white rounded-lg p-4 shadow-sm">
                    <h5 className="font-semibold text-purple-800 mb-2">🔧 Skip Connections</h5>
                    <ul className="text-purple-700 text-sm space-y-1">
                      <li>• 4th layer skip connection</li>
                      <li>• Preserves fine details</li>
                      <li>• Improves gradient flow</li>
                      <li>• Better convergence</li>
                    </ul>
                  </div>
                </div>
              </div>

              <div className="bg-gradient-to-r from-orange-50 to-red-50 border border-orange-200 rounded-xl p-6">
                <h4 className="font-bold text-orange-900 mb-4">Hierarchical Sampling Strategy</h4>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                  <div>
                    <h5 className="font-semibold text-orange-800 mb-3">🎯 Coarse Network</h5>
                    <ul className="text-orange-700 space-y-2">
                      <li>• 64 samples per ray</li>
                      <li>• Uniform sampling</li>
                      <li>• Rough density estimation</li>
                      <li>• Fast initial pass</li>
                    </ul>
                  </div>
                  <div>
                    <h5 className="font-semibold text-red-800 mb-3">🎯 Fine Network</h5>
                    <ul className="text-red-700 space-y-2">
                      <li>• 128 samples per ray</li>
                      <li>• Importance sampling</li>
                      <li>• Refined density estimation</li>
                      <li>• High-quality rendering</li>
                    </ul>
                  </div>
                </div>
              </div>
            </div>
          </div>

          {/* Training Process */}
          <div>
            <h3 className="text-2xl font-bold mb-4 text-gray-900">🚀 Training Process & Optimization</h3>
            
            <div className="space-y-6">
              <div className="bg-gradient-to-r from-indigo-50 to-purple-50 border border-indigo-200 rounded-xl p-6">
                <h4 className="font-bold text-indigo-900 mb-4">Training Pipeline</h4>
                <div className="space-y-4">
                  <div className="flex items-start space-x-4">
                    <div className="w-10 h-10 bg-indigo-500 text-white rounded-full flex items-center justify-center text-sm font-bold">1</div>
                    <div className="flex-1">
                      <h5 className="font-semibold text-indigo-800 mb-2">Ray Generation & Sampling</h5>
                      <p className="text-indigo-700 text-sm">
                        Generate camera rays from input viewpoints and sample points along each ray using stratified sampling.
                        This creates a set of 3D points to query the neural network.
                      </p>
                    </div>
                  </div>
                  
                  <div className="flex items-start space-x-4">
                    <div className="w-10 h-10 bg-indigo-500 text-white rounded-full flex items-center justify-center text-sm font-bold">2</div>
                    <div className="flex-1">
                      <h5 className="font-semibold text-indigo-800 mb-2">Neural Network Forward Pass</h5>
                      <p className="text-indigo-700 text-sm">
                        For each sampled point, apply positional encoding and pass through the MLP to predict density and color.
                        The network learns to map 5D coordinates to volume density and view-dependent radiance.
                      </p>
                    </div>
                  </div>
                  
                  <div className="flex items-start space-x-4">
                    <div className="w-10 h-10 bg-indigo-500 text-white rounded-full flex items-center justify-center text-sm font-bold">3</div>
                    <div className="flex-1">
                      <h5 className="font-semibold text-indigo-800 mb-2">Volume Rendering Integration</h5>
                      <p className="text-indigo-700 text-sm">
                        Use the volume rendering equation to integrate densities and colors along each ray, producing 
                        the final pixel color. This step converts the 3D representation to 2D images.
                      </p>
                    </div>
                  </div>
                  
                  <div className="flex items-start space-x-4">
                    <div className="w-10 h-10 bg-indigo-500 text-white rounded-full flex items-center justify-center text-sm font-bold">4</div>
                    <div className="flex-1">
                      <h5 className="font-semibold text-indigo-800 mb-2">Loss Computation & Backpropagation</h5>
                      <p className="text-indigo-700 text-sm">
                        Compute the photometric loss between rendered and ground truth images, then backpropagate 
                        gradients to update network weights. This drives the learning process.
                      </p>
                    </div>
                  </div>
                </div>
              </div>

              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                <div className="bg-yellow-50 border border-yellow-200 rounded-lg p-4">
                  <h4 className="font-semibold text-yellow-800 mb-3">🎯 Loss Functions</h4>
                  <ul className="text-yellow-700 space-y-2">
                    <li><strong>Photometric Loss:</strong> L2 distance between rendered and ground truth pixels</li>
                    <li><strong>Coarse Loss:</strong> Supervises the coarse network output</li>
                    <li><strong>Fine Loss:</strong> Supervises the fine network output</li>
                    <li><strong>Total Loss:</strong> L = L_c + L_f</li>
                  </ul>
                </div>
                
                <div className="bg-green-50 border border-green-200 rounded-lg p-4">
                  <h4 className="font-semibold text-green-800 mb-3">⚡ Optimization Techniques</h4>
                  <ul className="text-green-700 space-y-2">
                    <li><strong>Adam Optimizer:</strong> Adaptive learning rate optimization</li>
                    <li><strong>Learning Rate Scheduling:</strong> Exponential decay from 5e-4 to 5e-5</li>
                    <li><strong>Gradient Clipping:</strong> Prevents gradient explosion</li>
                    <li><strong>Early Stopping:</strong> Prevents overfitting</li>
                  </ul>
                </div>
              </div>
            </div>
          </div>

          {/* Advanced Concepts */}
          <div>
            <h3 className="text-2xl font-bold mb-4 text-gray-900">🔬 Advanced NeRF Concepts</h3>
            
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <div className="bg-blue-50 border border-blue-200 rounded-lg p-4">
                <h4 className="font-semibold text-blue-800 mb-3">🎨 View-Dependent Effects</h4>
                <p className="text-blue-700 text-sm mb-3">
                  NeRF naturally captures view-dependent effects like specular reflections, transparency, and 
                  subsurface scattering by conditioning the color output on viewing direction.
                </p>
                <ul className="text-blue-700 text-sm space-y-1">
                  <li>• Specular highlights</li>
                  <li>• Fresnel effects</li>
                  <li>• Transparency</li>
                  <li>• Subsurface scattering</li>
                </ul>
              </div>
              
              <div className="bg-purple-50 border border-purple-200 rounded-lg p-4">
                <h4 className="font-semibold text-purple-800 mb-3">🌊 Volume Density Interpretation</h4>
                <p className="text-purple-700 text-sm mb-3">
                  Volume density represents the probability of a ray terminating at each point, enabling 
                  realistic modeling of transparent and semi-transparent materials.
                </p>
                <ul className="text-purple-700 text-sm space-y-1">
                  <li>• Opacity control</li>
                  <li>• Transparency modeling</li>
                  <li>• Smoke and fog effects</li>
                  <li>• Soft shadows</li>
                </ul>
              </div>
            </div>

            <div className="bg-gradient-to-r from-pink-50 to-red-50 border border-pink-200 rounded-xl p-6 mt-6">
              <h4 className="font-bold text-pink-900 mb-4">🚀 Performance Optimizations</h4>
              <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                <div className="bg-white rounded-lg p-4 shadow-sm">
                  <h5 className="font-semibold text-pink-800 mb-2">⚡ Speed Optimizations</h5>
                  <ul className="text-pink-700 text-sm space-y-1">
                    <li>• Hierarchical sampling</li>
                    <li>• Early ray termination</li>
                    <li>• GPU acceleration</li>
                    <li>• Batch processing</li>
                  </ul>
                </div>
                <div className="bg-white rounded-lg p-4 shadow-sm">
                  <h5 className="font-semibold text-red-800 mb-2">🎯 Quality Improvements</h5>
                  <ul className="text-red-700 text-sm space-y-1">
                    <li>• Positional encoding</li>
                    <li>• Skip connections</li>
                    <li>• View-dependent modeling</li>
                    <li>• Multi-scale training</li>
                  </ul>
                </div>
                <div className="bg-white rounded-lg p-4 shadow-sm">
                  <h5 className="font-semibold text-purple-800 mb-2">💾 Memory Efficiency</h5>
                  <ul className="text-purple-700 text-sm space-y-1">
                    <li>• Gradient checkpointing</li>
                    <li>• Mixed precision</li>
                    <li>• Dynamic batching</li>
                    <li>• Memory pooling</li>
                  </ul>
                </div>
              </div>
            </div>
          </div>

          {/* Applications */}
          <div>
            <h3 className="text-2xl font-bold mb-4 text-gray-900">🌍 Real-World Applications</h3>
            
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
              <div className="bg-gradient-to-br from-blue-50 to-indigo-50 border border-blue-200 rounded-lg p-4">
                <h4 className="font-semibold text-blue-800 mb-3">🎬 Entertainment & Media</h4>
                <ul className="text-blue-700 text-sm space-y-1">
                  <li>• Virtual production</li>
                  <li>• 3D content creation</li>
                  <li>• Visual effects</li>
                  <li>• Game asset generation</li>
                </ul>
              </div>
              
              <div className="bg-gradient-to-br from-green-50 to-emerald-50 border border-green-200 rounded-lg p-4">
                <h4 className="font-semibold text-green-800 mb-3">🏗️ Architecture & Design</h4>
                <ul className="text-green-700 text-sm space-y-1">
                  <li>• Virtual walkthroughs</li>
                  <li>• Interior visualization</li>
                  <li>• Product design</li>
                  <li>• Heritage preservation</li>
                </ul>
              </div>
              
              <div className="bg-gradient-to-br from-purple-50 to-pink-50 border border-purple-200 rounded-lg p-4">
                <h4 className="font-semibold text-purple-800 mb-3">🔬 Scientific Research</h4>
                <ul className="text-purple-700 text-sm space-y-1">
                  <li>• Medical imaging</li>
                  <li>• Material science</li>
                  <li>• Archaeology</li>
                  <li>• Remote sensing</li>
                </ul>
              </div>
              
              <div className="bg-gradient-to-br from-orange-50 to-red-50 border border-orange-200 rounded-lg p-4">
                <h4 className="font-semibold text-orange-800 mb-3">🤖 Robotics & AI</h4>
                <ul className="text-orange-700 text-sm space-y-1">
                  <li>• Scene understanding</li>
                  <li>• Path planning</li>
                  <li>• Object manipulation</li>
                  <li>• Autonomous navigation</li>
                </ul>
              </div>
              
              <div className="bg-gradient-to-br from-yellow-50 to-amber-50 border border-yellow-200 rounded-lg p-4">
                <h4 className="font-semibold text-yellow-800 mb-3">📱 AR/VR Applications</h4>
                <ul className="text-yellow-700 text-sm space-y-1">
                  <li>• Virtual reality</li>
                  <li>• Augmented reality</li>
                  <li>• Mixed reality</li>
                  <li>• Spatial computing</li>
                </ul>
              </div>
              
              <div className="bg-gradient-to-br from-gray-50 to-slate-50 border border-gray-200 rounded-lg p-4">
                <h4 className="font-semibold text-gray-800 mb-3">📊 Data Visualization</h4>
                <ul className="text-gray-700 text-sm space-y-1">
                  <li>• 3D data exploration</li>
                  <li>• Scientific visualization</li>
                  <li>• Interactive exhibits</li>
                  <li>• Educational content</li>
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
      icon: Activity,
      content: (
        <div className="space-y-8">
          {/* Training Overview */}
          <div>
            <h3 className="text-2xl font-bold mb-4 text-gray-900">🎯 NeRF Training Overview</h3>
            <p className="text-gray-700 mb-6 text-lg leading-relaxed">
              Training a NeRF model is a computationally intensive process that requires careful attention to data quality, 
              hyperparameters, and system resources. This guide covers everything you need to know for successful NeRF training.
            </p>
            
            <div className="bg-gradient-to-r from-green-50 to-blue-50 border border-green-200 rounded-xl p-6">
              <h4 className="font-bold text-green-900 mb-4 text-lg">⚡ Training Performance Expectations</h4>
              <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                <div className="bg-white rounded-lg p-4 shadow-sm">
                  <h5 className="font-semibold text-green-800 mb-2">🕐 Training Time</h5>
                  <ul className="text-green-700 text-sm space-y-1">
                    <li>• Small scenes: 10-20 minutes</li>
                    <li>• Medium scenes: 20-40 minutes</li>
                    <li>• Large scenes: 40-90 minutes</li>
                    <li>• Complex scenes: 1-3 hours</li>
                  </ul>
                </div>
                <div className="bg-white rounded-lg p-4 shadow-sm">
                  <h5 className="font-semibold text-blue-800 mb-2">💾 Memory Usage</h5>
                  <ul className="text-blue-700 text-sm space-y-1">
                    <li>• GPU VRAM: 4-12GB</li>
                    <li>• System RAM: 8-32GB</li>
                    <li>• Storage: 10-50GB</li>
                    <li>• Checkpoints: 2-10GB</li>
                  </ul>
                </div>
                <div className="bg-white rounded-lg p-4 shadow-sm">
                  <h5 className="font-semibold text-purple-800 mb-2">🎯 Quality Metrics</h5>
                  <ul className="text-purple-700 text-sm space-y-1">
                    <li>• PSNR: 20-35 dB</li>
                    <li>• SSIM: 0.7-0.95</li>
                    <li>• LPIPS: 0.05-0.2</li>
                    <li>• Training steps: 50K-200K</li>
                  </ul>
                </div>
              </div>
            </div>
          </div>

          {/* Training Phases */}
          <div>
            <h3 className="text-2xl font-bold mb-4 text-gray-900">📈 Training Phases & Progress</h3>
            
            <div className="space-y-6">
              <div className="bg-gradient-to-r from-yellow-50 to-orange-50 border border-yellow-200 rounded-xl p-6">
                <h4 className="font-bold text-yellow-900 mb-4">Phase 1: Initial Scene Understanding (0-30%)</h4>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                  <div>
                    <h5 className="font-semibold text-yellow-800 mb-3">🎯 What's Happening</h5>
                    <ul className="text-yellow-700 space-y-2">
                      <li>• Camera pose refinement</li>
                      <li>• Basic scene geometry learning</li>
                      <li>• Initial density field estimation</li>
                      <li>• Coarse color approximation</li>
                    </ul>
                  </div>
                  <div>
                    <h5 className="font-semibold text-orange-800 mb-3">📊 Expected Metrics</h5>
                    <ul className="text-orange-700 space-y-2">
                      <li>• PSNR: 15-20 dB</li>
                      <li>• Loss: High, rapidly decreasing</li>
                      <li>• Learning rate: Maximum</li>
                      <li>• Progress: Slow but steady</li>
                    </ul>
                  </div>
                </div>
              </div>

              <div className="bg-gradient-to-r from-blue-50 to-indigo-50 border border-blue-200 rounded-xl p-6">
                <h4 className="font-bold text-blue-900 mb-4">Phase 2: Neural Network Optimization (30-70%)</h4>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                  <div>
                    <h5 className="font-semibold text-blue-800 mb-3">🎯 What's Happening</h5>
                    <ul className="text-blue-700 space-y-2">
                      <li>• Detailed geometry refinement</li>
                      <li>• View-dependent effects learning</li>
                      <li>• Texture and material modeling</li>
                      <li>• Fine detail capture</li>
                    </ul>
                  </div>
                  <div>
                    <h5 className="font-semibold text-indigo-800 mb-3">📊 Expected Metrics</h5>
                    <ul className="text-indigo-700 space-y-2">
                      <li>• PSNR: 20-30 dB</li>
                      <li>• Loss: Moderate, stable decrease</li>
                      <li>• Learning rate: Gradually decreasing</li>
                      <li>• Progress: Steady improvement</li>
                    </ul>
                  </div>
                </div>
              </div>

              <div className="bg-gradient-to-r from-green-50 to-emerald-50 border border-green-200 rounded-xl p-6">
                <h4 className="font-bold text-green-900 mb-4">Phase 3: Fine-tuning & Convergence (70-100%)</h4>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                  <div>
                    <h5 className="font-semibold text-green-800 mb-3">🎯 What's Happening</h5>
                    <ul className="text-green-700 space-y-2">
                      <li>• High-frequency detail refinement</li>
                      <li>• Specular reflection optimization</li>
                      <li>• Final texture polishing</li>
                      <li>• Convergence to optimal solution</li>
                    </ul>
                  </div>
                  <div>
                    <h5 className="font-semibold text-emerald-800 mb-3">📊 Expected Metrics</h5>
                    <ul className="text-emerald-700 space-y-2">
                      <li>• PSNR: 25-35 dB</li>
                      <li>• Loss: Low, minimal decrease</li>
                      <li>• Learning rate: Minimum</li>
                      <li>• Progress: Fine adjustments</li>
                    </ul>
                  </div>
                </div>
              </div>
            </div>
          </div>

          {/* Hyperparameters */}
          <div>
            <h3 className="text-2xl font-bold mb-4 text-gray-900">⚙️ Training Hyperparameters</h3>
            
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <div className="bg-purple-50 border border-purple-200 rounded-lg p-4">
                <h4 className="font-semibold text-purple-800 mb-3">🎯 Learning Rate Schedule</h4>
                <div className="bg-white rounded p-3 font-mono text-xs mb-3">
                  <p className="text-purple-600">Initial LR: 5e-4</p>
                  <p className="text-purple-600">Decay: Exponential</p>
                  <p className="text-purple-600">Final LR: 5e-5</p>
                  <p className="text-purple-600">Warmup: 1000 steps</p>
                </div>
                <p className="text-purple-700 text-sm">
                  Adaptive learning rate prevents overshooting and ensures stable convergence.
                </p>
              </div>
              
              <div className="bg-blue-50 border border-blue-200 rounded-lg p-4">
                <h4 className="font-semibold text-blue-800 mb-3">📊 Sampling Strategy</h4>
                <div className="bg-white rounded p-3 font-mono text-xs mb-3">
                  <p className="text-blue-600">Coarse samples: 64 per ray</p>
                  <p className="text-blue-600">Fine samples: 128 per ray</p>
                  <p className="text-blue-600">Batch size: 4096 rays</p>
                  <p className="text-blue-600">Chunk size: 8192</p>
                </div>
                <p className="text-blue-700 text-sm">
                  Hierarchical sampling balances quality and computational efficiency.
                </p>
              </div>
            </div>

            <div className="bg-gradient-to-r from-gray-50 to-slate-50 border border-gray-200 rounded-xl p-6 mt-6">
              <h4 className="font-bold text-gray-900 mb-4">🔧 Advanced Training Parameters</h4>
              <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                <div className="bg-white rounded-lg p-4 shadow-sm">
                  <h5 className="font-semibold text-gray-800 mb-2">🎨 Rendering Parameters</h5>
                  <ul className="text-gray-700 text-sm space-y-1">
                    <li>• Near plane: 0.1</li>
                    <li>• Far plane: 100.0</li>
                    <li>• White background: True</li>
                    <li>• Random background: False</li>
                  </ul>
                </div>
                <div className="bg-white rounded-lg p-4 shadow-sm">
                  <h5 className="font-semibold text-gray-800 mb-2">🔄 Optimization</h5>
                  <ul className="text-gray-700 text-sm space-y-1">
                    <li>• Optimizer: Adam</li>
                    <li>• Beta1: 0.9</li>
                    <li>• Beta2: 0.999</li>
                    <li>• Weight decay: 0.0</li>
                  </ul>
                </div>
                <div className="bg-white rounded-lg p-4 shadow-sm">
                  <h5 className="font-semibold text-gray-800 mb-2">📈 Monitoring</h5>
                  <ul className="text-gray-700 text-sm space-y-1">
                    <li>• Log interval: 100 steps</li>
                    <li>• Validation interval: 1000 steps</li>
                    <li>• Checkpoint interval: 5000 steps</li>
                    <li>• Early stopping: 50 epochs</li>
                  </ul>
                </div>
              </div>
            </div>
          </div>

          {/* Troubleshooting */}
          <div>
            <h3 className="text-2xl font-bold mb-4 text-gray-900">🔧 Training Troubleshooting</h3>
            
            <div className="space-y-6">
              <div className="bg-red-50 border border-red-200 rounded-xl p-6">
                <h4 className="font-bold text-red-900 mb-4">❌ Common Training Issues</h4>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                  <div>
                    <h5 className="font-semibold text-red-800 mb-3">🚫 Training Won't Start</h5>
                    <ul className="text-red-700 space-y-2">
                      <li>• Check GPU memory availability</li>
                      <li>• Verify image data integrity</li>
                      <li>• Ensure camera poses are estimated</li>
                      <li>• Check system resources</li>
                    </ul>
                  </div>
                  <div>
                    <h5 className="font-semibold text-red-800 mb-3">📉 Poor Quality Results</h5>
                    <ul className="text-red-700 space-y-2">
                      <li>• Insufficient image overlap</li>
                      <li>• Poor lighting conditions</li>
                      <li>• Motion blur in images</li>
                      <li>• Inaccurate camera poses</li>
                    </ul>
                  </div>
                </div>
              </div>

              <div className="bg-yellow-50 border border-yellow-200 rounded-xl p-6">
                <h4 className="font-bold text-yellow-900 mb-4">⚠️ Performance Issues</h4>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                  <div>
                    <h5 className="font-semibold text-yellow-800 mb-3">🐌 Slow Training</h5>
                    <ul className="text-yellow-700 space-y-2">
                      <li>• Reduce batch size</li>
                      <li>• Use fewer samples per ray</li>
                      <li>• Enable mixed precision</li>
                      <li>• Optimize GPU settings</li>
                    </ul>
                  </div>
                  <div>
                    <h5 className="font-semibold text-yellow-800 mb-3">💾 Memory Issues</h5>
                    <ul className="text-yellow-700 space-y-2">
                      <li>• Reduce chunk size</li>
                      <li>• Use gradient checkpointing</li>
                      <li>• Clear GPU cache</li>
                      <li>• Close other applications</li>
                    </ul>
                  </div>
                </div>
              </div>

              <div className="bg-green-50 border border-green-200 rounded-xl p-6">
                <h4 className="font-bold text-green-900 mb-4">✅ Best Practices</h4>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                  <div>
                    <h5 className="font-semibold text-green-800 mb-3">🎯 For Best Results</h5>
                    <ul className="text-green-700 space-y-2">
                      <li>• Use high-quality images</li>
                      <li>• Ensure good camera coverage</li>
                      <li>• Maintain consistent lighting</li>
                      <li>• Monitor training progress</li>
                    </ul>
                  </div>
                  <div>
                    <h5 className="font-semibold text-green-800 mb-3">⚡ For Fast Training</h5>
                    <ul className="text-green-700 space-y-2">
                      <li>• Use GPU acceleration</li>
                      <li>• Optimize batch sizes</li>
                      <li>• Enable mixed precision</li>
                      <li>• Use efficient sampling</li>
                    </ul>
                  </div>
                </div>
              </div>
            </div>
          </div>

          {/* Monitoring & Metrics */}
          <div>
            <h3 className="text-2xl font-bold mb-4 text-gray-900">📊 Training Monitoring & Metrics</h3>
            
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <div className="bg-blue-50 border border-blue-200 rounded-lg p-4">
                <h4 className="font-semibold text-blue-800 mb-3">📈 Key Metrics to Watch</h4>
                <ul className="text-blue-700 space-y-2">
                  <li><strong>PSNR (Peak Signal-to-Noise Ratio):</strong> Measures image quality (higher is better)</li>
                  <li><strong>Loss:</strong> Training loss should decrease steadily</li>
                  <li><strong>Learning Rate:</strong> Should decay according to schedule</li>
                  <li><strong>Steps:</strong> Training progress indicator</li>
                </ul>
              </div>
              
              <div className="bg-purple-50 border border-purple-200 rounded-lg p-4">
                <h4 className="font-semibold text-purple-800 mb-3">🎯 Quality Thresholds</h4>
                <ul className="text-purple-700 space-y-2">
                  <li><strong>Excellent:</strong> PSNR &gt; 30 dB</li>
                  <li><strong>Good:</strong> PSNR 25-30 dB</li>
                  <li><strong>Acceptable:</strong> PSNR 20-25 dB</li>
                  <li><strong>Poor:</strong> PSNR &lt; 20 dB</li>
                </ul>
              </div>
            </div>

            <div className="bg-gradient-to-r from-indigo-50 to-purple-50 border border-indigo-200 rounded-xl p-6 mt-6">
              <h4 className="font-bold text-indigo-900 mb-4">🔄 Training Checkpoints</h4>
              <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                <div className="bg-white rounded-lg p-4 shadow-sm">
                  <h5 className="font-semibold text-indigo-800 mb-2">💾 Automatic Saving</h5>
                  <ul className="text-indigo-700 text-sm space-y-1">
                    <li>• Every 5000 steps</li>
                    <li>• Best model preservation</li>
                    <li>• Training state backup</li>
                    <li>• Resume capability</li>
                  </ul>
                </div>
                <div className="bg-white rounded-lg p-4 shadow-sm">
                  <h5 className="font-semibold text-purple-800 mb-2">📁 Checkpoint Contents</h5>
                  <ul className="text-purple-700 text-sm space-y-1">
                    <li>• Model weights</li>
                    <li>• Optimizer state</li>
                    <li>• Training metrics</li>
                    <li>• Configuration</li>
                  </ul>
                </div>
                <div className="bg-white rounded-lg p-4 shadow-sm">
                  <h5 className="font-semibold text-blue-800 mb-2">🔄 Recovery Options</h5>
                  <ul className="text-blue-700 text-sm space-y-1">
                    <li>• Resume training</li>
                    <li>• Export model</li>
                    <li>• Continue optimization</li>
                    <li>• Model evaluation</li>
                  </ul>
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