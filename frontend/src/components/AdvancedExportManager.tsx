import React, { useState, useEffect } from 'react';

interface ExportConfig {
  format: string;
  resolution: number;
  texture_resolution: number;
  include_textures: boolean;
  bake_textures: boolean;
  optimize_mesh: boolean;
  compression: boolean;
  quality: string;
  bounds: number[];
}

interface ExportProgress {
  stage: string;
  progress: number;
  message: string;
  messages: string[];
}

interface AdvancedExportManagerProps {
  projectId: string;
  onExportComplete?: (files: Record<string, string>) => void;
}

export function AdvancedExportManager({ projectId, onExportComplete }: AdvancedExportManagerProps) {
  const [exportConfig, setExportConfig] = useState<ExportConfig>({
    format: 'gltf',
    resolution: 128,
    texture_resolution: 1024,
    include_textures: true,
    bake_textures: true,
    optimize_mesh: true,
    compression: true,
    quality: 'high',
    bounds: [-2, 2, -2, 2, -2, 2]
  });

  const [exportProgress, setExportProgress] = useState<ExportProgress | null>(null);
  const [isExporting, setIsExporting] = useState(false);
  const [exportedFiles, setExportedFiles] = useState<Record<string, string>>({});

  const exportFormats = [
    { id: 'gltf', name: 'GLTF 2.0', description: 'Standard 3D format for web and mobile' },
    { id: 'obj', name: 'Wavefront OBJ', description: 'Legacy 3D format with wide support' },
    { id: 'ply', name: 'PLY Point Cloud', description: 'Research-friendly format' },
    { id: 'usd', name: 'Universal Scene Description', description: 'Pixar USD format for professional workflows' },
    { id: 'fbx', name: 'Autodesk FBX', description: 'Animation and rigging support' },
    { id: 'stl', name: 'STL', description: '3D printing format' }
  ];

  const qualityOptions = [
    { id: 'low', name: 'Low', description: 'Fast export, lower quality' },
    { id: 'medium', name: 'Medium', description: 'Balanced quality and speed' },
    { id: 'high', name: 'High', description: 'Best quality, slower export' }
  ];

  const resolutionOptions = [
    { value: 64, label: '64x64x64 (Low)' },
    { value: 128, label: '128x128x128 (Medium)' },
    { value: 256, label: '256x256x256 (High)' },
    { value: 512, label: '512x512x512 (Ultra)' }
  ];

  const textureResolutionOptions = [
    { value: 512, label: '512x512' },
    { value: 1024, label: '1024x1024' },
    { value: 2048, label: '2048x2048' },
    { value: 4096, label: '4096x4096' }
  ];

  const handleExport = async () => {
    setIsExporting(true);
    setExportProgress({
      stage: 'initialization',
      progress: 0,
      message: 'Starting export...',
      messages: []
    });

    try {
      const response = await fetch(`/api/v1/export/${projectId}/advanced`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(exportConfig)
      });

      if (!response.ok) {
        throw new Error('Export failed to start');
      }

      // Start polling for progress
      pollExportProgress();

    } catch (error) {
      console.error('Export failed:', error);
      setExportProgress({
        stage: 'error',
        progress: 0,
        message: `Export failed: ${error}`,
        messages: []
      });
      setIsExporting(false);
    }
  };

  const pollExportProgress = async () => {
    const pollInterval = setInterval(async () => {
      try {
        const response = await fetch(`/api/v1/export/${projectId}/progress`);
        if (response.ok) {
          const progress = await response.json();
          setExportProgress(progress);

          if (progress.stage === 'complete' || progress.stage === 'error') {
            clearInterval(pollInterval);
            setIsExporting(false);
            
            if (progress.stage === 'complete') {
              // Fetch exported files
              const filesResponse = await fetch(`/api/v1/projects/${projectId}`);
              if (filesResponse.ok) {
                const project = await filesResponse.json();
                setExportedFiles(project.mesh_files || {});
                onExportComplete?.(project.mesh_files || {});
              }
            }
          }
        }
      } catch (error) {
        console.error('Failed to poll export progress:', error);
        clearInterval(pollInterval);
        setIsExporting(false);
      }
    }, 1000);
  };

  const downloadFile = async (format: string) => {
    try {
      const response = await fetch(`/api/v1/projects/${projectId}/download-mesh?format=${format}`);
      if (response.ok) {
        const blob = await response.blob();
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `nerf_model.${format}`;
        a.click();
        URL.revokeObjectURL(url);
      }
    } catch (error) {
      console.error('Download failed:', error);
    }
  };

  const downloadAll = async () => {
    try {
      const response = await fetch(`/api/v1/projects/${projectId}/download-all-meshes`);
      if (response.ok) {
        const blob = await response.blob();
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `nerf_meshes_${projectId}.zip`;
        a.click();
        URL.revokeObjectURL(url);
      }
    } catch (error) {
      console.error('Download failed:', error);
    }
  };

  const getProgressColor = (stage: string) => {
    switch (stage) {
      case 'complete': return 'text-green-600';
      case 'error': return 'text-red-600';
      case 'initialization': return 'text-blue-600';
      case 'mesh_extraction': return 'text-purple-600';
      case 'optimization': return 'text-orange-600';
      case 'texture_baking': return 'text-pink-600';
      case 'export': return 'text-indigo-600';
      case 'compression': return 'text-teal-600';
      default: return 'text-gray-600';
    }
  };

  return (
    <div className="bg-white rounded-lg shadow-lg p-6">
      <h3 className="text-xl font-semibold mb-6">Advanced Export Manager</h3>

      {/* Export Configuration */}
      <div className="space-y-6">
        {/* Format Selection */}
        <div>
          <label className="block text-sm font-medium mb-3">Export Format</label>
          <div className="grid grid-cols-1 gap-3">
            {exportFormats.map(format => (
              <label key={format.id} className="flex items-center p-3 border rounded-lg cursor-pointer hover:bg-gray-50">
                <input
                  type="radio"
                  name="format"
                  value={format.id}
                  checked={exportConfig.format === format.id}
                  onChange={(e) => setExportConfig({...exportConfig, format: e.target.value})}
                  className="mr-3"
                  disabled={isExporting}
                />
                <div>
                  <div className="font-medium">{format.name}</div>
                  <div className="text-sm text-gray-500">{format.description}</div>
                </div>
              </label>
            ))}
          </div>
        </div>

        {/* Quality Settings */}
        <div className="grid grid-cols-2 gap-6">
          <div>
            <label className="block text-sm font-medium mb-2">Quality Level</label>
            <select 
              value={exportConfig.quality}
              onChange={(e) => setExportConfig({...exportConfig, quality: e.target.value})}
              className="w-full p-2 border rounded"
              disabled={isExporting}
            >
              {qualityOptions.map(option => (
                <option key={option.id} value={option.id}>
                  {option.name} - {option.description}
                </option>
              ))}
            </select>
          </div>

          <div>
            <label className="block text-sm font-medium mb-2">Mesh Resolution</label>
            <select 
              value={exportConfig.resolution}
              onChange={(e) => setExportConfig({...exportConfig, resolution: parseInt(e.target.value)})}
              className="w-full p-2 border rounded"
              disabled={isExporting}
            >
              {resolutionOptions.map(option => (
                <option key={option.value} value={option.value}>
                  {option.label}
                </option>
              ))}
            </select>
          </div>
        </div>

        {/* Texture Settings */}
        <div className="grid grid-cols-2 gap-6">
          <div>
            <label className="block text-sm font-medium mb-2">Texture Resolution</label>
            <select 
              value={exportConfig.texture_resolution}
              onChange={(e) => setExportConfig({...exportConfig, texture_resolution: parseInt(e.target.value)})}
              className="w-full p-2 border rounded"
              disabled={isExporting}
            >
              {textureResolutionOptions.map(option => (
                <option key={option.value} value={option.value}>
                  {option.label}
                </option>
              ))}
            </select>
          </div>

          <div>
            <label className="block text-sm font-medium mb-2">Bounds</label>
            <div className="grid grid-cols-3 gap-2">
              {['X', 'Y', 'Z'].map((axis, i) => (
                <div key={axis}>
                  <label className="block text-xs text-gray-600">{axis}</label>
                  <input
                    type="number"
                    value={exportConfig.bounds[i * 2]}
                    onChange={(e) => {
                      const newBounds = [...exportConfig.bounds];
                      newBounds[i * 2] = parseFloat(e.target.value);
                      setExportConfig({...exportConfig, bounds: newBounds});
                    }}
                    className="w-full p-1 border rounded text-sm"
                    disabled={isExporting}
                    step="0.1"
                  />
                  <input
                    type="number"
                    value={exportConfig.bounds[i * 2 + 1]}
                    onChange={(e) => {
                      const newBounds = [...exportConfig.bounds];
                      newBounds[i * 2 + 1] = parseFloat(e.target.value);
                      setExportConfig({...exportConfig, bounds: newBounds});
                    }}
                    className="w-full p-1 border rounded text-sm mt-1"
                    disabled={isExporting}
                    step="0.1"
                  />
                </div>
              ))}
            </div>
          </div>
        </div>

        {/* Options */}
        <div className="grid grid-cols-2 gap-6">
          <div className="space-y-3">
            <label className="flex items-center">
              <input
                type="checkbox"
                checked={exportConfig.include_textures}
                onChange={(e) => setExportConfig({...exportConfig, include_textures: e.target.checked})}
                className="mr-2"
                disabled={isExporting}
              />
              <span className="text-sm">Include Textures</span>
            </label>
            <label className="flex items-center">
              <input
                type="checkbox"
                checked={exportConfig.bake_textures}
                onChange={(e) => setExportConfig({...exportConfig, bake_textures: e.target.checked})}
                className="mr-2"
                disabled={isExporting}
              />
              <span className="text-sm">Bake Textures from NeRF</span>
            </label>
          </div>
          <div className="space-y-3">
            <label className="flex items-center">
              <input
                type="checkbox"
                checked={exportConfig.optimize_mesh}
                onChange={(e) => setExportConfig({...exportConfig, optimize_mesh: e.target.checked})}
                className="mr-2"
                disabled={isExporting}
              />
              <span className="text-sm">Optimize Mesh</span>
            </label>
            <label className="flex items-center">
              <input
                type="checkbox"
                checked={exportConfig.compression}
                onChange={(e) => setExportConfig({...exportConfig, compression: e.target.checked})}
                className="mr-2"
                disabled={isExporting}
              />
              <span className="text-sm">Compress Files</span>
            </label>
          </div>
        </div>

        {/* Export Button */}
        <button
          onClick={handleExport}
          disabled={isExporting}
          className="w-full bg-blue-600 text-white py-3 px-4 rounded-lg hover:bg-blue-700 transition-colors disabled:bg-gray-400"
        >
          {isExporting ? 'Exporting...' : 'Start Advanced Export'}
        </button>
      </div>

      {/* Export Progress */}
      {exportProgress && (
        <div className="mt-6 p-4 bg-gray-50 rounded-lg">
          <h4 className="font-medium mb-2">Export Progress</h4>
          <div className="space-y-2">
            <div className="flex justify-between text-sm">
              <span className={getProgressColor(exportProgress.stage)}>
                {exportProgress.stage.replace('_', ' ').toUpperCase()}
              </span>
              <span>{Math.round(exportProgress.progress * 100)}%</span>
            </div>
            <div className="w-full bg-gray-200 rounded-full h-2">
              <div 
                className="bg-blue-600 h-2 rounded-full transition-all duration-300" 
                style={{ width: `${exportProgress.progress * 100}%` }}
              />
            </div>
            <p className="text-sm text-gray-600">{exportProgress.message}</p>
            
            {/* Progress Messages */}
            {exportProgress.messages.length > 0 && (
              <div className="mt-2">
                <details className="text-sm">
                  <summary className="cursor-pointer text-gray-600">Show Details</summary>
                  <div className="mt-1 space-y-1">
                    {exportProgress.messages.map((msg, i) => (
                      <div key={i} className="text-xs text-gray-500">{msg}</div>
                    ))}
                  </div>
                </details>
              </div>
            )}
          </div>
        </div>
      )}

      {/* Exported Files */}
      {Object.keys(exportedFiles).length > 0 && (
        <div className="mt-6">
          <h4 className="font-medium mb-3">Exported Files</h4>
          <div className="space-y-2">
            {Object.entries(exportedFiles).map(([format, path]) => (
              <div key={format} className="flex items-center justify-between p-2 bg-gray-50 rounded">
                <span className="text-sm font-medium">{format.toUpperCase()}</span>
                <button
                  onClick={() => downloadFile(format)}
                  className="px-3 py-1 bg-green-600 text-white rounded hover:bg-green-700 text-sm"
                >
                  Download
                </button>
              </div>
            ))}
            <button
              onClick={downloadAll}
              className="w-full mt-3 px-4 py-2 bg-purple-600 text-white rounded hover:bg-purple-700"
            >
              Download All as ZIP
            </button>
          </div>
        </div>
      )}
    </div>
  );
} 