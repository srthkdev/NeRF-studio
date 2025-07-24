
import { useState } from 'react';

const API_URL = 'http://localhost:8000/api/v1';

interface AdvancedExportManagerProps {
  projectId: string;
}

const AdvancedExportManager = ({ projectId }: AdvancedExportManagerProps) => {
  const [exportOptions, setExportOptions] = useState({
    format: 'gltf',
    resolution: 128,
    include_textures: true,
    texture_resolution: 1024,
    optimize_mesh: true,
  });
  const [isExporting, setIsExporting] = useState(false);
  const [exportStatus, setExportStatus] = useState('');

  const handleExport = async () => {
    setIsExporting(true);
    setExportStatus('Starting export...');
    
    try {
      const response = await fetch(`${API_URL}/projects/${projectId}/export/advanced`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(exportOptions),
      });
      const data = await response.json();
      
      if (response.ok) {
        setExportStatus(`Export started: ${data.export_job_id}`);
        console.log('Export started:', data);
      } else {
        setExportStatus(`Export failed: ${data.detail || 'Unknown error'}`);
      }
    } catch (error) {
      setExportStatus(`Export error: ${error instanceof Error ? error.message : 'Unknown error'}`);
    } finally {
      setIsExporting(false);
    }
  };

  return (
    <div className="bg-white p-6 rounded-lg shadow-md">
      <h2 className="text-xl font-bold mb-4">Advanced Export</h2>
      <div className="space-y-4">
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-1">
            Format:
          </label>
          <select 
            value={exportOptions.format} 
            onChange={(e) => setExportOptions({ ...exportOptions, format: e.target.value })}
            className="w-full border border-gray-300 rounded-md px-3 py-2 focus:outline-none focus:ring-2 focus:ring-blue-500"
          >
            <option value="gltf">GLTF</option>
            <option value="obj">OBJ</option>
            <option value="ply">PLY</option>
          </select>
        </div>
        
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-1">
            Mesh Resolution:
          </label>
          <input 
            type="number" 
            value={exportOptions.resolution} 
            onChange={(e) => setExportOptions({ ...exportOptions, resolution: parseInt(e.target.value) })}
            className="w-full border border-gray-300 rounded-md px-3 py-2 focus:outline-none focus:ring-2 focus:ring-blue-500"
            min="64"
            max="512"
          />
        </div>
        
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-1">
            Texture Resolution:
          </label>
          <input 
            type="number" 
            value={exportOptions.texture_resolution} 
            onChange={(e) => setExportOptions({ ...exportOptions, texture_resolution: parseInt(e.target.value) })}
            className="w-full border border-gray-300 rounded-md px-3 py-2 focus:outline-none focus:ring-2 focus:ring-blue-500"
            min="256"
            max="4096"
          />
        </div>
        
        <div className="space-y-2">
          <label className="flex items-center">
            <input 
              type="checkbox" 
              checked={exportOptions.include_textures} 
              onChange={(e) => setExportOptions({ ...exportOptions, include_textures: e.target.checked })}
              className="mr-2"
            />
            <span className="text-sm text-gray-700">Include Textures</span>
          </label>
          
          <label className="flex items-center">
            <input 
              type="checkbox" 
              checked={exportOptions.optimize_mesh} 
              onChange={(e) => setExportOptions({ ...exportOptions, optimize_mesh: e.target.checked })}
              className="mr-2"
            />
            <span className="text-sm text-gray-700">Optimize Mesh</span>
          </label>
        </div>
        
        <button 
          onClick={handleExport}
          disabled={isExporting}
          className={`w-full py-2 px-4 rounded-md font-medium ${
            isExporting 
              ? 'bg-gray-400 cursor-not-allowed' 
              : 'bg-blue-500 hover:bg-blue-600'
          } text-white transition-colors`}
        >
          {isExporting ? 'Exporting...' : 'Export Model'}
        </button>
        
        {exportStatus && (
          <div className="mt-2 p-2 bg-gray-100 rounded text-sm">
            {exportStatus}
          </div>
        )}
      </div>
    </div>
  );
};

export default AdvancedExportManager;
