import React from 'react';

interface SystemMetrics {
  cpu_percent?: number;
  memory_percent?: number;
  gpu_utilization?: number;
}

interface TrainingMetrics {
  step?: number;
  loss?: number;
  psnr?: number;
  lr?: number;
}

interface PerformanceDashboardProps {
  systemMetrics: SystemMetrics | null;
  trainingMetrics: TrainingMetrics | null;
}

const PerformanceDashboard: React.FC<PerformanceDashboardProps> = ({ systemMetrics, trainingMetrics }) => {
  return (
    <div className="bg-white p-6 rounded-lg shadow-md">
      <h2 className="text-xl font-bold mb-4">Performance Dashboard</h2>
      <div className="space-y-4">
        <div className="bg-gray-50 p-4 rounded-lg">
          <h3 className="text-lg font-semibold mb-3 text-gray-800">System Metrics</h3>
          {systemMetrics ? (
            <div className="space-y-2">
              <div className="flex justify-between items-center">
                <span className="text-sm text-gray-600">CPU Usage:</span>
                <span className="font-mono text-sm">{systemMetrics.cpu_percent?.toFixed(1) || 'N/A'}%</span>
              </div>
              <div className="flex justify-between items-center">
                <span className="text-sm text-gray-600">Memory:</span>
                <span className="font-mono text-sm">{systemMetrics.memory_percent?.toFixed(1) || 'N/A'}%</span>
              </div>
              <div className="flex justify-between items-center">
                <span className="text-sm text-gray-600">GPU:</span>
                <span className="font-mono text-sm">{systemMetrics.gpu_utilization?.toFixed(1) || 'N/A'}%</span>
              </div>
            </div>
          ) : (
            <p className="text-gray-500 text-sm">Loading system metrics...</p>
          )}
        </div>
        
        <div className="bg-gray-50 p-4 rounded-lg">
          <h3 className="text-lg font-semibold mb-3 text-gray-800">Training Metrics</h3>
          {trainingMetrics ? (
            <div className="space-y-2">
              <div className="flex justify-between items-center">
                <span className="text-sm text-gray-600">Step:</span>
                <span className="font-mono text-sm">{trainingMetrics.step || 0}</span>
              </div>
              <div className="flex justify-between items-center">
                <span className="text-sm text-gray-600">Loss:</span>
                <span className="font-mono text-sm">{trainingMetrics.loss?.toFixed(4) || 'N/A'}</span>
              </div>
              <div className="flex justify-between items-center">
                <span className="text-sm text-gray-600">PSNR:</span>
                <span className="font-mono text-sm">{trainingMetrics.psnr?.toFixed(2) || 'N/A'} dB</span>
              </div>
              <div className="flex justify-between items-center">
                <span className="text-sm text-gray-600">Learning Rate:</span>
                <span className="font-mono text-sm">{trainingMetrics.lr?.toExponential(2) || 'N/A'}</span>
              </div>
            </div>
          ) : (
            <p className="text-gray-500 text-sm">No training in progress.</p>
          )}
        </div>
      </div>
    </div>
  );
};

export default PerformanceDashboard;