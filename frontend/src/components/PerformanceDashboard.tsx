import React, { useState, useEffect } from 'react';
import { Line, Bar } from 'react-chartjs-2';

interface SystemMetrics {
  timestamp: number;
  cpu_percent: number;
  memory_percent: number;
  memory_available: number;
  memory_total: number;
  gpu_memory_used?: number;
  gpu_memory_total?: number;
  gpu_utilization?: number;
  disk_usage_percent: number;
  network_io?: {
    bytes_sent: number;
    bytes_recv: number;
    packets_sent: number;
    packets_recv: number;
  };
}

interface TrainingMetrics {
  timestamp: number;
  step: number;
  loss: number;
  psnr: number;
  learning_rate: number;
  batch_size: number;
  samples_per_second: number;
  memory_allocated: number;
  memory_reserved: number;
  gpu_memory_used: number;
  render_time: number;
  total_time: number;
}

interface PerformanceAlert {
  timestamp: string;
  alert: {
    metric: string;
    threshold: number;
    condition: string;
    severity: string;
    message: string;
    action?: string;
  };
  metric_value: number;
  system_metrics: any;
}

interface PerformanceBaselines {
  baselines: Record<string, number>;
  regression_thresholds: Record<string, number>;
}

export function PerformanceDashboard() {
  const [systemMetrics, setSystemMetrics] = useState<SystemMetrics | null>(null);
  const [trainingMetrics, setTrainingMetrics] = useState<TrainingMetrics | null>(null);
  const [metricsHistory, setMetricsHistory] = useState<SystemMetrics[]>([]);
  const [alerts, setAlerts] = useState<PerformanceAlert[]>([]);
  const [baselines, setBaselines] = useState<PerformanceBaselines | null>(null);
  const [isMonitoring, setIsMonitoring] = useState(false);
  const [refreshInterval, setRefreshInterval] = useState(5000);

  // Fetch system metrics
  const fetchSystemMetrics = async () => {
    try {
      const response = await fetch('/api/v1/system/metrics');
      if (response.ok) {
        const data = await response.json();
        setSystemMetrics(data);
        setMetricsHistory(prev => [...prev.slice(-50), data]); // Keep last 50 metrics
      }
    } catch (error) {
      console.error('Failed to fetch system metrics:', error);
    }
  };

  // Fetch training metrics
  const fetchTrainingMetrics = async () => {
    try {
      const response = await fetch('/api/v1/training/metrics');
      if (response.ok) {
        const data = await response.json();
        setTrainingMetrics(data);
      }
    } catch (error) {
      console.error('Failed to fetch training metrics:', error);
    }
  };

  // Fetch performance alerts
  const fetchAlerts = async () => {
    try {
      const response = await fetch('/api/v1/performance/alerts');
      if (response.ok) {
        const data = await response.json();
        setAlerts(data.alerts || []);
      }
    } catch (error) {
      console.error('Failed to fetch alerts:', error);
    }
  };

  // Fetch performance baselines
  const fetchBaselines = async () => {
    try {
      const response = await fetch('/api/v1/performance/baselines');
      if (response.ok) {
        const data = await response.json();
        setBaselines(data);
      }
    } catch (error) {
      console.error('Failed to fetch baselines:', error);
    }
  };

  // Add performance alert
  const addAlert = async (alert: any) => {
    try {
      const response = await fetch('/api/v1/performance/alerts', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(alert)
      });
      if (response.ok) {
        fetchAlerts(); // Refresh alerts
      }
    } catch (error) {
      console.error('Failed to add alert:', error);
    }
  };

  // Run performance test
  const runPerformanceTest = async () => {
    try {
      const response = await fetch('/api/v1/performance/test', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ duration: 60 })
      });
      if (response.ok) {
        const results = await response.json();
        console.log('Performance test results:', results);
      }
    } catch (error) {
      console.error('Performance test failed:', error);
    }
  };

  // Start/stop monitoring
  useEffect(() => {
    if (isMonitoring) {
      const interval = setInterval(() => {
        fetchSystemMetrics();
        fetchTrainingMetrics();
        fetchAlerts();
      }, refreshInterval);

      return () => clearInterval(interval);
    }
  }, [isMonitoring, refreshInterval]);

  // Initial fetch
  useEffect(() => {
    fetchSystemMetrics();
    fetchTrainingMetrics();
    fetchAlerts();
    fetchBaselines();
  }, []);

  // Chart data for system metrics
  const systemChartData = {
    labels: metricsHistory.map((_, i) => i),
    datasets: [
      {
        label: 'CPU %',
        data: metricsHistory.map(m => m.cpu_percent),
        borderColor: 'rgb(255, 99, 132)',
        backgroundColor: 'rgba(255, 99, 132, 0.1)',
        tension: 0.1
      },
      {
        label: 'Memory %',
        data: metricsHistory.map(m => m.memory_percent),
        borderColor: 'rgb(54, 162, 235)',
        backgroundColor: 'rgba(54, 162, 235, 0.1)',
        tension: 0.1
      },
      {
        label: 'GPU Memory %',
        data: metricsHistory.map(m => m.gpu_memory_used ? (m.gpu_memory_used / (m.gpu_memory_total || 1)) * 100 : 0),
        borderColor: 'rgb(75, 192, 192)',
        backgroundColor: 'rgba(75, 192, 192, 0.1)',
        tension: 0.1
      }
    ]
  };

  const getSeverityColor = (severity: string) => {
    switch (severity) {
      case 'critical': return 'text-red-600 bg-red-100';
      case 'error': return 'text-red-600 bg-red-50';
      case 'warning': return 'text-yellow-600 bg-yellow-50';
      default: return 'text-gray-600 bg-gray-50';
    }
  };

  const getMetricStatus = (current: number, baseline: number, threshold: number) => {
    const ratio = current / baseline;
    if (ratio < threshold) return 'text-red-600';
    if (ratio < 0.9) return 'text-yellow-600';
    return 'text-green-600';
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex justify-between items-center">
        <h2 className="text-2xl font-bold">Performance Dashboard</h2>
        <div className="flex items-center space-x-4">
          <label className="flex items-center">
            <input
              type="checkbox"
              checked={isMonitoring}
              onChange={(e) => setIsMonitoring(e.target.checked)}
              className="mr-2"
            />
            <span className="text-sm">Auto-refresh</span>
          </label>
          <select
            value={refreshInterval}
            onChange={(e) => setRefreshInterval(parseInt(e.target.value))}
            className="text-sm border rounded px-2 py-1"
          >
            <option value={1000}>1s</option>
            <option value={5000}>5s</option>
            <option value={10000}>10s</option>
            <option value={30000}>30s</option>
          </select>
          <button
            onClick={runPerformanceTest}
            className="px-4 py-2 bg-purple-600 text-white rounded hover:bg-purple-700 text-sm"
          >
            Run Test
          </button>
        </div>
      </div>

      {/* Current Metrics */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        {/* CPU Usage */}
        <div className="bg-white p-4 rounded-lg shadow">
          <h3 className="text-sm font-medium text-gray-500">CPU Usage</h3>
          <div className="mt-2 flex items-baseline">
            <span className="text-2xl font-semibold">
              {systemMetrics?.cpu_percent?.toFixed(1) || '0'}%
            </span>
          </div>
          <div className="mt-2">
            <div className="w-full bg-gray-200 rounded-full h-2">
              <div
                className="bg-blue-600 h-2 rounded-full"
                style={{ width: `${systemMetrics?.cpu_percent || 0}%` }}
              />
            </div>
          </div>
        </div>

        {/* Memory Usage */}
        <div className="bg-white p-4 rounded-lg shadow">
          <h3 className="text-sm font-medium text-gray-500">Memory Usage</h3>
          <div className="mt-2 flex items-baseline">
            <span className="text-2xl font-semibold">
              {systemMetrics?.memory_percent?.toFixed(1) || '0'}%
            </span>
          </div>
          <div className="mt-2">
            <div className="w-full bg-gray-200 rounded-full h-2">
              <div
                className="bg-green-600 h-2 rounded-full"
                style={{ width: `${systemMetrics?.memory_percent || 0}%` }}
              />
            </div>
          </div>
          <div className="text-xs text-gray-500 mt-1">
            {((systemMetrics?.memory_total || 0) / 1024**3).toFixed(1)} GB total
          </div>
        </div>

        {/* GPU Memory */}
        <div className="bg-white p-4 rounded-lg shadow">
          <h3 className="text-sm font-medium text-gray-500">GPU Memory</h3>
          <div className="mt-2 flex items-baseline">
            <span className="text-2xl font-semibold">
              {systemMetrics?.gpu_memory_used?.toFixed(1) || '0'} GB
            </span>
          </div>
          <div className="mt-2">
            <div className="w-full bg-gray-200 rounded-full h-2">
              <div
                className="bg-purple-600 h-2 rounded-full"
                style={{ 
                  width: `${systemMetrics?.gpu_memory_used && systemMetrics?.gpu_memory_total 
                    ? (systemMetrics.gpu_memory_used / systemMetrics.gpu_memory_total) * 100 
                    : 0}%` 
                }}
              />
            </div>
          </div>
          <div className="text-xs text-gray-500 mt-1">
            {systemMetrics?.gpu_memory_total?.toFixed(1) || '0'} GB total
          </div>
        </div>

        {/* Training Speed */}
        <div className="bg-white p-4 rounded-lg shadow">
          <h3 className="text-sm font-medium text-gray-500">Training Speed</h3>
          <div className="mt-2 flex items-baseline">
            <span className="text-2xl font-semibold">
              {trainingMetrics?.samples_per_second?.toFixed(1) || '0'}
            </span>
            <span className="text-sm text-gray-500 ml-1">samples/s</span>
          </div>
          <div className="text-xs text-gray-500 mt-1">
            Loss: {trainingMetrics?.loss?.toFixed(4) || '0.0000'}
          </div>
          <div className="text-xs text-gray-500">
            PSNR: {trainingMetrics?.psnr?.toFixed(2) || '0.00'} dB
          </div>
        </div>
      </div>

      {/* Charts */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* System Metrics Chart */}
        <div className="bg-white p-4 rounded-lg shadow">
          <h3 className="text-lg font-medium mb-4">System Metrics Over Time</h3>
          <Line
            data={systemChartData}
            options={{
              responsive: true,
              scales: {
                y: {
                  beginAtZero: true,
                  max: 100
                }
              },
              plugins: {
                legend: {
                  position: 'top' as const,
                }
              }
            }}
          />
        </div>

        {/* Performance Baselines */}
        <div className="bg-white p-4 rounded-lg shadow">
          <h3 className="text-lg font-medium mb-4">Performance Baselines</h3>
          {baselines ? (
            <div className="space-y-3">
              {Object.entries(baselines.baselines).map(([metric, value]) => (
                <div key={metric} className="flex justify-between items-center">
                  <span className="text-sm font-medium">{metric.replace('_', ' ')}</span>
                  <span className="text-sm">{value.toFixed(2)}</span>
                </div>
              ))}
              <div className="border-t pt-3 mt-3">
                <h4 className="text-sm font-medium mb-2">Regression Thresholds</h4>
                {Object.entries(baselines.regression_thresholds).map(([metric, threshold]) => (
                  <div key={metric} className="flex justify-between items-center text-sm">
                    <span>{metric.replace('_', ' ')}</span>
                    <span>{threshold * 100}%</span>
                  </div>
                ))}
              </div>
            </div>
          ) : (
            <p className="text-gray-500">No baseline data available</p>
          )}
        </div>
      </div>

      {/* Alerts */}
      <div className="bg-white p-4 rounded-lg shadow">
        <div className="flex justify-between items-center mb-4">
          <h3 className="text-lg font-medium">Performance Alerts</h3>
          <button
            onClick={() => addAlert({
              metric: 'cpu_percent',
              threshold: 90,
              condition: 'above',
              severity: 'warning',
              message: 'High CPU usage detected'
            })}
            className="px-3 py-1 bg-blue-600 text-white rounded hover:bg-blue-700 text-sm"
          >
            Add Test Alert
          </button>
        </div>
        
        {alerts.length > 0 ? (
          <div className="space-y-2">
            {alerts.map((alert, index) => (
              <div key={index} className={`p-3 rounded ${getSeverityColor(alert.alert.severity)}`}>
                <div className="flex justify-between items-start">
                  <div>
                    <div className="font-medium">{alert.alert.message}</div>
                    <div className="text-sm">
                      {alert.alert.metric}: {alert.metric_value.toFixed(2)} 
                      {alert.alert.condition} {alert.alert.threshold}
                    </div>
                    <div className="text-xs text-gray-500">
                      {new Date(alert.timestamp).toLocaleString()}
                    </div>
                  </div>
                  <span className={`px-2 py-1 rounded text-xs font-medium ${
                    alert.alert.severity === 'critical' ? 'bg-red-200 text-red-800' :
                    alert.alert.severity === 'error' ? 'bg-red-100 text-red-700' :
                    'bg-yellow-100 text-yellow-700'
                  }`}>
                    {alert.alert.severity.toUpperCase()}
                  </span>
                </div>
              </div>
            ))}
          </div>
        ) : (
          <p className="text-gray-500">No active alerts</p>
        )}
      </div>

      {/* Training Metrics Details */}
      {trainingMetrics && (
        <div className="bg-white p-4 rounded-lg shadow">
          <h3 className="text-lg font-medium mb-4">Training Performance Details</h3>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            <div>
              <div className="text-sm text-gray-500">Step</div>
              <div className="text-lg font-semibold">{trainingMetrics.step}</div>
            </div>
            <div>
              <div className="text-sm text-gray-500">Learning Rate</div>
              <div className="text-lg font-semibold">{trainingMetrics.learning_rate.toExponential(2)}</div>
            </div>
            <div>
              <div className="text-sm text-gray-500">Batch Size</div>
              <div className="text-lg font-semibold">{trainingMetrics.batch_size}</div>
            </div>
            <div>
              <div className="text-sm text-gray-500">Render Time</div>
              <div className="text-lg font-semibold">{trainingMetrics.render_time.toFixed(3)}s</div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
} 