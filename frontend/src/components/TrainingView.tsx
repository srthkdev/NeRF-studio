import { motion } from 'framer-motion';
import { 
  Activity, 
  TrendingUp, 
  Clock, 
  BarChart3, 
  Zap,
  Target,
  Gauge
} from 'lucide-react';

interface TrainingMetrics {
  step: number;
  loss: number;
  psnr: number;
  lr: number;
}

interface TrainingLogEntry {
  step?: number;
  loss?: number;
  psnr?: number;
  progress?: number;
  status?: string;
  message?: string;
}

interface TrainingViewProps {
  trainingProgress: number;
  trainingLog: TrainingLogEntry[];
  trainingMetrics: TrainingMetrics | null;
}

const TrainingView = ({ trainingProgress, trainingLog, trainingMetrics }: TrainingViewProps) => {
  const isTraining = trainingProgress > 0 && trainingProgress < 100;
  const isCompleted = trainingProgress === 100;

  return (
    <motion.div 
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      className="bg-gradient-to-r from-blue-50 to-purple-50 border border-blue-200 rounded-2xl shadow-lg p-8 mb-8"
    >
      <div className="flex items-center space-x-3 mb-6">
        <div className="w-10 h-10 bg-gradient-to-r from-blue-500 to-purple-600 rounded-lg flex items-center justify-center">
          <Activity className="w-6 h-6 text-white" />
        </div>
        <div>
          <h2 className="text-2xl font-bold text-gray-900">Training Progress</h2>
          <p className="text-gray-600">Real-time NeRF model training</p>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
        {/* Progress Overview */}
        <div className="lg:col-span-2">
          <div className="bg-white rounded-xl p-6 shadow-sm">
            <div className="flex items-center justify-between mb-4">
              <h3 className="text-lg font-semibold text-gray-900">Training Status</h3>
              <div className={`px-3 py-1 rounded-full text-sm font-medium ${
                isCompleted 
                  ? 'bg-green-100 text-green-800' 
                  : isTraining 
                    ? 'bg-blue-100 text-blue-800' 
                    : 'bg-gray-100 text-gray-800'
              }`}>
                {isCompleted ? 'Completed' : isTraining ? 'Training' : 'Ready'}
              </div>
            </div>

            {/* Progress Bar */}
            <div className="space-y-3 mb-6">
              <div className="flex justify-between text-sm">
                <span className="text-gray-600">Progress</span>
                <span className="font-semibold">{trainingProgress}%</span>
              </div>
              <div className="w-full bg-gray-200 rounded-full h-4 overflow-hidden">
                <motion.div
                  className="bg-gradient-to-r from-blue-500 via-purple-500 to-pink-500 h-4 rounded-full"
                  initial={{ width: 0 }}
                  animate={{ width: `${trainingProgress}%` }}
                  transition={{ duration: 0.8, ease: "easeOut" }}
                />
              </div>
            </div>

            {/* Training Metrics */}
            {trainingMetrics && (
              <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                <div className="text-center p-3 bg-gray-50 rounded-lg">
                  <div className="flex items-center justify-center mb-2">
                    <Target className="w-5 h-5 text-blue-500" />
                  </div>
                  <div className="text-2xl font-bold text-gray-900">{trainingMetrics.step}</div>
                  <div className="text-xs text-gray-600">Steps</div>
                </div>
                
                <div className="text-center p-3 bg-gray-50 rounded-lg">
                  <div className="flex items-center justify-center mb-2">
                    <TrendingUp className="w-5 h-5 text-green-500" />
                  </div>
                  <div className="text-2xl font-bold text-gray-900">
                    {trainingMetrics.loss ? trainingMetrics.loss.toFixed(4) : 'N/A'}
                  </div>
                  <div className="text-xs text-gray-600">Loss</div>
                </div>
                
                <div className="text-center p-3 bg-gray-50 rounded-lg">
                  <div className="flex items-center justify-center mb-2">
                    <BarChart3 className="w-5 h-5 text-purple-500" />
                  </div>
                  <div className="text-2xl font-bold text-gray-900">
                    {trainingMetrics.psnr ? trainingMetrics.psnr.toFixed(2) : 'N/A'}
                  </div>
                  <div className="text-xs text-gray-600">PSNR</div>
                </div>
                
                <div className="text-center p-3 bg-gray-50 rounded-lg">
                  <div className="flex items-center justify-center mb-2">
                    <Zap className="w-5 h-5 text-yellow-500" />
                  </div>
                  <div className="text-2xl font-bold text-gray-900">
                    {trainingMetrics.lr ? trainingMetrics.lr.toExponential(2) : 'N/A'}
                  </div>
                  <div className="text-xs text-gray-600">Learning Rate</div>
                </div>
              </div>
            )}
          </div>
        </div>

        {/* Training Log */}
        <div className="bg-white rounded-xl p-6 shadow-sm">
          <div className="flex items-center space-x-2 mb-4">
            <Clock className="w-5 h-5 text-gray-600" />
            <h3 className="text-lg font-semibold text-gray-900">Training Log</h3>
          </div>
          
          <div className="space-y-3 max-h-96 overflow-y-auto">
            {trainingLog.length > 0 ? (
              trainingLog.slice(-10).reverse().map((log, index) => (
                <motion.div
                  key={index}
                  initial={{ opacity: 0, x: -20 }}
                  animate={{ opacity: 1, x: 0 }}
                  transition={{ delay: index * 0.1 }}
                  className="p-3 bg-gray-50 rounded-lg border-l-4 border-blue-500"
                >
                  <div className="text-sm font-mono text-gray-800">
                    {log.step && <div>Step: {log.step}</div>}
                    {log.loss && <div>Loss: {log.loss.toFixed(4)}</div>}
                    {log.psnr && <div>PSNR: {log.psnr.toFixed(2)}</div>}
                    {log.message && <div className="text-gray-600">{log.message}</div>}
                    {log.status && (
                      <div className={`inline-block px-2 py-1 rounded text-xs font-medium ${
                        log.status === 'completed' ? 'bg-green-100 text-green-800' :
                        log.status === 'training' ? 'bg-blue-100 text-blue-800' :
                        'bg-gray-100 text-gray-800'
                      }`}>
                        {log.status}
                      </div>
                    )}
                  </div>
                </motion.div>
              ))
            ) : (
              <div className="text-center py-8 text-gray-500">
                <Gauge className="w-12 h-12 mx-auto mb-3 text-gray-300" />
                <p>No training logs yet</p>
                <p className="text-sm">Start training to see progress</p>
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Training Tips */}
      {isTraining && (
        <motion.div 
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          className="mt-6 bg-blue-50 border border-blue-200 rounded-xl p-4"
        >
          <div className="flex items-start space-x-3">
            <div className="w-6 h-6 bg-blue-500 rounded-full flex items-center justify-center flex-shrink-0 mt-0.5">
              <span className="text-white text-xs font-bold">i</span>
            </div>
            <div>
              <h4 className="font-semibold text-blue-900 mb-1">Training in Progress</h4>
              <p className="text-blue-800 text-sm">
                Your NeRF model is currently training. This process typically takes 10-30 minutes depending on 
                the number of images and scene complexity. You can monitor real-time progress above.
              </p>
            </div>
          </div>
        </motion.div>
      )}

      {isCompleted && (
        <motion.div 
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          className="mt-6 bg-green-50 border border-green-200 rounded-xl p-4"
        >
          <div className="flex items-start space-x-3">
            <div className="w-6 h-6 bg-green-500 rounded-full flex items-center justify-center flex-shrink-0 mt-0.5">
              <span className="text-white text-xs font-bold">✓</span>
            </div>
            <div>
              <h4 className="font-semibold text-green-900 mb-1">Training Completed!</h4>
              <p className="text-green-800 text-sm">
                Your NeRF model has been successfully trained. You can now explore the 3D scene viewer 
                and export your model in various formats.
              </p>
            </div>
          </div>
        </motion.div>
      )}
    </motion.div>
  );
};

export default TrainingView; 