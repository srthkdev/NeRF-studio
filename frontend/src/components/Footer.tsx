import { Heart } from 'lucide-react';

const Footer = () => {
  return (
    <footer className="bg-white/80 backdrop-blur-md border-t border-gray-200 mt-auto">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6">
        <div className="flex flex-col md:flex-row justify-between items-center space-y-4 md:space-y-0">
          <div className="flex items-center space-x-2 text-gray-600">
            <span>Made with</span>
            <Heart size={16} className="text-red-500 fill-current animate-pulse" />
            <span>by</span>
            <a 
              href="https://github.com/srthkdev" 
              target="_blank" 
              rel="noopener noreferrer"
              className="font-semibold text-blue-600 hover:text-blue-800 transition-colors"
            >
              sarthak
            </a>
          </div>
          
          <div className="text-sm text-gray-500">
            <span>NeRF Studio v1.0.0</span>
            <span className="mx-2">•</span>
            <span>Powered by Neural Radiance Fields</span>
          </div>
        </div>
      </div>
    </footer>
  );
};

export default Footer; 