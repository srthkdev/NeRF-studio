import { Link, useLocation } from 'react-router-dom';
import { motion } from 'framer-motion';
import { Github, FileText, Home, BookOpen } from 'lucide-react';

const Header = () => {
  const location = useLocation();

  const navItems = [
    { path: '/', label: 'Dashboard', icon: Home },
    { path: '/docs', label: 'Documentation', icon: BookOpen },
  ];

  return (
    <motion.header 
      initial={{ y: -100 }}
      animate={{ y: 0 }}
      className="bg-white/80 backdrop-blur-md border-b border-gray-200 sticky top-0 z-50"
    >
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex justify-between items-center py-4">
          {/* Logo and Title */}
          <Link to="/" className="flex items-center space-x-3 group">
            <div className="w-10 h-10 bg-gradient-to-br from-blue-500 to-purple-600 rounded-lg flex items-center justify-center">
              <span className="text-white font-bold text-lg">N</span>
            </div>
            <div>
              <h1 className="text-xl font-bold text-gray-900 group-hover:text-blue-600 transition-colors">
                NeRF Studio
              </h1>
              <p className="text-sm text-gray-500">Neural Radiance Fields Platform</p>
            </div>
          </Link>

          {/* Navigation */}
          <nav className="hidden md:flex items-center space-x-8">
            {navItems.map((item) => {
              const Icon = item.icon;
              const isActive = location.pathname === item.path;
              
              return (
                <Link
                  key={item.path}
                  to={item.path}
                  className={`flex items-center space-x-2 px-3 py-2 rounded-lg transition-all duration-200 ${
                    isActive
                      ? 'bg-blue-50 text-blue-600 border border-blue-200'
                      : 'text-gray-600 hover:text-blue-600 hover:bg-gray-50'
                  }`}
                >
                  <Icon size={18} />
                  <span className="font-medium">{item.label}</span>
                </Link>
              );
            })}
          </nav>

          {/* External Links */}
          <div className="flex items-center space-x-4">
            <a
              href="https://arxiv.org/pdf/2003.08934"
              target="_blank"
              rel="noopener noreferrer"
              className="flex items-center space-x-2 px-3 py-2 text-gray-600 hover:text-blue-600 hover:bg-gray-50 rounded-lg transition-all duration-200"
            >
              <FileText size={18} />
              <span className="hidden sm:inline font-medium">NeRF Paper</span>
            </a>
            
            <a
              href="https://github.com/srthkdev/NeRF-studio"
              target="_blank"
              rel="noopener noreferrer"
              className="flex items-center space-x-2 px-3 py-2 text-gray-600 hover:text-gray-900 hover:bg-gray-50 rounded-lg transition-all duration-200"
            >
              <Github size={18} />
              <span className="hidden sm:inline font-medium">GitHub</span>
            </a>
          </div>
        </div>
      </div>
    </motion.header>
  );
};

export default Header; 