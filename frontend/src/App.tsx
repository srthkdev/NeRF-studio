import React from 'react'
import { Canvas } from '@react-three/fiber'
import { OrbitControls } from '@react-three/drei'
import './App.css'

function App() {
  return (
    <div className="App">
      <header className="bg-gray-900 text-white p-4">
        <h1 className="text-2xl font-bold">NeRF Studio</h1>
        <p className="text-gray-300">Neural Radiance Fields Platform</p>
      </header>
      
      <main className="flex-1 p-4">
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-4 h-full">
          <div className="bg-white rounded-lg shadow-lg p-4">
            <h2 className="text-xl font-semibold mb-4">Project Management</h2>
            <div className="space-y-4">
              <button className="w-full bg-blue-600 text-white py-2 px-4 rounded hover:bg-blue-700">
                Upload Images
              </button>
              <button className="w-full bg-green-600 text-white py-2 px-4 rounded hover:bg-green-700">
                Start Training
              </button>
            </div>
          </div>
          
          <div className="bg-white rounded-lg shadow-lg p-4">
            <h2 className="text-xl font-semibold mb-4">3D Viewer</h2>
            <div className="h-96 border border-gray-200 rounded">
              <Canvas>
                <ambientLight intensity={0.5} />
                <pointLight position={[10, 10, 10]} />
                <mesh>
                  <boxGeometry args={[1, 1, 1]} />
                  <meshStandardMaterial color={'orange'} />
                </mesh>
                <OrbitControls />
              </Canvas>
            </div>
          </div>
        </div>
      </main>
    </div>
  )
}

export default App