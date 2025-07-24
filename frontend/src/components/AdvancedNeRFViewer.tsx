import React, { useRef } from 'react';
import { Canvas, useThree, useFrame } from '@react-three/fiber';
import { OrbitControls, PerspectiveCamera, Environment, ContactShadows, Text } from '@react-three/drei';
import * as THREE from 'three';

interface NeRFMeshProps {
  projectId: string;
  enableLOD: boolean;
}

function NeRFMesh({ projectId, enableLOD }: NeRFMeshProps) {
  const meshRef = useRef<THREE.Mesh>(null);
  const { camera } = useThree();
  const [isLoading, setIsLoading] = React.useState(true);
  const [hasModel, setHasModel] = React.useState(false);

  // Level of Detail system
  useFrame(() => {
    if (enableLOD && meshRef.current) {
      const distance = camera.position.distanceTo(meshRef.current.position);
      // You would implement logic here to switch mesh resolution based on distance
      // For now, this is a placeholder.
      Math.min(Math.floor(distance / 2), 3); // LOD level calculation
    }
  });

  // Check if model exists for this project
  React.useEffect(() => {
    const checkModel = async () => {
      try {
        const response = await fetch(`http://localhost:8000/api/v1/projects/${projectId}`);
        const project = await response.json();
        // Check if project has completed training
        setHasModel(project.status === 'completed' || project.data?.last_export);
        setIsLoading(false);
      } catch (error) {
        console.error('Error checking model:', error);
        setIsLoading(false);
      }
    };

    if (projectId) {
      checkModel();
    }
  }, [projectId]);

  if (isLoading) {
    return (
      <mesh ref={meshRef}>
        <boxGeometry args={[1, 1, 1]} />
        <meshStandardMaterial color="gray" wireframe />
      </mesh>
    );
  }

  if (!hasModel) {
    return (
      <group>
        <mesh ref={meshRef}>
          <boxGeometry args={[2, 2, 2]} />
          <meshStandardMaterial color="orange" transparent opacity={0.5} />
        </mesh>
        <Text position={[0, 3, 0]} fontSize={0.2} color="white">
          No trained model available
        </Text>
      </group>
    );
  }

  // Placeholder for actual NeRF model rendering
  return (
    <mesh ref={meshRef} castShadow receiveShadow>
      <sphereGeometry args={[1.5, 32, 32]} />
      <meshStandardMaterial color="lightblue" />
    </mesh>
  );
}

interface AdvancedNeRFViewerProps {
  projectId: string;
}

function AdvancedNeRFViewer({ projectId }: AdvancedNeRFViewerProps) {
  return (
    <div className="w-full h-96 border border-gray-300 rounded-lg overflow-hidden">
      <div className="bg-gray-100 p-2 border-b">
        <h3 className="text-lg font-semibold">3D NeRF Viewer</h3>
        <p className="text-sm text-gray-600">Project: {projectId}</p>
      </div>
      <Canvas shadows gl={{ antialias: true, alpha: false, powerPreference: "high-performance" }}>
        <PerspectiveCamera makeDefault fov={45} position={[0, 0, 10]} near={0.1} far={1000} />
        <Environment preset="studio" />
        <ambientLight intensity={0.2} />
        <directionalLight position={[10, 10, 5]} intensity={1} castShadow shadow-mapSize={[2048, 2048]} />
        <NeRFMesh projectId={projectId} enableLOD={true} />
        <OrbitControls enablePan={true} enableZoom={true} enableRotate={true} dampingFactor={0.05} minDistance={1} maxDistance={50} />
        <ContactShadows position={[0, -1, 0]} opacity={0.4} scale={10} blur={2} />
      </Canvas>
    </div>
  );
}

export default AdvancedNeRFViewer;