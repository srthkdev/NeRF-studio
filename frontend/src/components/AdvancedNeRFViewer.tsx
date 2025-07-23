import React, { useRef, useMemo, useState, useEffect } from 'react';
import { Canvas, useFrame, useThree } from '@react-three/fiber';
import { 
  OrbitControls, 
  Environment, 
  ContactShadows,
  PerspectiveCamera,
  useGLTF,
  Lightformer,
  Text
} from '@react-three/drei';
import { Bloom, EffectComposer, SSAO, ToneMapping } from '@react-three/postprocessing';
import * as THREE from 'three';

interface NeRFViewerProps {
  modelData?: any;
  quality?: 'low' | 'medium' | 'high';
  enableLOD?: boolean;
  showFrustums?: boolean;
  showAxes?: boolean;
  showGrid?: boolean;
}

// Custom volume rendering shader for NeRF visualization
const volumeShader = {
  uniforms: {
    densityTexture: { value: null },
    colorTexture: { value: null },
    cameraPosition: { value: new THREE.Vector3() },
    stepSize: { value: 0.01 },
    opacityThreshold: { value: 0.01 },
    quality: { value: 1.0 },
    bounds: { value: new THREE.Vector4(-1, 1, -1, 1) }
  },
  vertexShader: `
    varying vec3 vPosition;
    varying vec3 vNormal;
    varying vec3 vWorldPosition;
    
    void main() {
      vPosition = position;
      vNormal = normal;
      vWorldPosition = (modelMatrix * vec4(position, 1.0)).xyz;
      gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
    }
  `,
  fragmentShader: `
    uniform sampler3D densityTexture;
    uniform sampler3D colorTexture;
    uniform vec3 cameraPosition;
    uniform float stepSize;
    uniform float opacityThreshold;
    uniform float quality;
    uniform vec4 bounds;
    
    varying vec3 vPosition;
    varying vec3 vNormal;
    varying vec3 vWorldPosition;
    
    vec3 getRayDirection(vec3 worldPos) {
      return normalize(worldPos - cameraPosition);
    }
    
    bool intersectBox(vec3 rayOrigin, vec3 rayDir, vec3 boxMin, vec3 boxMax, out float tNear, out float tFar) {
      vec3 invDir = 1.0 / rayDir;
      vec3 tMin = (boxMin - rayOrigin) * invDir;
      vec3 tMax = (boxMax - rayOrigin) * invDir;
      
      vec3 t1 = min(tMin, tMax);
      vec3 t2 = max(tMin, tMax);
      
      tNear = max(max(t1.x, t1.y), t1.z);
      tFar = min(min(t2.x, t2.y), t2.z);
      
      return tFar >= tNear && tFar > 0.0;
    }
    
    vec3 worldToTexture(vec3 worldPos) {
      vec3 normalized = (worldPos - vec3(bounds.x, bounds.z, bounds.x)) / 
                       (vec3(bounds.y, bounds.w, bounds.y) - vec3(bounds.x, bounds.z, bounds.x));
      return normalized * 0.5 + 0.5;
    }
    
    void main() {
      vec3 rayOrigin = cameraPosition;
      vec3 rayDir = getRayDirection(vWorldPosition);
      
      vec3 boxMin = vec3(bounds.x, bounds.z, bounds.x);
      vec3 boxMax = vec3(bounds.y, bounds.w, bounds.y);
      
      float tNear, tFar;
      if (!intersectBox(rayOrigin, rayDir, boxMin, boxMax, tNear, tFar)) {
        discard;
      }
      
      vec3 startPos = rayOrigin + rayDir * tNear;
      vec3 endPos = rayOrigin + rayDir * tFar;
      
      float rayLength = length(endPos - startPos);
      int numSteps = int(rayLength / stepSize * quality);
      
      vec4 color = vec4(0.0);
      float transmittance = 1.0;
      
      for (int i = 0; i < numSteps; i++) {
        float t = float(i) / float(numSteps - 1);
        vec3 samplePos = mix(startPos, endPos, t);
        vec3 texCoord = worldToTexture(samplePos);
        
        // Check if we're within texture bounds
        if (any(lessThan(texCoord, vec3(0.0))) || any(greaterThan(texCoord, vec3(1.0)))) {
          continue;
        }
        
        float density = texture(densityTexture, texCoord).r;
        
        if (density > opacityThreshold) {
          vec3 sampleColor = texture(colorTexture, texCoord).rgb;
          float alpha = 1.0 - exp(-density * stepSize);
          
          color.rgb += transmittance * alpha * sampleColor;
          color.a += transmittance * alpha;
          
          transmittance *= (1.0 - alpha);
          
          if (transmittance < 0.01) break;
        }
      }
      
      gl_FragColor = color;
    }
  `
};

// Level of Detail component
function LODController({ children, distance }: { children: React.ReactNode; distance: number }) {
  const { camera } = useThree();
  const meshRef = useRef<THREE.Group>(null);
  const [lodLevel, setLodLevel] = useState(0);
  
  useFrame(() => {
    if (meshRef.current) {
      const dist = camera.position.distanceTo(meshRef.current.position);
      const newLodLevel = Math.min(Math.floor(dist / distance), 3);
      if (newLodLevel !== lodLevel) {
        setLodLevel(newLodLevel);
      }
    }
  });
  
  return (
    <group ref={meshRef}>
      {React.Children.map(children, (child, index) => {
        if (index === lodLevel) {
          return child;
        }
        return null;
      })}
    </group>
  );
}

// NeRF Volume Renderer Component
function NeRFVolumeRenderer({ modelData, quality }: { modelData: any; quality: string }) {
  const meshRef = useRef<THREE.Mesh>(null);
  const { camera } = useThree();
  
  // Quality settings
  const qualitySettings = useMemo(() => ({
    low: { stepSize: 0.02, opacityThreshold: 0.05, quality: 0.5 },
    medium: { stepSize: 0.01, opacityThreshold: 0.01, quality: 1.0 },
    high: { stepSize: 0.005, opacityThreshold: 0.005, quality: 2.0 }
  }), []);
  
  const settings = qualitySettings[quality as keyof typeof qualitySettings] || qualitySettings.medium;
  
  // Create 3D textures for density and color
  const textures = useMemo(() => {
    if (!modelData) return { density: null, color: null };
    
    // Create dummy 3D textures (in real implementation, these would come from the NeRF model)
    const size = 64;
    const densityData = new Float32Array(size * size * size);
    const colorData = new Float32Array(size * size * size * 3);
    
    // Fill with some sample data
    for (let i = 0; i < size * size * size; i++) {
      const x = (i % size) / size;
      const y = Math.floor((i % (size * size)) / size) / size;
      const z = Math.floor(i / (size * size)) / size;
      
      // Create a sphere-like density field
      const dist = Math.sqrt((x - 0.5) ** 2 + (y - 0.5) ** 2 + (z - 0.5) ** 2);
      densityData[i] = Math.max(0, 1 - dist * 2);
      
      // Create color based on position
      colorData[i * 3] = x;
      colorData[i * 3 + 1] = y;
      colorData[i * 3 + 2] = z;
    }
    
    const densityTexture = new THREE.DataTexture3D(densityData, size, size, size);
    densityTexture.format = THREE.RedFormat;
    densityTexture.type = THREE.FloatType;
    densityTexture.needsUpdate = true;
    
    const colorTexture = new THREE.DataTexture3D(colorData, size, size, size);
    colorTexture.format = THREE.RGBFormat;
    colorTexture.type = THREE.FloatType;
    colorTexture.needsUpdate = true;
    
    return { density: densityTexture, color: colorTexture };
  }, [modelData]);
  
  // Update shader uniforms
  useFrame(() => {
    if (meshRef.current && meshRef.current.material) {
      const material = meshRef.current.material as THREE.ShaderMaterial;
      material.uniforms.cameraPosition.value.copy(camera.position);
      material.uniforms.stepSize.value = settings.stepSize;
      material.uniforms.opacityThreshold.value = settings.opacityThreshold;
      material.uniforms.quality.value = settings.quality;
      
      if (textures.density) {
        material.uniforms.densityTexture.value = textures.density;
      }
      if (textures.color) {
        material.uniforms.colorTexture.value = textures.color;
      }
    }
  });
  
  return (
    <mesh ref={meshRef} castShadow receiveShadow>
      <boxGeometry args={[2, 2, 2]} />
      <shaderMaterial 
        {...volumeShader}
        transparent={true}
        side={THREE.DoubleSide}
        uniforms={{
          ...volumeShader.uniforms,
          densityTexture: { value: textures.density },
          colorTexture: { value: textures.color }
        }}
      />
    </mesh>
  );
}

// Camera Frustum Visualization
function CameraFrustum({ position, rotation, color = "orange" }: {
  position: [number, number, number];
  rotation: [number, number, number];
  color?: string;
}) {
  return (
    <group position={position} rotation={rotation}>
      {/* Camera body */}
      <mesh>
        <boxGeometry args={[0.1, 0.1, 0.05]} />
        <meshStandardMaterial color={color} />
      </mesh>
      
      {/* Camera frustum */}
      <mesh position={[0, 0, 0.1]}>
        <coneGeometry args={[0.05, 0.2, 8]} />
        <meshStandardMaterial color={color} transparent opacity={0.7} />
      </mesh>
    </group>
  );
}

// Coordinate Axes
function CoordinateAxes() {
  return (
    <group>
      {/* Origin point */}
      <mesh position={[0, 0, 0]}>
        <sphereGeometry args={[0.02, 8, 8]} />
        <meshStandardMaterial color="white" />
      </mesh>
      
      {/* X axis */}
      <mesh position={[0.5, 0, 0]}>
        <cylinderGeometry args={[0.005, 0.005, 1]} />
        <meshStandardMaterial color="red" />
      </mesh>
      
      {/* Y axis */}
      <mesh position={[0, 0.5, 0]}>
        <cylinderGeometry args={[0.005, 0.005, 1]} />
        <meshStandardMaterial color="green" />
      </mesh>
      
      {/* Z axis */}
      <mesh position={[0, 0, 0.5]}>
        <cylinderGeometry args={[0.005, 0.005, 1]} />
        <meshStandardMaterial color="blue" />
      </mesh>
      
      {/* Axis labels */}
      <Text position={[1.1, 0, 0]} fontSize={0.1} color="red">X</Text>
      <Text position={[0, 1.1, 0]} fontSize={0.1} color="green">Y</Text>
      <Text position={[0, 0, 1.1]} fontSize={0.1} color="blue">Z</Text>
    </group>
  );
}

// Quality Controls Component
function QualityControls({ quality, onQualityChange }: {
  quality: string;
  onQualityChange: (quality: string) => void;
}) {
  return (
    <div className="absolute top-4 right-4 bg-black bg-opacity-50 text-white p-3 rounded">
      <div className="text-sm font-medium mb-2">Quality</div>
      <div className="space-y-1">
        {(['low', 'medium', 'high'] as const).map((q) => (
          <label key={q} className="flex items-center cursor-pointer">
            <input
              type="radio"
              name="quality"
              value={q}
              checked={quality === q}
              onChange={(e) => onQualityChange(e.target.value)}
              className="mr-2"
            />
            <span className="capitalize">{q}</span>
          </label>
        ))}
      </div>
    </div>
  );
}

// Main Advanced NeRF Viewer Component
export function AdvancedNeRFViewer({ 
  modelData, 
  quality = 'medium', 
  enableLOD = true,
  showFrustums = true,
  showAxes = true,
  showGrid = true
}: NeRFViewerProps) {
  const [currentQuality, setCurrentQuality] = useState(quality);
  
  return (
    <div className="relative w-full h-screen">
      <Canvas 
        shadows 
        gl={{ 
          antialias: true, 
          alpha: false,
          powerPreference: "high-performance",
          logarithmicDepthBuffer: true
        }}
      >
        {/* Advanced Camera Setup */}
        <PerspectiveCamera 
          makeDefault 
          fov={45} 
          position={[3, 3, 3]}
          near={0.1}
          far={1000}
        />
        
        {/* Professional Lighting */}
        <Environment preset="studio" />
        <ambientLight intensity={0.2} />
        <directionalLight 
          position={[10, 10, 5]} 
          intensity={1}
          castShadow
          shadow-mapSize={[2048, 2048]}
        />
        
        {/* NeRF Volume Renderer with LOD */}
        <LODController distance={2}>
          <NeRFVolumeRenderer modelData={modelData} quality={currentQuality} />
          <NeRFVolumeRenderer modelData={modelData} quality="low" />
          <NeRFVolumeRenderer modelData={modelData} quality="low" />
          <NeRFVolumeRenderer modelData={modelData} quality="low" />
        </LODController>
        
        {/* Camera Frustums */}
        {showFrustums && modelData?.cameraPoses?.map((pose: any, i: number) => (
          <CameraFrustum 
            key={i}
            position={pose.position}
            rotation={pose.rotation}
            color={i % 2 === 0 ? "orange" : "yellow"}
          />
        ))}
        
        {/* Coordinate Axes */}
        {showAxes && <CoordinateAxes />}
        
        {/* Grid Helper */}
        {showGrid && <gridHelper args={[10, 10]} />}
        
        {/* Advanced Controls */}
        <OrbitControls 
          enablePan={true}
          enableZoom={true}
          enableRotate={true}
          dampingFactor={0.05}
          minDistance={0.5}
          maxDistance={50}
        />
        
        {/* Contact Shadows */}
        <ContactShadows 
          position={[0, -1, 0]} 
          opacity={0.4} 
          scale={10} 
          blur={2} 
        />
        
        {/* Post-processing Effects */}
        <EffectComposer>
          <Bloom 
            intensity={0.5} 
            luminanceThreshold={0.9} 
            luminanceSmoothing={0.9}
          />
          <SSAO 
            samples={16} 
            radius={0.1} 
            intensity={1} 
          />
          <ToneMapping 
            adaptive={true} 
            resolution={256} 
            middleGrey={0.6} 
            maxLuminance={16.0} 
            averageLuminance={1.0} 
            adaptationRate={1.0} 
          />
        </EffectComposer>
      </Canvas>
      
      {/* Quality Controls */}
      <QualityControls 
        quality={currentQuality} 
        onQualityChange={setCurrentQuality} 
      />
      
      {/* Performance Info */}
      <div className="absolute bottom-4 left-4 bg-black bg-opacity-50 text-white p-2 rounded text-sm">
        <div>Quality: {currentQuality}</div>
        <div>LOD: {enableLOD ? 'Enabled' : 'Disabled'}</div>
        {modelData && <div>Model: Loaded</div>}
      </div>
    </div>
  );
} 