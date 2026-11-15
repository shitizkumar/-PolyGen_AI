import { Canvas } from "@react-three/fiber";
import { OrbitControls, Environment } from "@react-three/drei";
import { Suspense, useRef } from "react";
import { useFrame } from "@react-three/fiber";

function SampleMesh() {
  const meshRef = useRef();

  useFrame((_, delta) => {
    if (meshRef.current) {
      meshRef.current.rotation.y += delta * 0.35;
    }
  });

  return (
    <mesh ref={meshRef}>
      <torusKnotGeometry args={[1, 0.3, 128, 16]} />
      <meshStandardMaterial
        color="#7c8cff"
        metalness={0.55}
        roughness={0.25}
      />
    </mesh>
  );
}

export default function ModelViewer() {
  return (
    <Canvas camera={{ position: [0, 1, 5], fov: 45 }}>
      <color attach="background" args={["#101827"]} />
      <ambientLight intensity={1.2} />
      <directionalLight position={[4, 6, 5]} intensity={2} />
      <Suspense fallback={null}>
        <SampleMesh />
        <Environment preset="city" />
      </Suspense>
      <OrbitControls enableDamping />
    </Canvas>
  );
}
