import { Suspense, useMemo, useRef, useState } from "react";
import { Canvas } from "@react-three/fiber";
import { OrbitControls } from "@react-three/drei";
import { OBJLoader } from "three/addons/loaders/OBJLoader.js";
import { Box3, MeshStandardMaterial, Vector3 } from "three";

function SampleModel() {
  return (
    <mesh>
      <torusKnotGeometry args={[1, 0.3, 128, 16]} />
      <meshStandardMaterial
        color="#8b7cff"
        metalness={0.5}
        roughness={0.3}
      />
    </mesh>
  );
}

function ObjModel({ source }) {
  const model = useMemo(() => {
    const object = new OBJLoader().parse(source);

    const bounds = new Box3().setFromObject(object);
    const size = bounds.getSize(new Vector3());
    const center = bounds.getCenter(new Vector3());

    const maxDimension = Math.max(size.x, size.y, size.z);

    if (!Number.isFinite(maxDimension) || maxDimension <= 0) {
      throw new Error("The OBJ file has no usable geometry.");
    }

    object.position.sub(center);

    object.traverse((child) => {
      if (child.isMesh) {
        child.material = new MeshStandardMaterial({
          color: "#a5b4fc",
          metalness: 0.2,
          roughness: 0.65,
          side: 2,
        });
      }
    });

    return {
      object,
      scale: 2.8 / maxDimension,
    };
  }, [source]);

  return (
    <primitive
      object={model.object}
      scale={model.scale}
    />
  );
}

export default function ModelViewer({ objSource = null }) {
  const controlsRef = useRef(null);
  const [autoRotate, setAutoRotate] = useState(false);

  return (
    <div className="model-viewer">
      <div className="viewer-toolbar">
        <button
          type="button"
          onClick={() => controlsRef.current?.reset()}
        >
          Reset View
        </button>

        <button
          type="button"
          aria-pressed={autoRotate}
          onClick={() => setAutoRotate((value) => !value)}
        >
          {autoRotate ? "Stop Rotation" : "Auto Rotate"}
        </button>
      </div>

      <Canvas camera={{ position: [0, 1, 5], fov: 45 }}>
        <color attach="background" args={["#101827"]} />
        <ambientLight intensity={1.5} />
        <directionalLight position={[4, 6, 5]} intensity={2.5} />
        <directionalLight position={[-4, 2, -3]} intensity={1} />

        <Suspense fallback={null}>
          {objSource ? (
            <ObjModel source={objSource} />
          ) : (
            <SampleModel />
          )}
        </Suspense>

        <OrbitControls
          ref={controlsRef}
          makeDefault
          autoRotate={autoRotate}
          autoRotateSpeed={1.5}
          enableRotate
          enableZoom
          enablePan
          enableDamping
          dampingFactor={0.08}
          minDistance={1.5}
          maxDistance={20}
        />
      </Canvas>
    </div>
  );
}