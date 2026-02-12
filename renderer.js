import * as THREE from 'three';
import { FilesetResolver, HandLandmarker } from '@mediapipe/tasks-vision';
import { EffectComposer } from 'three/addons/postprocessing/EffectComposer.js';
import { RenderPass } from 'three/addons/postprocessing/RenderPass.js';
import { UnrealBloomPass } from 'three/addons/postprocessing/UnrealBloomPass.js';
import { OutputPass } from 'three/addons/postprocessing/OutputPass.js';

const video = document.getElementById('webcam');
const canvasElement = document.getElementById('output_canvas');
const loading = document.getElementById('loading');

let handLandmarker;
let scene, camera, renderer, composer;
let backgroundPlane;
let handVisualizers = [];
let physicsCube;

const HAND_CONNECTIONS = [
    [0, 1], [1, 2], [2, 3], [3, 4], // Thumb
    [0, 5], [5, 6], [6, 7], [7, 8], // Index
    [9, 10], [10, 11], [11, 12], // Middle
    [13, 14], [14, 15], [15, 16], // Ring
    [0, 17], [17, 18], [18, 19], [19, 20], // Pinky
    [5, 9], [9, 13], [13, 17], // Palm
    [0, 5], [0, 17] // Base
];

// Helper: Map screen-space (0-1) to 3D World Space at a specific Z depth
function screenToWorld(x, y, z, cam, w, h) {
    const vec = new THREE.Vector3((x * 2) - 1, -((y * 2) - 1), 0.5);
    vec.unproject(cam);
    const dir = vec.sub(cam.position).normalize();
    const distance = (z - cam.position.z) / dir.z;
    return cam.position.clone().add(dir.multiplyScalar(distance));
}

const GLOBAL_Z_DEPTH = 20; // Match background plane (Camera at 100, Dist 80 = Z 20)

class HandVisualizer {
    constructor(scene) {
        this.group = new THREE.Group();
        this.joints = [];
        this.bones = [];
        this.auxElements = []; // For rings/auras
        scene.add(this.group);

        const jointGeometry = new THREE.IcosahedronGeometry(0.8, 0); // Slightly smaller
        const auraGeometry = new THREE.IcosahedronGeometry(1.6, 1);

        const jointMaterial = new THREE.MeshBasicMaterial({ color: 0x00ffff, wireframe: true, transparent: true, opacity: 0.8 });
        const auraMaterial = new THREE.MeshBasicMaterial({ color: 0x00ffff, wireframe: true, transparent: true, opacity: 0.15 });
        const tipMaterial = new THREE.MeshBasicMaterial({ color: 0xff0055 });
        const boneMaterial = new THREE.MeshBasicMaterial({ color: 0x00aaff, transparent: true, opacity: 0.5 });

        // 1. Create Joints and Aura Shells
        for (let i = 0; i < 21; i++) {
            const jointGroup = new THREE.Group();
            this.group.add(jointGroup);

            let mainNode = new THREE.Mesh(
                [4, 8, 12, 16, 20].includes(i) ? new THREE.SphereGeometry(0.6, 16, 16) : jointGeometry,
                [4, 8, 12, 16, 20].includes(i) ? tipMaterial : jointMaterial
            );
            jointGroup.add(mainNode);

            // Add aura shell for main joints
            if (![4, 8, 12, 16, 20].includes(i)) {
                let aura = new THREE.Mesh(auraGeometry, auraMaterial);
                jointGroup.add(aura);
                this.auxElements.push({ mesh: aura, speed: 0.01 });
            }

            // Special Holographic Rings for wrist (0) and finger bases (5, 9, 13, 17)
            if ([0, 5, 9, 13, 17].includes(i)) {
                const ringGeo = new THREE.TorusGeometry(1.8, 0.04, 8, 32); // Scaled down
                const ringMat = new THREE.MeshBasicMaterial({ color: 0x00ffff, transparent: true, opacity: 0.4 });
                const ring = new THREE.Mesh(ringGeo, ringMat);
                ring.rotation.x = Math.PI / 2;
                jointGroup.add(ring);
                this.auxElements.push({ mesh: ring, speed: 0.03, axis: 'z' });
            }

            this.joints.push(jointGroup);
        }

        // 2. Create Substantial Bones (Cylinders)
        for (let i = 0; i < HAND_CONNECTIONS.length; i++) {
            const cylinder = new THREE.Mesh(new THREE.CylinderGeometry(0.12, 0.12, 1, 8), boneMaterial); // Thinner tubes
            this.group.add(cylinder);
            this.bones.push(cylinder);
        }
    }

    update(landmarks, cam, w, h, xOffset, yOffset, displayWidth, displayHeight) {
        this.group.visible = true;
        const mapped = landmarks.map(lm => {
            // Normalize to screen space considering the cropped display
            const screenX = ((1 - lm.x) * displayWidth + xOffset) / w;
            const screenY = (lm.y * displayHeight + yOffset) / h;
            // Map to the EXACT plane where the video is (GLOBAL_Z_DEPTH)
            return screenToWorld(screenX, screenY, GLOBAL_Z_DEPTH, cam, w, h);
        });

        // Update Joint Positions and Animate Aux
        for (let i = 0; i < 21; i++) {
            this.joints[i].position.copy(mapped[i]);
            // Subtle pulse
            const s = 1 + Math.sin(Date.now() * 0.005) * 0.05;
            this.joints[i].scale.set(s, s, s);
        }

        // Animate Rings and Auras
        this.auxElements.forEach(el => {
            if (el.axis === 'z') el.mesh.rotation.z += el.speed;
            else {
                el.mesh.rotation.x += el.speed;
                el.mesh.rotation.y += el.speed;
            }
        });

        // Update Bones (Position, Rotation, Scale)
        for (let i = 0; i < HAND_CONNECTIONS.length; i++) {
            const [sIdx, eIdx] = HAND_CONNECTIONS[i];
            const start = mapped[sIdx];
            const end = mapped[eIdx];
            const bone = this.bones[i];

            // Position at midpoint
            bone.position.copy(start).lerp(end, 0.5);

            // Align with direction
            const direction = new THREE.Vector3().subVectors(end, start);

            bone.scale.set(1, direction.length(), 1);
            bone.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), direction.clone().normalize());
        }
    }

    hide() { this.group.visible = false; }
}

class PhysicsCube {
    constructor(scene, size = 12) { // Slightly bigger for better fluid feel
        this.size = size;
        this.group = new THREE.Group();
        scene.add(this.group);

        this.cubeGroup = new THREE.Group(); // Sub-group specifically for the box + lines for easy scaling
        this.group.add(this.cubeGroup);

        const cubeGeo = new THREE.BoxGeometry(1, 1, 1); // Unit cube, scaled via cubeGroup
        const cubeMat = new THREE.MeshBasicMaterial({ color: 0x00ffff, wireframe: true, transparent: true, opacity: 0.2 });
        this.cubeGroup.add(new THREE.Mesh(cubeGeo, cubeMat));
        this.cubeGroup.add(new THREE.LineSegments(new THREE.EdgesGeometry(cubeGeo), new THREE.LineBasicMaterial({ color: 0x00ffff, transparent: true, opacity: 0.5 })));

        this.cubeGroup.scale.set(size, size, size);

        this.particleCount = 50; // Increased for fluid look
        this.particles = [];
        for (let i = 0; i < this.particleCount; i++) {
            const mesh = new THREE.Mesh(new THREE.SphereGeometry(0.4, 8, 8), new THREE.MeshBasicMaterial({ color: 0xff0055 }));
            mesh.position.set((Math.random() - 0.5) * (size - 1), (Math.random() - 0.5) * (size - 1), (Math.random() - 0.5) * (size - 1));
            this.group.add(mesh);
            this.particles.push({ mesh, velocity: new THREE.Vector3() });
        }
    }

    setSize(newSize) {
        this.size = Math.max(5, Math.min(newSize, 40)); // Clamped size
        this.cubeGroup.scale.set(this.size, this.size, this.size);
    }

    update() {
        // Disabled auto-rotation as requested
        const b = this.size / 2 - 0.4;

        // 2. Global gravity projected into local space
        const globalGravity = new THREE.Vector3(0, -0.008, 0);
        const localGravity = globalGravity.clone().applyQuaternion(this.group.quaternion.clone().invert());

        const cohesionRange = 2.0;
        const cohesionStrength = 0.0001;
        const separationRange = 1.4;
        const separationStrength = 0.01;
        const particleRadius = 0.4;
        const collisionDist = particleRadius * 2;

        for (let i = 0; i < this.particles.length; i++) {
            const p = this.particles[i];
            p.velocity.add(localGravity);

            // 3. Fluid & Separation Interactions
            for (let j = i + 1; j < this.particles.length; j++) {
                const p2 = this.particles[j];
                const diff = new THREE.Vector3().subVectors(p2.mesh.position, p.mesh.position);
                const dist = diff.length();
                if (dist < 0.001) continue;

                // Hard Collision Resolve
                if (dist < collisionDist) {
                    const overlap = collisionDist - dist;
                    const resolve = diff.clone().normalize().multiplyScalar(overlap * 0.5);
                    p.mesh.position.sub(resolve);
                    p2.mesh.position.add(resolve);
                    const vDiff = new THREE.Vector3().subVectors(p2.velocity, p.velocity);
                    const dot = vDiff.dot(diff.clone().normalize());
                    if (dot < 0) {
                        const impulse = diff.clone().normalize().multiplyScalar(dot * 0.8);
                        p.velocity.add(impulse);
                        p2.velocity.sub(impulse);
                    }
                }
                if (dist < cohesionRange) {
                    const force = diff.clone().normalize().multiplyScalar(cohesionStrength);
                    p.velocity.add(force);
                    p2.velocity.sub(force);
                }
                if (dist < separationRange) {
                    const force = diff.clone().normalize().multiplyScalar(separationStrength * (1 - dist / separationRange));
                    p.velocity.sub(force);
                    p2.velocity.add(force);
                }
            }

            p.velocity.multiplyScalar(0.97);
            p.mesh.position.add(p.velocity);

            if (p.mesh.position.x > b) { p.mesh.position.x = b; p.velocity.x *= -0.5; }
            if (p.mesh.position.x < -b) { p.mesh.position.x = -b; p.velocity.x *= -0.5; }
            if (p.mesh.position.y > b) { p.mesh.position.y = b; p.velocity.y *= -0.5; }
            if (p.mesh.position.y < -b) { p.mesh.position.y = -b; p.velocity.y *= -0.5; }
            if (p.mesh.position.z > b) { p.mesh.position.z = b; p.velocity.z *= -0.5; }
            if (p.mesh.position.z < -b) { p.mesh.position.z = -b; p.velocity.z *= -0.5; }
        }
    }

    updatePosition(cam, w, h) {
        const pos = screenToWorld(0.2, 0.2, GLOBAL_Z_DEPTH, cam, w, h);
        this.group.position.copy(pos);
    }
}

function initThree() {
    scene = new THREE.Scene();
    camera = new THREE.PerspectiveCamera(45, window.innerWidth / window.innerHeight, 1, 2000);
    camera.position.z = 100; // Camera position

    renderer = new THREE.WebGLRenderer({ canvas: canvasElement, alpha: true, antialias: true });
    renderer.setSize(window.innerWidth, window.innerHeight);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2)); // Limit pixel ratio for performance

    const videoTexture = new THREE.VideoTexture(video);
    videoTexture.colorSpace = THREE.SRGBColorSpace;

    backgroundPlane = new THREE.Mesh(
        new THREE.PlaneGeometry(1, 1),
        new THREE.MeshBasicMaterial({
            map: videoTexture,
            depthTest: false, // Ensure video doesn't obscure 3D via depth buffer
            depthWrite: false
        })
    );
    backgroundPlane.renderOrder = -1; // Draw first, before anything else
    scene.add(backgroundPlane);

    composer = new EffectComposer(renderer);
    composer.addPass(new RenderPass(scene, camera));
    composer.addPass(new UnrealBloomPass(new THREE.Vector2(window.innerWidth, window.innerHeight), 1.5, 0.4, 0.85));
    composer.addPass(new OutputPass());

    handVisualizers = [new HandVisualizer(scene), new HandVisualizer(scene)];

    physicsCube = new PhysicsCube(scene);

    window.addEventListener('resize', onWindowResize);
    onWindowResize(); // Call once to set initial sizes and positions
}

function onWindowResize() {
    const w = window.innerWidth;
    const h = window.innerHeight;
    camera.aspect = w / h;
    camera.updateProjectionMatrix();
    renderer.setSize(w, h);
    composer.setSize(w, h);

    // Scale background plane to fill the view at a specific distance
    const distance = 80; // Distance from camera to the plane
    const vFOV = THREE.MathUtils.degToRad(camera.fov); // Vertical FOV in radians
    const planeHeight = 2 * Math.tan(vFOV / 2) * distance;
    const planeWidth = planeHeight * camera.aspect;

    backgroundPlane.position.z = camera.position.z - distance;
    backgroundPlane.scale.set(planeWidth, planeHeight, 1);
    backgroundPlane.scale.x *= -1; // Mirror the video horizontally

    if (physicsCube) physicsCube.updatePosition(camera, w, h);
}

const loadLandmarker = async () => {
    const vision = await FilesetResolver.forVisionTasks("./node_modules/@mediapipe/tasks-vision/wasm");
    handLandmarker = await HandLandmarker.createFromOptions(vision, {
        baseOptions: { modelAssetPath: "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task", delegate: "GPU" },
        runningMode: "VIDEO", numHands: 2
    });
    loading.style.display = "none";
    navigator.mediaDevices.getUserMedia({ video: true }).then(stream => {
        video.srcObject = stream;
        video.addEventListener("loadeddata", () => { onWindowResize(); requestAnimationFrame(renderLoop); });
    });
};

// Interaction state
let lastPinchDist = -1;
let lastPinchPos = null;

function detectPinch(landmarks) {
    if (!landmarks) return null;
    const thumbTip = landmarks[4];
    const indexTip = landmarks[8];
    const dist = Math.sqrt(Math.pow(thumbTip.x - indexTip.x, 2) + Math.pow(thumbTip.y - indexTip.y, 2));
    // Check if thumb and index tips are close (normalized units)
    return dist < 0.05 ? { x: (thumbTip.x + indexTip.x) / 2, y: (thumbTip.y + indexTip.y) / 2 } : null;
}

let lastTime = -1;
function renderLoop() {
    if (video.currentTime !== lastTime) {
        lastTime = video.currentTime;
        const results = handLandmarker.detectForVideo(video, performance.now());

        const w = window.innerWidth;
        const h = window.innerHeight;
        const vR = video.videoWidth / video.videoHeight;
        const wR = w / h;

        let dW, dH;
        if (wR > vR) { dW = w; dH = w / vR; }
        else { dH = h; dW = h * vR; }

        const xO = (w - dW) / 2;
        const yO = (h - dH) / 2;

        const pinches = [];
        if (results.landmarks) {
            for (let i = 0; i < 2; i++) {
                if (results.landmarks[i]) {
                    handVisualizers[i].update(results.landmarks[i], camera, w, h, xO, yO, dW, dH);
                    const p = detectPinch(results.landmarks[i]);
                    if (p) pinches.push(p);
                } else {
                    handVisualizers[i].hide();
                }
            }
        }

        // Interaction Logic
        if (pinches.length === 2) {
            // Two-hand Scaling: Expand/Shrink
            const dist = Math.sqrt(Math.pow(pinches[0].x - pinches[1].x, 2) + Math.pow(pinches[0].y - pinches[1].y, 2));
            if (lastPinchDist > 0) {
                const delta = dist / lastPinchDist;
                physicsCube.setSize(physicsCube.size * delta);
            }
            lastPinchDist = dist;
            lastPinchPos = null;
        } else if (pinches.length === 1) {
            // One-hand Rotation
            if (lastPinchPos) {
                const dx = pinches[0].x - lastPinchPos.x;
                const dy = pinches[0].y - lastPinchPos.y;
                physicsCube.group.rotation.y += dx * 5;
                physicsCube.group.rotation.x += dy * 5;
            }
            lastPinchPos = pinches[0];
            lastPinchDist = -1;
        } else {
            lastPinchDist = -1;
            lastPinchPos = null;
        }
    }

    if (physicsCube) physicsCube.update();

    composer.render();
    requestAnimationFrame(renderLoop);
}

initThree();
loadLandmarker();
