import { createBridge } from './webgpu.js';

const CANVAS_WIDTH = 800;
const CANVAS_HEIGHT = 600;

async function main() {
    // Check WebGPU support
    if (!navigator.gpu) {
        document.body.innerHTML = '<h2>WebGPU is not supported in this browser.</h2><p>Try Chrome 113+ or Firefox 121+.</p>';
        return;
    }

    // Request adapter and device
    const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
    if (!adapter) {
        document.body.innerHTML = '<h2>No WebGPU adapter found.</h2>';
        return;
    }
    const device = await adapter.requestDevice();
    const queue = device.queue;

    // Set up canvas
    const canvas = document.getElementById('webgpu-canvas');
    canvas.width = CANVAS_WIDTH;
    canvas.height = CANVAS_HEIGHT;
    const context = canvas.getContext('webgpu');
    const canvasFormat = navigator.gpu.getPreferredCanvasFormat();

    // Create bridge and pre-initialize with WebGPU objects
    const bridge = createBridge();
    bridge.setPreInitialized(adapter, device, queue, context, canvasFormat);

    // Map canvas format string to C API integer
    const FORMAT_MAP = {
        'rgba8unorm': 18, 'rgba8unorm-srgb': 19,
        'bgra8unorm': 23, 'bgra8unorm-srgb': 24,
    };
    const formatInt = FORMAT_MAP[canvasFormat] || 23;

    // Fetch GLB model
    const glbResponse = await fetch('hornet.glb');
    if (!glbResponse.ok) {
        document.body.innerHTML = '<h2>Failed to load model.glb</h2><p>Place a GLB file as web/model.glb</p>';
        return;
    }
    const glbBytes = new Uint8Array(await glbResponse.arrayBuffer());

    // Load WASM module
    const response = await fetch('example.wasm');
    // Add debug helper to bridge
    bridge.env.console_log_int = (val) => {
        if (val === -1) console.error('WASM PANIC at line:');
        else console.log('WASM debug:', val);
    };

    const { instance } = await WebAssembly.instantiateStreaming(response, { env: bridge.env });

    // Give bridge access to WASM memory
    bridge.setMemory(instance.exports.memory);

    // Call _initialize if present (C3 runtime init for globals/statics)
    if (instance.exports._initialize) {
        instance.exports._initialize();
    }

    // Allocate WASM memory and copy GLB data
    const glbPtr = instance.exports.alloc(glbBytes.length);
    new Uint8Array(instance.exports.memory.buffer).set(glbBytes, glbPtr);

    // Initialize the renderer with GLB data
    instance.exports.init(CANVAS_WIDTH, CANVAS_HEIGHT, formatInt, glbPtr, glbBytes.length);

    // Render loop
    function frame() {
        instance.exports.render_frame();
        requestAnimationFrame(frame);
    }
    requestAnimationFrame(frame);

    console.log('WebGPU glTF viewer running!');
}

main().catch(err => {
    console.error('Fatal error:', err);
    document.body.innerHTML = `<h2>Error</h2><pre>${err.message}\n${err.stack}</pre>`;
});
