const CANVAS_WIDTH = 800;
const CANVAS_HEIGHT = 600;

// Extract the JS bridge embedded in the WASM binary via $embed.
// Instantiate once with no-op stubs (just to read static data), then
// evaluate the JS to get the real createBridge function.
async function loadBridge(wasmBytes) {
    const module = await WebAssembly.compile(wasmBytes);
    const stubs = {};
    for (const imp of WebAssembly.Module.imports(module)) {
        if (!stubs[imp.module]) stubs[imp.module] = {};
        if (imp.kind === 'function') stubs[imp.module][imp.name] = () => 0;
    }
    const { exports } = await WebAssembly.instantiate(module, stubs);
    const ptr = exports.get_bridge_js();
    const len = exports.get_bridge_js_len();
    const js = new TextDecoder().decode(new Uint8Array(exports.memory.buffer, ptr, len));
    return new Function(js.replace('export function', 'function') + '\nreturn createBridge;')();
}

// Parse GLB and decode embedded images to RGBA pixel arrays
async function decodeGLBImages(glbBytes) {
    const view = new DataView(glbBytes.buffer, glbBytes.byteOffset, glbBytes.byteLength);
    const jsonChunkLength = view.getUint32(12, true);
    const jsonStr = new TextDecoder().decode(glbBytes.slice(20, 20 + jsonChunkLength));
    const json = JSON.parse(jsonStr);

    const binStart = 20 + jsonChunkLength + 8;
    const binChunkLength = view.getUint32(20 + jsonChunkLength, true);
    const binData = glbBytes.slice(binStart, binStart + binChunkLength);

    const decoded = [];
    if (!json.images) return decoded;

    for (const image of json.images) {
        if (image.bufferView === undefined) { decoded.push(null); continue; }
        const bv = json.bufferViews[image.bufferView];
        const offset = bv.byteOffset || 0;
        const blob = new Blob([binData.slice(offset, offset + bv.byteLength)], {
            type: image.mimeType || 'image/png',
        });
        const bitmap = await createImageBitmap(blob);
        const canvas = new OffscreenCanvas(bitmap.width, bitmap.height);
        const ctx = canvas.getContext('2d');
        ctx.drawImage(bitmap, 0, 0);
        const imageData = ctx.getImageData(0, 0, bitmap.width, bitmap.height);
        decoded.push({
            width: bitmap.width,
            height: bitmap.height,
            data: new Uint8Array(imageData.data.buffer),
        });
        bitmap.close();
    }
    return decoded;
}

async function main() {
    if (!navigator.gpu) {
        document.body.innerHTML = '<h2>WebGPU is not supported in this browser.</h2><p>Try Chrome 113+ or Firefox 121+.</p>';
        return;
    }

    const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
    if (!adapter) {
        document.body.innerHTML = '<h2>No WebGPU adapter found.</h2>';
        return;
    }
    const device = await adapter.requestDevice();
    const queue = device.queue;

    const canvas = document.getElementById('webgpu-canvas');
    canvas.width = CANVAS_WIDTH;
    canvas.height = CANVAS_HEIGHT;
    const context = canvas.getContext('webgpu');
    const canvasFormat = navigator.gpu.getPreferredCanvasFormat();

    // Fetch WASM and GLB in parallel
    const [wasmBytes, glbResponse] = await Promise.all([
        fetch('example.wasm').then(r => r.arrayBuffer()),
        fetch('hornet.glb'),
    ]);
    if (!glbResponse.ok) {
        document.body.innerHTML = '<h2>Failed to load hornet.glb</h2>';
        return;
    }
    const glbBytes = new Uint8Array(await glbResponse.arrayBuffer());

    // Extract JS bridge from the WASM binary, then create the real bridge
    const createBridge = await loadBridge(wasmBytes);
    const bridge = createBridge();
    bridge.setPreInitialized(adapter, device, queue, context, canvasFormat);

    // Decode embedded GLB images before WASM init
    const decodedImages = await decodeGLBImages(glbBytes);
    bridge.setDecodedImages(decodedImages);

    const FORMAT_MAP = {
        'rgba8unorm': 22, 'rgba8unorm-srgb': 23,
        'bgra8unorm': 27, 'bgra8unorm-srgb': 28,
    };
    const formatInt = FORMAT_MAP[canvasFormat] || 27;

    // Instantiate for real with the bridge imports
    const module = await WebAssembly.compile(wasmBytes);
    const instance = await WebAssembly.instantiate(module, { env: bridge.env });
    bridge.setMemory(instance.exports.memory);

    if (instance.exports._initialize) {
        instance.exports._initialize();
    }

    const glbPtr = instance.exports.alloc(glbBytes.length);
    new Uint8Array(instance.exports.memory.buffer).set(glbBytes, glbPtr);
    instance.exports.init(CANVAS_WIDTH, CANVAS_HEIGHT, formatInt, glbPtr, glbBytes.length);

    function frame() {
        instance.exports.render_frame();
        requestAnimationFrame(frame);
    }
    requestAnimationFrame(frame);
}

main().catch(err => {
    console.error('Fatal error:', err);
    document.body.innerHTML = `<h2>Error</h2><pre>${err.message}\n${err.stack}</pre>`;
});
