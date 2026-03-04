# wgpu-gltf

Running example can be seen [here](https://tonis2.github.io/wgpu-gltf/web/) 

C3 WebGPU bindings + glTF loader targeting **wasm32**, running in the browser. The `wgpu` library provides C3 bindings to the WebGPU API via an embedded JS bridge, and the `gltf` library parses glTF/GLB files. Together they let you build GPU-rendered 3D apps in C3 that compile to a single `.wasm` file.

The included example loads a `.glb` model with textures and renders it with a simple Lambertian shader.

## Build and run

Requires [C3](https://c3-lang.org/) compiler.


First clone the project recursively, to get it with dependencies

```bash
git clone https://github.com/tonis2/wgpu-gltf --recursive
```

```bash
c3c build example
python3 -m http.server 8080 --directory web
# Open http://localhost:8080
```

Place a `.glb` file as `web/model.glb` for the default model, or use the file upload button to load any GLB at runtime.


## Known issues

**wasm32 TempAllocator bug**: C3's default temp allocator (`tmem`) crashes with `unreachable` on the second WASM export call. Workaround: pre-allocate an `ArenaAllocator` buffer during `init()` and use `mem::@scoped(&arena)` for allocation-heavy code paths. See `src/main.c3` `load_model()` for an example.
