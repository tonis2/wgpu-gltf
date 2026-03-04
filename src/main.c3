module example;

import std::io;
import std::math;
import std::core::mem;
import std::core::mem::allocator;
import std::collections::object;
import std::core::dstring;
import wgpu;
import gltf;

// JS image bridge (decoded images provided by main.js before WASM init)
extern fn uint js_get_image_count() @cname("js_get_image_count");
extern fn void js_get_image_info(uint index, uint* out_ptr) @cname("js_get_image_info");
extern fn void js_copy_image_to_texture(uint index, wgpu::Texture texture) @cname("js_copy_image_to_texture");

// Provide missing compiler-rt intrinsic for wasm32 (128-bit overflow multiply)
fn int128 __muloti4(int128 a, int128 b, int* overflow) @export("__muloti4") {
    *overflow = 0;
    int128 result = a * b;
    if (a != 0 && b != 0) {
        if (a > 0 && b > 0 && result < 0) *overflow = 1;
        if (a < 0 && b < 0 && result < 0) *overflow = 1;
        if (a > 0 && b < 0 && result > 0) *overflow = 1;
        if (a < 0 && b > 0 && result > 0) *overflow = 1;
    }
    return result;
}

// ----------------------------------------------------------------
// WGSL Shader
// ----------------------------------------------------------------
const char* WGSL_SOURCE = `
struct Uniforms {
    mvp:   mat4x4<f32>,
    model: mat4x4<f32>,
}
@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(0) @binding(1) var base_texture: texture_2d<f32>;
@group(0) @binding(2) var base_sampler: sampler;

struct VertexIn {
    @location(0) pos:    vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) uv:     vec2<f32>,
}
struct VertexOut {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0)       world_normal: vec3<f32>,
    @location(1)       uv: vec2<f32>,
}

@vertex
fn vs_main(in: VertexIn) -> VertexOut {
    var out: VertexOut;
    out.clip_pos = uniforms.mvp * vec4<f32>(in.pos, 1.0);
    out.world_normal = (uniforms.model * vec4<f32>(in.normal, 0.0)).xyz;
    out.uv = in.uv;
    return out;
}

@fragment
fn fs_main(in: VertexOut) -> @location(0) vec4<f32> {
    let light_dir = normalize(vec3<f32>(0.5, 1.0, 0.3));
    let n = normalize(in.world_normal);
    let ndotl = max(dot(n, light_dir), 0.0);
    let ambient = 0.15;
    let diffuse = ndotl * 0.85;
    let tex_color = textureSample(base_texture, base_sampler, in.uv);
    let color = tex_color.rgb * (ambient + diffuse);
    return vec4<f32>(color, tex_color.a);
}
`;

// ----------------------------------------------------------------
// Interleaved vertex: position + normal
// ----------------------------------------------------------------
alias Vec3f = float[<3>];
alias Vec2f = float[<2>];

struct Vertex {
    Vec3f position;
    Vec3f normal;
    Vec2f texcoord;
}

alias Mat4 = math::Matrix4f;

// ----------------------------------------------------------------
// Global state
// ----------------------------------------------------------------
struct State {
    // Canvas-level (created once)
    Device       device;
    Queue        queue;
    Surface      surface;
    wgpu::Texture  depth_texture;
    TextureView  depth_view;
    TextureFormat surf_format;
    // Model-level (recreated per model load)
    GpuBuffer    vb;
    GpuBuffer    ib;
    GpuBuffer    ub;
    BindGroup    bind_group;
    RenderPipeline pipeline;
    wgpu::Texture  color_texture;
    TextureView  color_view;
    wgpu::Sampler  color_sampler;
    Mat4         proj;
    Mat4         view;
    float        angle;
    uint         index_count;
    IndexFormat  index_format;
    bool         model_loaded;
}

State g;

// Pre-allocated arena for glTF parsing. Uses a fixed buffer so it never
// needs to expand via the heap (which crashes on wasm32 on second call).
const usz PARSE_ARENA_SIZE = 16 * 1024 * 1024; // 16 MB
char* g_arena_buf;
ArenaAllocator g_arena;

// ----------------------------------------------------------------
// Exported functions
// ----------------------------------------------------------------

fn char* alloc(uint size) @export("alloc") {
    return (char*)mem::malloc(size);
}

fn void dealloc(char* ptr) @export("dealloc") {
    mem::free(ptr);
}

fn void init(
    uint canvas_width, uint canvas_height, uint surf_format_uint
) @export("init") {
    TextureFormat surf_format = (TextureFormat)surf_format_uint;

    // Allocate parse buffer early (heap works on first export call)
    g_arena_buf = (char*)mem::malloc(PARSE_ARENA_SIZE);
    g_arena.init(g_arena_buf[0..PARSE_ARENA_SIZE]);

    init_renderer(canvas_width, canvas_height, surf_format);
}

fn void load_model(char* glb_data, uint glb_size) @export("load_model") {
    cleanup_model();
    g_arena.clear();

    // Use @scoped so ALL allocations inside gltf::parse (Lists, HashMaps, etc.)
    // use the pre-allocated arena instead of the broken wasm32 heap.
    gltf::Gltf model;
    mem::@scoped(&g_arena) {
        io::ByteReader reader;
        reader.init(glb_data[0..glb_size]);
        model = gltf::parse(&reader, &g_arena)!!;
    };
    init_model(model);
}

fn void cleanup_model() {
    if (!g.model_loaded) return;
    g.model_loaded = false;
    g.vb.destroy();
    g.ib.destroy();
    g.ub.destroy();
    wgpu::textureDestroy(g.color_texture);
    wgpu::textureViewRelease(g.color_view);
    wgpu::samplerRelease(g.color_sampler);
    wgpu::bindGroupRelease(g.bind_group);
    wgpu::renderPipelineRelease(g.pipeline);
}

fn void init_renderer(
    uint canvas_width, uint canvas_height, TextureFormat surf_format
) {
    // --- WebGPU setup (once per session) ---

    // Instance
    InstanceDescriptor inst_desc;
    Instance instance = wgpu::createInstance(&inst_desc);

    // Adapter
    Adapter adapter;
    RequestAdapterOptions adapter_opts = {
        .power_preference = wgpu::POWER_PREFERENCE_HIGH_PERFORMANCE,
        .feature_level    = wgpu::FEATURE_LEVEL_CORE,
    };
    RequestAdapterCallbackInfo adapter_cb = {
        .mode      = wgpu::CALLBACK_MODE_ALLOW_SPONTANEOUS,
        .callback  = &on_adapter,
        .userdata1 = &adapter,
    };
    wgpu::instanceRequestAdapter(instance, &adapter_opts, adapter_cb);

    // Device
    Device device;
    DeviceDescriptor dev_desc;
    RequestDeviceCallbackInfo device_cb = {
        .mode      = wgpu::CALLBACK_MODE_ALLOW_SPONTANEOUS,
        .callback  = &on_device,
        .userdata1 = &device,
    };
    wgpu::adapterRequestDevice(adapter, &dev_desc, device_cb);

    Queue queue = wgpu::deviceGetQueue(device);

    // Surface
    Surface surface;
    SurfaceDescriptor surf_desc = {
        .label = wgpu::sv_empty(),
    };
    surface = wgpu::instanceCreateSurface(instance, &surf_desc);

    SurfaceConfiguration surf_config = {
        .device       = device,
        .format       = surf_format,
        .usage        = wgpu::TEXTURE_USAGE_RENDER_ATTACHMENT,
        .width        = canvas_width,
        .height       = canvas_height,
        .alpha_mode   = wgpu::COMPOSITE_ALPHA_MODE_AUTO,
        .present_mode = wgpu::PRESENT_MODE_FIFO,
    };
    wgpu::surfaceConfigure(surface, &surf_config);

    // Depth texture
    TextureDescriptor depth_tex_desc = {
        .usage     = wgpu::TEXTURE_USAGE_RENDER_ATTACHMENT,
        .dimension = wgpu::TEXTURE_DIMENSION_2D,
        .size      = { .width = canvas_width, .height = canvas_height, .depth_or_array_layers = 1 },
        .format    = wgpu::TEXTURE_FORMAT_DEPTH24_PLUS,
        .mip_level_count = 1,
        .sample_count    = 1,
    };
    wgpu::Texture depth_texture = wgpu::deviceCreateTexture(device, &depth_tex_desc);
    TextureView depth_view = wgpu::textureCreateView(depth_texture, null);

    // Projection matrix
    float aspect = (float)canvas_width / (float)canvas_height;
    Mat4 proj = math::matrix4f_perspective((float)math::PI / 4.0f, aspect, 0.01f, 1000.0f);

    // Store canvas-level state
    g.device        = device;
    g.queue         = queue;
    g.surface       = surface;
    g.depth_texture = depth_texture;
    g.depth_view    = depth_view;
    g.surf_format   = surf_format;
    g.proj          = proj;
}

fn void init_model(gltf::Gltf model) {
    // --- Extract vertex data from first mesh ---
    gltf::Mesh mesh = model.meshes[0];
    gltf::Primitive primitive = mesh.primitives[0];

    gltf::Accessor pos_accessor = model.accessors[primitive.attributes["POSITION"]!!];
    gltf::Accessor index_accessor = model.accessors[primitive.attributes["indices"]!!];

    // Normal accessor (optional)
    usz? normal_attr = primitive.attributes["NORMAL"];
    bool has_normals = @ok(normal_attr);
    gltf::Accessor normal_accessor;
    if (has_normals) {
        normal_accessor = model.accessors[normal_attr!!];
    }

    // Texcoord accessor (optional)
    usz? texcoord_attr = primitive.attributes["TEXCOORD_0"];
    bool has_texcoords = @ok(texcoord_attr);
    gltf::Accessor texcoord_accessor;
    if (has_texcoords) {
        texcoord_accessor = model.accessors[texcoord_attr!!];
    }

    // Build interleaved vertex array
    uint vertex_count = pos_accessor.count;
    Vertex[] vertices = mem::new_array(Vertex, vertex_count);
    for (uint i = 0; i < vertex_count; i++) {
        vertices[i].position = model.@cast_buffer(pos_accessor, i, Vec3f);
        if (has_normals) {
            vertices[i].normal = model.@cast_buffer(normal_accessor, i, Vec3f);
        } else {
            vertices[i].normal = { 0.0, 1.0, 0.0 };
        }
        if (has_texcoords) {
            vertices[i].texcoord = model.@cast_buffer(texcoord_accessor, i, Vec2f);
        } else {
            vertices[i].texcoord = { 0.0, 0.0 };
        }
    }

    // Extract index data
    gltf::BufferView index_view = model.bufferViews[index_accessor.view];
    gltf::Buffer index_buffer_data = model.buffers[index_view.buffer];
    usz idx_offset = index_accessor.offset + index_view.offset;
    usz idx_byte_size = index_view.byteLength;
    char* idx_raw = &index_buffer_data.data[idx_offset];

    IndexFormat idx_format;
    if (index_accessor.data_type == gltf::DataType.USHORT) {
        idx_format = wgpu::INDEX_FORMAT_UINT16;
    } else {
        idx_format = wgpu::INDEX_FORMAT_UINT32;
    }
    uint index_count = index_accessor.count;

    // Bounding box for camera placement
    Vec3f bb_min = pos_accessor.min.xyz;
    Vec3f bb_max = pos_accessor.max.xyz;
    Vec3f center = (bb_min + bb_max) * 0.5f;
    Vec3f extent = bb_max - bb_min;
    float max_extent = extent.x;
    if (extent.y > max_extent) max_extent = extent.y;
    if (extent.z > max_extent) max_extent = extent.z;
    float cam_dist = max_extent * 1.5f;

    // Vertex buffer
    usz vb_size = vertex_count * Vertex.sizeof;
    GpuBuffer vb = wgpu::gpu_buffer(g.device, wgpu::BUFFER_USAGE_VERTEX | wgpu::BUFFER_USAGE_COPY_DST, vb_size);
    vb.upload(g.queue, vertices.ptr, vb_size);

    // Index buffer
    GpuBuffer ib = wgpu::gpu_buffer(g.device, wgpu::BUFFER_USAGE_INDEX | wgpu::BUFFER_USAGE_COPY_DST, idx_byte_size);
    ib.upload(g.queue, idx_raw, idx_byte_size);

    // Uniform buffer (128 bytes = MVP mat4 + model mat4)
    GpuBuffer ub = wgpu::gpu_buffer(g.device, wgpu::BUFFER_USAGE_UNIFORM | wgpu::BUFFER_USAGE_COPY_DST, 128);

    // --- Base color texture ---
    int image_index = -1;
    usz? mat_attr = primitive.attributes["material"];
    if (@ok(mat_attr)) {
        gltf::Material mat = model.materials[mat_attr!!];
        if (mat.baseColorTexture.source > -1) {
            image_index = mat.baseColorTexture.source;
        }
    }

    wgpu::Texture color_texture;
    TextureView color_view;
    uint image_count = js_get_image_count();
    if (image_index >= 0 && (uint)image_index < image_count) {
        uint[2] img_info;
        js_get_image_info((uint)image_index, &img_info);
        uint img_w = img_info[0];
        uint img_h = img_info[1];
        TextureDescriptor color_tex_desc = {
            .usage     = wgpu::TEXTURE_USAGE_TEXTURE_BINDING | wgpu::TEXTURE_USAGE_COPY_DST,
            .dimension = wgpu::TEXTURE_DIMENSION_2D,
            .size      = { .width = img_w, .height = img_h, .depth_or_array_layers = 1 },
            .format    = wgpu::TEXTURE_FORMAT_RGBA8_UNORM,
            .mip_level_count = 1,
            .sample_count    = 1,
        };
        color_texture = wgpu::deviceCreateTexture(g.device, &color_tex_desc);
        js_copy_image_to_texture((uint)image_index, color_texture);
        color_view = wgpu::textureCreateView(color_texture, null);
    } else {
        TextureDescriptor white_tex_desc = {
            .usage     = wgpu::TEXTURE_USAGE_TEXTURE_BINDING | wgpu::TEXTURE_USAGE_COPY_DST,
            .dimension = wgpu::TEXTURE_DIMENSION_2D,
            .size      = { .width = 1, .height = 1, .depth_or_array_layers = 1 },
            .format    = wgpu::TEXTURE_FORMAT_RGBA8_UNORM,
            .mip_level_count = 1,
            .sample_count    = 1,
        };
        color_texture = wgpu::deviceCreateTexture(g.device, &white_tex_desc);
        char[4] white_pixel = { '\xFF', '\xFF', '\xFF', '\xFF' };
        TexelCopyTextureInfo dest = { .texture = color_texture };
        TexelCopyBufferLayout layout = { .bytes_per_row = 4, .rows_per_image = 1 };
        Extent3D write_size = { .width = 1, .height = 1, .depth_or_array_layers = 1 };
        wgpu::queueWriteTexture(g.queue, &dest, &white_pixel, 4, &layout, &write_size);
        color_view = wgpu::textureCreateView(color_texture, null);
    }

    // Create sampler
    SamplerDescriptor samp_desc = {
        .address_mode_u = wgpu::ADDRESS_MODE_REPEAT,
        .address_mode_v = wgpu::ADDRESS_MODE_REPEAT,
        .mag_filter     = wgpu::FILTER_MODE_LINEAR,
        .min_filter     = wgpu::FILTER_MODE_LINEAR,
        .mipmap_filter  = wgpu::MIPMAP_FILTER_MODE_NEAREST,
    };
    wgpu::Sampler color_sampler = wgpu::deviceCreateSampler(g.device, &samp_desc);

    // Bind group layout (3 entries: uniform buffer, texture, sampler)
    BindGroupLayoutEntry[3] bgl_entries = {
        {
            .binding    = 0,
            .visibility = wgpu::SHADER_STAGE_VERTEX | wgpu::SHADER_STAGE_FRAGMENT,
            .buffer     = {
                .type              = wgpu::BUFFER_BINDING_TYPE_UNIFORM,
                .has_dynamic_offset = 0,
                .min_binding_size  = 128,
            },
        },
        {
            .binding    = 1,
            .visibility = wgpu::SHADER_STAGE_FRAGMENT,
            .texture    = {
                .sample_type    = wgpu::TEXTURE_SAMPLE_TYPE_FLOAT,
                .view_dimension = wgpu::TEXTURE_VIEW_DIMENSION_2D,
            },
        },
        {
            .binding    = 2,
            .visibility = wgpu::SHADER_STAGE_FRAGMENT,
            .sampler    = {
                .type = wgpu::SAMPLER_BINDING_TYPE_FILTERING,
            },
        },
    };
    BindGroupLayoutDescriptor bgl_desc = {
        .label         = wgpu::sv("BGL"),
        .entries_count = 3,
        .entries       = &bgl_entries,
    };
    BindGroupLayout bgl = wgpu::deviceCreateBindGroupLayout(g.device, &bgl_desc);

    // Pipeline layout
    PipelineLayoutDescriptor pl_desc = {
        .label                    = wgpu::sv("PL"),
        .bind_group_layouts_count = 1,
        .bind_group_layouts       = &bgl,
    };
    PipelineLayout pipeline_layout = wgpu::deviceCreatePipelineLayout(g.device, &pl_desc);

    // Bind group (3 entries)
    BindGroupEntry[3] bg_entries = {
        {
            .binding = 0,
            .buffer  = ub.handle,
            .offset  = 0,
            .size    = 128,
        },
        {
            .binding      = 1,
            .texture_view = color_view,
        },
        {
            .binding = 2,
            .sampler = color_sampler,
        },
    };
    BindGroupDescriptor bg_desc = {
        .label         = wgpu::sv("BG"),
        .layout        = bgl,
        .entries_count = 3,
        .entries       = &bg_entries,
    };
    BindGroup bind_group = wgpu::deviceCreateBindGroup(g.device, &bg_desc);

    // Shader module
    ShaderSourceWGSL wgsl_src = {
        .chain_next = null,
        .s_type     = wgpu::S_TYPE_SHADER_SOURCE_WGSL,
        .code       = wgpu::sv(WGSL_SOURCE),
    };
    ShaderModuleDescriptor sm_desc = {
        .next_in_chain = &wgsl_src,
        .label         = wgpu::sv("Shader"),
    };
    ShaderModule shader = wgpu::deviceCreateShaderModule(g.device, &sm_desc);

    // Vertex layout: stride=32, three attributes (pos, normal, texcoord)
    VertexAttribute[3] vert_attrs = {
        { .format = wgpu::VERTEX_FORMAT_FLOAT32X3, .offset = 0,  .shader_location = 0 },
        { .format = wgpu::VERTEX_FORMAT_FLOAT32X3, .offset = 12, .shader_location = 1 },
        { .format = wgpu::VERTEX_FORMAT_FLOAT32X2, .offset = 24, .shader_location = 2 },
    };
    VertexBufferLayout vbl = {
        .step_mode        = wgpu::VERTEX_STEP_MODE_VERTEX,
        .array_stride     = 32,
        .attributes_count = 3,
        .attributes       = &vert_attrs,
    };

    // Depth stencil state
    DepthStencilState ds_state = {
        .format              = wgpu::TEXTURE_FORMAT_DEPTH24_PLUS,
        .depth_write_enabled = wgpu::OPTIONAL_BOOL_TRUE,
        .depth_compare       = wgpu::COMPARE_FUNCTION_LESS,
    };

    // Render pipeline
    ColorTargetState color_target = {
        .format     = g.surf_format,
        .write_mask = wgpu::COLOR_WRITE_MASK_ALL,
    };
    FragmentState frag_state = {
        .shader_module = shader,
        .entry_point   = wgpu::sv("fs_main"),
        .targets_count = 1,
        .targets       = &color_target,
    };
    VertexState vert_state = {
        .shader_module = shader,
        .entry_point   = wgpu::sv("vs_main"),
        .buffers_count = 1,
        .buffers       = &vbl,
    };
    PrimitiveState prim_state = {
        .topology   = wgpu::PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,
        .front_face = wgpu::FRONT_FACE_CCW,
        .cull_mode  = wgpu::CULL_MODE_BACK,
    };
    MultisampleState ms_state = {
        .count = 1,
        .mask  = 0xFFFFFFFF,
    };
    RenderPipelineDescriptor rp_desc = {
        .label        = wgpu::sv("Pipeline"),
        .layout       = pipeline_layout,
        .vertex       = vert_state,
        .primitive    = prim_state,
        .multisample  = ms_state,
        .fragment     = &frag_state,
        .depth_stencil = &ds_state,
    };
    RenderPipeline pipeline = wgpu::deviceCreateRenderPipeline(g.device, &rp_desc);

    // View matrix from bounding box
    Mat4 view_mat = math::MATRIX4F_IDENTITY.translate({ -center.x, -center.y, -(center.z + cam_dist) });

    // Store model-level state
    g.vb            = vb;
    g.ib            = ib;
    g.ub            = ub;
    g.bind_group    = bind_group;
    g.pipeline      = pipeline;
    g.color_texture = color_texture;
    g.color_view    = color_view;
    g.color_sampler = color_sampler;
    g.view          = view_mat;
    g.angle         = 0.0f;
    g.index_count   = index_count;
    g.index_format  = idx_format;
    g.model_loaded  = true;
}

fn void render_frame() @export("render_frame") {
    if (!g.model_loaded) return;

    // Rotate model
    g.angle += 0.01f;
    Mat4 model = math::MATRIX4F_IDENTITY.rotate_y(g.angle);
    Mat4 mvp = g.proj * g.view * model;

    // Upload uniforms (transpose for WGSL column-major)
    Mat4 mvp_t = mvp.transpose();
    Mat4 model_t = model.transpose();
    g.ub.upload(g.queue, &mvp_t, 64, 0);
    g.ub.upload(g.queue, &model_t, 64, 64);

    // Get surface texture
    SurfaceTexture surf_tex;
    wgpu::surfaceGetCurrentTexture(g.surface, &surf_tex);
    TextureView frame_view = wgpu::textureCreateView(surf_tex.texture, null);

    // Encode render pass
    CommandEncoderDescriptor enc_desc;
    CommandEncoder encoder = wgpu::deviceCreateCommandEncoder(g.device, &enc_desc);

    RenderPassColorAttachment color_att = {
        .view        = frame_view,
        .depth_slice = wgpu::WGPU_DEPTH_SLICE_UNDEFINED,
        .load_op     = wgpu::LOAD_OP_CLEAR,
        .store_op    = wgpu::STORE_OP_STORE,
        .clear_value = { 0.1, 0.1, 0.15, 1.0 },
    };
    RenderPassDepthStencilAttachment depth_att = {
        .view              = g.depth_view,
        .depth_load_op     = wgpu::LOAD_OP_CLEAR,
        .depth_store_op    = wgpu::STORE_OP_STORE,
        .depth_clear_value = 1.0f,
    };
    RenderPassDescriptor rp_desc = {
        .label                      = wgpu::sv("Frame"),
        .color_attachments_count    = 1,
        .color_attachments          = &color_att,
        .depth_stencil_attachment   = &depth_att,
    };
    RenderPassEncoder pass = wgpu::commandEncoderBeginRenderPass(encoder, &rp_desc);

    wgpu::renderPassEncoderSetPipeline(pass, g.pipeline);
    wgpu::renderPassEncoderSetBindGroup(pass, 0, g.bind_group, 0, null);
    wgpu::renderPassEncoderSetVertexBuffer(pass, 0, g.vb.handle, 0, g.vb.size);
    wgpu::renderPassEncoderSetIndexBuffer(pass, g.ib.handle, g.index_format, 0, g.ib.size);
    wgpu::renderPassEncoderDrawIndexed(pass, g.index_count, 1, 0, 0, 0);
    wgpu::renderPassEncoderEnd(pass);

    CommandBufferDescriptor cb_desc;
    CommandBuffer cmd = wgpu::commandEncoderFinish(encoder, &cb_desc);
    wgpu::queueSubmit(g.queue, 1, &cmd);
    wgpu::surfacePresent(g.surface);
}

// ----------------------------------------------------------------
// Callbacks (called synchronously by JS bridge)
// ----------------------------------------------------------------

fn void on_adapter(RequestAdapterStatus status, Adapter adapter, StringView msg, void* ud1, void* ud2) {
    Adapter* result = (Adapter*)ud1;
    if (status == wgpu::REQUEST_ADAPTER_STATUS_SUCCESS) {
        *result = adapter;
    }
}

fn void on_device(RequestDeviceStatus status, Device device, StringView msg, void* ud1, void* ud2) {
    Device* result = (Device*)ud1;
    if (status == wgpu::REQUEST_DEVICE_STATUS_SUCCESS) {
        *result = device;
    }
}
