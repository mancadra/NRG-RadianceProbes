import {ArcballCamera} from "arcball_camera";
import {Controller} from "ez_canvas_controller";
import {mat4, vec3} from "gl-matrix";

//import shaderCode from "./shaders.wgsl";
import shaderCode from "./probe-shaders.wgsl";
import {
    colormaps,
    fetchVolume,
    fillSelector,
    getCubeMesh,
    getVolumeDimensions,
    linearToSRGB,
    sphericalDir,
    uploadImage,
    uploadVolume,
    volumes
} from "./volume.js";

let probesNeedUpdate = true;
let renderCount = 0;
let sumTime = 0;
const MAX_PROBE_DENSITY = 64;

export default async function runProbeRenderer() {
    let DRAW_PROBES = document.getElementById("drawProbes").checked ? 1 : 0;
    let PROBE_DENSITY = parseInt(document.getElementById("probeDensity").value);
    let PROBE_SAMPLES = parseInt(document.getElementById("probeSamples").value);
    let DRAW_VOLUME = document.getElementById("drawVolume").checked ? 1 : 0;

    if (navigator.gpu === undefined) {
        document.getElementById("webgpu-canvas").setAttribute("style", "display:none;");
        document.getElementById("no-webgpu").setAttribute("style", "display:block;");
        return;
    }

    console.log("radiance probes");
    console.log("PROBE_DENSITY", PROBE_DENSITY);
    console.log("PROBE_SAMPLES", PROBE_SAMPLES);

    var adapter = await navigator.gpu.requestAdapter();
    if (!adapter) {
        document.getElementById("webgpu-canvas").setAttribute("style", "display:none;");
        document.getElementById("no-webgpu").setAttribute("style", "display:block;");
        return;
    }
    const adapterLimits = adapter.limits;
    //console.log("Adapter's supported maxBufferSize:", adapterLimits.maxBufferBindingSize);
    //var device = await adapter.requestDevice();
    const device = await adapter.requestDevice({
    requiredLimits: {
        maxBufferSize: adapterLimits.maxStorageBufferBindingSize
    }
    });

    //console.log("Device's actual maxBufferSize:", device.limits.maxBufferBindingSize);


    // Get a context to display our rendered image on the canvas
    var canvas = document.getElementById("webgpu-canvas");
    var context = canvas.getContext("webgpu");

    // Setup shader modules
    var shaderModule = device.createShaderModule({code: shaderCode});
    var compilationInfo = await shaderModule.getCompilationInfo();
    if (compilationInfo.messages.length > 0) {
        var hadError = false;
        console.log("Shader compilation log:");
        for (var i = 0; i < compilationInfo.messages.length; ++i) {
            var msg = compilationInfo.messages[i];
            console.log(`${msg.lineNum}:${msg.linePos} - ${msg.message}`);
            hadError = hadError || msg.type == "error";
        }
        if (hadError) {
            console.log("Shader failed to compile");
            return;
        }
    }

    const defaultEye = vec3.set(vec3.create(), 0.5, 0.5, 2.5);
    const center = vec3.set(vec3.create(), 0.5, 0.5, 0.5);
    const up = vec3.set(vec3.create(), 0.0, 1.0, 0.0);

    const cube = getCubeMesh();

    // Upload cube to use to trigger raycasting of the volume
    var vertexBuffer = device.createBuffer({
        size: cube.vertices.length * 4,
        usage: GPUBufferUsage.VERTEX,
        mappedAtCreation: true
    });
    new Float32Array(vertexBuffer.getMappedRange()).set(cube.vertices);
    vertexBuffer.unmap();

    var indexBuffer = device.createBuffer(
        {size: cube.indices.length * 4, usage: GPUBufferUsage.INDEX, mappedAtCreation: true});
    new Uint16Array(indexBuffer.getMappedRange()).set(cube.indices);
    indexBuffer.unmap();

    var viewParamsSize = (20 + 4 * 4) * 4;
    var viewParamsBuffer = device.createBuffer(
        {size: viewParamsSize, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST});

    var sampler = device.createSampler({
        magFilter: "linear",
        minFilter: "linear",
    });

    var volumePicker = document.getElementById("volumeList");
    var colormapPicker = document.getElementById("colormapList");

    fillSelector(volumePicker, volumes);
    fillSelector(colormapPicker, colormaps);

    // Fetch and upload the volume
    var volumeName = "Bonsai";
    if (window.location.hash) {
        var linkedDataset = decodeURI(window.location.hash.substring(1));
        if (linkedDataset in volumes) {
            volumePicker.value = linkedDataset;
            volumeName = linkedDataset;
        } else {
            alert(`Linked to invalid data set ${linkedDataset}`);
            return;
        }
    }

    var volumeDims = getVolumeDimensions(volumes[volumeName]);
    const longestAxis = Math.max(volumeDims[0], Math.max(volumeDims[1], volumeDims[2]));
    var volumeScale = [
        volumeDims[0] / longestAxis,
        volumeDims[1] / longestAxis,
        volumeDims[2] / longestAxis
    ];

    var colormapName = "Cool Warm";
    var colormapTexture = await uploadImage(device, colormaps[colormapName]);

    var volumeTexture =
        await fetchVolume(volumes[volumeName])
            .then((volumeData) => { return uploadVolume(device, volumeDims, volumeData); });

    // We need to ping-pong the accumulation buffers because read-write storage textures are
    // missing and we can't have the same texture bound as both a read texture and storage
    // texture
    var accumBuffers = [
        device.createTexture({
            size: [canvas.width, canvas.height, 1],
            format: "rgba32float",
            usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.STORAGE_BINDING
        }),
        device.createTexture({
            size: [canvas.width, canvas.height, 1],
            format: "rgba32float",
            usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.STORAGE_BINDING
        })
    ];

    var accumBufferViews = [accumBuffers[0].createView(), accumBuffers[1].createView()];

    //const bufferSize = MAX_PROBE_DENSITY ** 3 * (4 + 9 * 4) * 4; // position + SH coefficients (vec3[9]) + padding
    const bufferSize = MAX_PROBE_DENSITY ** 3 * 160;

    var probeBufferWrite = device.createBuffer({
        size:  bufferSize,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC});
    //console.log(bufferSize);

    var probeBufferRead = device.createBuffer({
        size: bufferSize,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC});
        probeBufferWrite.label = "probeBufferWrite";

    
    /*
    // Initialize probes to zero
    device.queue.writeBuffer(probeBufferWrite, 0, new Float32Array(bufferSize));
    device.queue.writeBuffer(probeBufferRead, 0, new Float32Array(bufferSize));
    */

    // Setup render outputs
    var swapChainFormat = "bgra8unorm";
    context.configure({
        device: device,
        format: swapChainFormat,
        usage: GPUTextureUsage.OUTPUT_ATTACHMENT,
        alphaMode: "premultiplied"
    });

    var bindGroupLayout = device.createBindGroupLayout({
        entries: [
            {
                binding: 0,
                visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT | GPUShaderStage.COMPUTE,
                buffer: {type: "uniform"}
            },
            {binding: 1, visibility: GPUShaderStage.FRAGMENT | GPUShaderStage.COMPUTE, texture: {viewDimension: "3d"}},
            {binding: 2, visibility: GPUShaderStage.FRAGMENT | GPUShaderStage.COMPUTE, texture: {viewDimension: "2d"}},
            {binding: 3, visibility: GPUShaderStage.FRAGMENT | GPUShaderStage.COMPUTE, sampler: {type: "filtering"}},
            {
                binding: 4,
                visibility: GPUShaderStage.FRAGMENT,
                texture: {sampleType: "unfilterable-float", viewDimension: "2d"}
            },
            {
                binding: 5,
                visibility: GPUShaderStage.FRAGMENT,
                storageTexture: {
                    // Would be great to have read-write back
                    access: "write-only",
                    format: "rgba32float"
                }
            },
            {
                binding: 6,
                visibility: GPUShaderStage.COMPUTE,
                buffer: { 
                    type: 'storage',
                    hasDynamicOffset: false,
                    minBindingSize: 0
                }
            },
            {
                binding: 7,
                visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT,
                buffer: { 
                    type: 'read-only-storage'
                }
            },
        ]
    });

    // Create render pipeline
    var layout = device.createPipelineLayout({bindGroupLayouts: [bindGroupLayout]});
    var computePipeline = device.createComputePipeline({
        layout: device.createPipelineLayout({
            bindGroupLayouts: [bindGroupLayout]
        }),
        compute: {
            module: shaderModule,
            entryPoint: "init_probes"
        }
    });

    var vertexState = {
        module: shaderModule,
        entryPoint: "vertex_main",
        buffers: [{
            arrayStride: 3 * 4,
            attributes: [{format: "float32x3", offset: 0, shaderLocation: 0}]
        }]
    };

    var fragmentState = {
        module: shaderModule,
        entryPoint: "fragment_main",
        targets: [{
            format: swapChainFormat,
            blend: {
                color: {srcFactor: "one", dstFactor: "one-minus-src-alpha"},
                alpha: {srcFactor: "one", dstFactor: "one-minus-src-alpha"}
            }
        }]
    };

    var renderPipeline = device.createRenderPipeline({
        layout: layout,
        vertex: vertexState,
        fragment: fragmentState,
        primitive: {
            topology: "triangle-strip",
            stripIndexFormat: "uint16",
            cullMode: "front",
        }
    });

    var clearColor = linearToSRGB(0.1);
    var renderPassDesc = {
        colorAttachments: [{
            view: undefined,
            loadOp: "clear",
            storeOp: "store",
            clearValue: [clearColor, clearColor, clearColor, 1]
        }]
    };

    const probeVertexState = {
        module: shaderModule,
        entryPoint: "probe_vertex_main",
        buffers: []
    };

    const probeFragmentState = {
        module: shaderModule,
        entryPoint: "probe_fragment_main",
        targets: [{
            format: swapChainFormat
        }]
    };

    const probePipeline = device.createRenderPipeline({
        layout: layout,
        vertex: probeVertexState,
        fragment: probeFragmentState,
        primitive: {
            topology: "triangle-list",
            cullMode: "none"
        }
    });

    var camera = new ArcballCamera(defaultEye, center, up, 2, [canvas.width, canvas.height]);
    var proj = mat4.perspective(
        mat4.create(), 50 * Math.PI / 180.0, canvas.width / canvas.height, 0.1, 100);
    var projView = mat4.create();

    var frameId = 0;

    // Register mouse and touch listeners
    var controller = new Controller();
    controller.mousemove = function(prev, cur, evt) {
        if (evt.buttons == 1) {
            frameId = 0;
            camera.rotate(prev, cur);

        } else if (evt.buttons == 2) {
            frameId = 0;
            camera.pan([cur[0] - prev[0], prev[1] - cur[1]]);
        }
    };
    controller.wheel = function(amt) {
        frameId = 0;
        camera.zoom(amt);
    };
    controller.pinch = controller.wheel;
    controller.twoFingerDrag = function(drag) {
        frameId = 0;
        camera.pan(drag);
    };
    controller.registerForCanvas(canvas);

    // Reset accumulation when the light parameters change
    var lightPhiSlider = document.getElementById("phiRange");
    var lightThetaSlider = document.getElementById("thetaRange");
    var lightStrengthSlider = document.getElementById("lightStrength");
    lightPhiSlider.oninput = function() {
        frameId = 0;
    };
    lightThetaSlider.oninput = function() {
        frameId = 0;
    };
    lightStrengthSlider.oninput = function() {
        frameId = 0;
    };

    var lightPhiValue = lightPhiSlider.value;
    var lightThetaValue = lightThetaSlider.value;
    var lightStrengthValue = lightStrengthSlider.value;

    var bindGroupEntries = [
        {binding: 0, resource: {buffer: viewParamsBuffer}},
        {binding: 1, resource: volumeTexture.createView()},
        {binding: 2, resource: colormapTexture.createView()},
        {binding: 3, resource: sampler},
        // Updated each frame because we need to ping pong the accumulation buffers
        {binding: 4, resource: accumBufferViews[0]},
        {binding: 5, resource: accumBufferViews[1]},
        {binding: 6, resource: { buffer: probeBufferWrite } },
        {binding: 7, resource: { buffer: probeBufferRead } },
    ];

    var bindGroup = device.createBindGroup({layout: bindGroupLayout, entries: bindGroupEntries});

    var upload = device.createBuffer({
        size: viewParamsSize,
        usage: GPUBufferUsage.MAP_WRITE | GPUBufferUsage.COPY_SRC,
        mappedAtCreation: false
    });

    var sigmaTScale = 100.0;  // 100 je relativno malo če želimo videti sence
    var sigmaSScale = 1.0;

    async function updateProbes(device, frameId) {
        console.log(`Updating probes for frame ${frameId} (probe density = ${PROBE_DENSITY}, probe samples = ${PROBE_SAMPLES})`);
        const start = performance.now();
        projView = mat4.mul(projView, proj, camera.camera);

        var lightDir = sphericalDir(lightThetaSlider.value, lightPhiSlider.value);
        {
            await upload.mapAsync(GPUMapMode.WRITE);
            var eyePos = camera.eyePos();
            var map = upload.getMappedRange();
            var f32map = new Float32Array(map);
            var u32map = new Uint32Array(map);
            var i32map = new Int32Array(map);

            // TODO: A struct layout size computer/writer utility would help here
            f32map.set(projView, 0);
            f32map.set(eyePos, 16);
            f32map.set(volumeScale, 16 + 4);
            f32map.set(lightDir, 16 + 4 * 2);
            f32map.set([lightStrengthSlider.value], 16 + 4 * 2 + 3);
            u32map[28] = frameId;
            f32map[29] = sigmaTScale;
            f32map[30] = sigmaSScale;
            u32map[31] = DRAW_PROBES;
            i32map[32] = PROBE_DENSITY;
            i32map[33] = PROBE_SAMPLES;

            upload.unmap();
        }
        
        {
            // Dispatch compute shader
            const computeEncoder = device.createCommandEncoder();
            const computePass = computeEncoder.beginComputePass();
            computePass.setPipeline(computePipeline);
            computePass.setBindGroup(0, bindGroup);

            const workgroupSize = 8;
            const dispatchCount = Math.ceil(PROBE_DENSITY / workgroupSize);
            //computePass.dispatchWorkgroups(1, 1, 8);
            computePass.dispatchWorkgroups(Math.ceil(PROBE_DENSITY / workgroupSize), Math.ceil(PROBE_DENSITY / workgroupSize), PROBE_DENSITY);
            //computePass.dispatchWorkgroups(dispatchCount, dispatchCount, dispatchCount);
            computePass.end();
            device.queue.submit([computeEncoder.finish()]);
            await device.queue.onSubmittedWorkDone();
        }

        {
            const copyEncoder = device.createCommandEncoder();
            copyEncoder.copyBufferToBuffer(probeBufferWrite, 0, probeBufferRead, 0, bufferSize);
            await device.queue.submit([copyEncoder.finish()]);
            await device.queue.onSubmittedWorkDone();
        }

        const end = performance.now();
        const time = end - start;
        console.log(`Probe initialization time(probe density = ${PROBE_DENSITY}, probe samples = ${PROBE_SAMPLES}): \n${time} ms`);
    }

    //await updateProbes(device, 0);

    const render = async () => {
        const start = performance.now();
        const copyEncoder = device.createCommandEncoder();
        copyEncoder.copyBufferToBuffer(probeBufferWrite, 0, probeBufferRead, 0, bufferSize);
        await device.queue.submit([copyEncoder.finish()]);
        await device.queue.onSubmittedWorkDone();

        // Fetch a new volume or colormap if a new one was selected and update the probes
        if (volumeName != volumePicker.value) {
            volumeName = volumePicker.value;
            history.replaceState(history.state, "", "#" + volumeName);

            volumeDims = getVolumeDimensions(volumes[volumeName]);
            const longestAxis =
                Math.max(volumeDims[0], Math.max(volumeDims[1], volumeDims[2]));
            volumeScale = [
                volumeDims[0] / longestAxis,
                volumeDims[1] / longestAxis,
                volumeDims[2] / longestAxis
            ];

            volumeTexture = await fetchVolume(volumes[volumeName]).then((volumeData) => {
                return uploadVolume(device, volumeDims, volumeData);
            });

            // Reset accumulation and update the bindgroup
            frameId = 0;
            bindGroupEntries[1].resource = volumeTexture.createView();
            probesNeedUpdate = true;
        }

        if (colormapName != colormapPicker.value) {
            colormapName = colormapPicker.value;
            colormapTexture = await uploadImage(device, colormaps[colormapName]);

            // Reset accumulation and update the bindgroup
            frameId = 0;
            bindGroupEntries[2].resource = colormapTexture.createView();
            probesNeedUpdate = true;
        }

        // Update camera buffer
        projView = mat4.mul(projView, proj, camera.camera);

        var lightDir = sphericalDir(lightThetaSlider.value, lightPhiSlider.value);

        if (lightPhiValue != lightPhiSlider.value || lightThetaValue != lightThetaSlider.value || lightStrengthValue != lightStrengthSlider.value) {
            lightPhiValue = lightPhiSlider.value;
            lightThetaValue = lightThetaSlider.value;
            lightStrengthValue = lightStrengthSlider.value;
            probesNeedUpdate = true;
        }

        let draw_probes_value = document.getElementById("drawProbes").checked ? 1 : 0;
        if (DRAW_PROBES != draw_probes_value) {
            DRAW_PROBES = draw_probes_value;
        }

        let draw_volume_value = document.getElementById("drawVolume").checked ? 1 : 0;
        if (DRAW_VOLUME != draw_volume_value) {
            DRAW_VOLUME = draw_volume_value;
            probesNeedUpdate = true;
            frameId = 0;
        }

        if (PROBE_DENSITY != parseInt(document.getElementById("probeDensity").value) || PROBE_SAMPLES != parseInt(document.getElementById("probeSamples").value)) {
            frameId = 0;
            PROBE_DENSITY = parseInt(document.getElementById("probeDensity").value);
            PROBE_SAMPLES = parseInt(document.getElementById("probeSamples").value);
            probesNeedUpdate = true;
        }

        if (probesNeedUpdate) {
            //device.queue.writeBuffer(probeBufferWrite, 0, new Float32Array(bufferSize));
            //device.queue.writeBuffer(probeBufferRead, 0, new Float32Array(bufferSize));
            
            await updateProbes(device, frameId);
            frameId = 0;
            //renderCount = 0;
            //sumTime = 0;
            probesNeedUpdate = false;
        }

        {
            await upload.mapAsync(GPUMapMode.WRITE);
            var eyePos = camera.eyePos();
            var map = upload.getMappedRange();
            var f32map = new Float32Array(map);
            var u32map = new Uint32Array(map);
            var i32map = new Int32Array(map);

            // TODO: A struct layout size computer/writer utility would help here
            f32map.set(projView, 0);
            f32map.set(eyePos, 16);
            f32map.set(volumeScale, 16 + 4);
            f32map.set(lightDir, 16 + 4 * 2);
            f32map.set([lightStrengthSlider.value], 16 + 4 * 2 + 3);
            u32map[28] = frameId;
            f32map[29] = sigmaTScale;
            f32map[30] = sigmaSScale;
            u32map[31] = DRAW_PROBES;
            i32map[32] = PROBE_DENSITY;
            i32map[33] = PROBE_SAMPLES;

            upload.unmap();
        }

        bindGroupEntries[4].resource = accumBufferViews[frameId % 2];
        bindGroupEntries[5].resource = accumBufferViews[(frameId + 1) % 2];

        bindGroupEntries[6].resource = { buffer: probeBufferWrite };
        bindGroupEntries[7].resource = { buffer: probeBufferRead };

        bindGroup = device.createBindGroup({layout: bindGroupLayout, entries: bindGroupEntries});

        var commandEncoder = device.createCommandEncoder();
        commandEncoder.copyBufferToBuffer(upload, 0, viewParamsBuffer, 0, viewParamsSize);

        renderPassDesc.colorAttachments[0].view = context.getCurrentTexture().createView();
        var renderPass = commandEncoder.beginRenderPass(renderPassDesc);

        if (DRAW_VOLUME) {
            renderPass.setPipeline(renderPipeline);
            renderPass.setBindGroup(0, bindGroup);
            renderPass.setVertexBuffer(0, vertexBuffer);
            renderPass.setIndexBuffer(indexBuffer, "uint16");
            renderPass.draw(cube.vertices.length / 3, 1, 0, 0);
        }

        if (DRAW_PROBES) {
            //console.log(`Rendering ${PROBE_DENSITY}^3 radiance probes for ${volumeName}`);
            renderPass.setPipeline(probePipeline);
            renderPass.setBindGroup(0, bindGroup);
            renderPass.draw(6, PROBE_DENSITY ** 3, 0, 0);
        }
        
        renderPass.end();
        device.queue.submit([commandEncoder.finish()]);
        
        frameId += 1;
        requestAnimationFrame(render);

        const end = performance.now();
        sumTime += end - start;
        renderCount++;
        if (renderCount == 1000) {
            console.log(`Average render time for radiance probes (probe density = ${PROBE_DENSITY}, probe samples = ${PROBE_SAMPLES}): \n${sumTime / renderCount} ms`);
            sumTime = 0;
            renderCount = 0;
        }
    };
    requestAnimationFrame(render);
}
