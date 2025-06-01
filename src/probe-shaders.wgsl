const M_PI: f32 = 3.14159265358979323846;

// Reduce clutter/keyboard pain
alias float2 = vec2<f32>;
alias float3 = vec3<f32>;
alias float4 = vec4<f32>;
alias uint2 = vec2<u32>;
alias int2 = vec2<i32>;
alias int3 = vec3<i32>;

// TODO: Would need to write a custom webpack loader for wgsl that
// processes #include to be able to #include this
struct LCGRand {
     state: u32,
};

// Used for storing SH coefficients for probes (we are using the second order of SH)
struct SHCoefficients {
    L00: float3,
    L1m1: float3, L10: float3, L11: float3,
    L2m2: float3, L2m1: float3, L20: float3, L21: float3, L22: float3,
};

struct Probe {
    position: float3,
    sh_coeffs: SHCoefficients,
};

// Used for seeding random number generation
fn murmur_hash3_mix(hash_in: u32, k_in: u32) -> u32
{
    let c1 = 0xcc9e2d51u;
    let c2 = 0x1b873593u;
    let r1 = 15u;
    let r2 = 13u;
    let m = 5u;
    let n = 0xe6546b64u;

    var k = k_in * c1;
    k = (k << r1) | (k >> (32u - r1));
    k *= c2;

    var hash = hash_in ^ k;
    hash = ((hash << r2) | (hash >> (32u - r2))) * m + n;

    return hash;
}

fn murmur_hash3_finalize(hash_in: u32) -> u32
{
    var hash = hash_in ^ (hash_in >> 16u);
    hash *= 0x85ebca6bu;
    hash ^= hash >> 13u;
    hash *= 0xc2b2ae35u;
    hash ^= hash >> 16u;

    return hash;
}

// Random number generators
fn lcg_random(rng: ptr<function, LCGRand>) -> u32
{
    let m = 1664525u;
    let n = 1013904223u;
    (*rng).state = (*rng).state * m + n;
    return (*rng).state;
}

fn lcg_randomf(rng: ptr<function, LCGRand>) -> f32
{
	return ldexp(f32(lcg_random(rng)), -32);
}

fn get_rng(frame_id: u32, pixel: int2, dims: int2) -> LCGRand
{
    var rng: LCGRand;
    rng.state = murmur_hash3_mix(0u, u32(pixel.x + pixel.y * dims.x));
    rng.state = murmur_hash3_mix(rng.state, frame_id);
    rng.state = murmur_hash3_finalize(rng.state);
    return rng;
}

struct VertexInput {
    @location(0) position: float3,
};

struct VertexOutput {
    @builtin(position) position: float4,
    @location(0) transformed_eye: float3,
    @location(1) ray_dir: float3,
    @location(2) color: vec3<f32>,
};

struct ViewParams {
    proj_view: mat4x4<f32>,
    eye_pos: float4,
    volume_scale: float4,
    light_dir: float4,
    frame_id: u32,
    sigma_t_scale: f32, // A scaling factor for the extinction coefficient (controls the density of the volume)
    sigma_s_scale: f32, // A scaling factor for the scattering coefficient (controls how much light scatters).
    draw_probes: u32,
    probe_density: i32,
    probe_samples: i32,
};

@group(0) @binding(0) var<uniform> params: ViewParams;
@group(0) @binding(1) var volume: texture_3d<f32>;
@group(0) @binding(2) var colormap: texture_2d<f32>;
@group(0) @binding(3) var tex_sampler: sampler;
// Used for accumulatinng frames over time to reduce noise
@group(0) @binding(4) var accum_buffer_in: texture_2d<f32>;
@group(0) @binding(5) var accum_buffer_out: texture_storage_2d<rgba32float, write>;
@group(0) @binding(6) var<storage, read_write> probe_data_write: array<Probe>;
@group(0) @binding(7) var<storage, read> probe_data_read: array<Probe>;
// Used for tracking the dirty state of the probe data (probes in need of an update)

@vertex
fn vertex_main(vert: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    // Translate the volume to place its center at the origin to scale it
    var volume_translation = float3(0.5) - params.volume_scale.xyz * 0.5;
    var world_pos = vert.position * params.volume_scale.xyz + volume_translation;
    out.position = params.proj_view * float4(world_pos, 1.0);

    // Transform the eye into the scaled space
    out.transformed_eye = (params.eye_pos.xyz - volume_translation) / params.volume_scale.xyz;
    out.ray_dir = vert.position - out.transformed_eye;
    return out;
}

// Computes the intersection of a ray with a unit axis-aligned bounding box
fn intersect_box(orig: float3, dir: float3) -> float2 {
	var box_min = float3(0.0);
	var box_max = float3(1.0);
	var inv_dir = 1.0 / dir;
	var tmin_tmp = (box_min - orig) * inv_dir;
	var tmax_tmp = (box_max - orig) * inv_dir;
	var tmin = min(tmin_tmp, tmax_tmp);
	var tmax = max(tmin_tmp, tmax_tmp);
	var t0 = max(tmin.x, max(tmin.y, tmin.z));
	var t1 = min(tmax.x, min(tmax.y, tmax.z));
	return float2(t0, t1);
}

fn sample_spherical_direction(s: float2) -> float3 {
    let cos_theta = 1.0 - 2.0 * s.x;
    let sin_theta = sqrt(max(0.0, 1.0 - cos_theta * cos_theta));
    let phi = s.y * 2.0 * M_PI;
    return float3(cos(phi) * sin_theta, sin(phi) * sin_theta, cos_theta);
}

fn linear_to_srgb(x: f32) -> f32 {
	if (x <= 0.0031308) {
		return 12.92 * x;
	}
	return 1.055 * pow(x, 1.0 / 2.4) - 0.055;
}

struct SamplingResult {
    scattering_event: bool,
    color: float3,
    transmittance: f32,
}

// Uses Woodcock tracking to sample a scattering event along a ray. It may return a scattering event with color and transmittance.
fn sample_woodcock(orig: float3,
                   dir: float3,
                   interval: float2,
                   t: ptr<function, f32>,
                   rng: ptr<function, LCGRand>)
                   -> SamplingResult {
    var result: SamplingResult;
    result.scattering_event = false;
    result.color = float3(0.0);
    result.transmittance = 0.0;
    loop {
        let samples = float2(lcg_randomf(rng), lcg_randomf(rng));

        *t -= log(1.0 - samples.x) / params.sigma_t_scale;
        if (*t >= interval.y) {
            break;
        }

        var p = orig + *t * dir;
        var val = textureSampleLevel(volume, tex_sampler, p, 0.0).r;
        // TODO: opacity from transfer function in UI instead of just based on the scalar value
        // Opacity values from the transfer fcn will already be in [0, 1]
        var density = val;
        //var sample_opacity = textureSampleLevel(colormap, tex_sampler, float2(val, 0.5), 0.0).a;
        // Here the sigma t scale will cancel out
        if (density > samples.y) {
            result.scattering_event = true;
            result.color = textureSampleLevel(colormap, tex_sampler, float2(val, 0.5), 0.0).rgb;
            result.transmittance = (1.0 - val);
            break;
        }
    }
    return result;
}

// Estimates transmittance by tracking until an absorption event (returns 0 if absorbed, 1 if not).
fn delta_tracking_transmittance(orig: float3,
                                dir: float3,
                                interval: float2,
                                rng: ptr<function, LCGRand>) -> f32 {
    var transmittance = 1.0;
    var t = interval.x;
    loop {
        let samples = float2(lcg_randomf(rng), lcg_randomf(rng));

        t -= log(1.0 - samples.x) / params.sigma_t_scale;
        if (t >= interval.y) {
            break;
        }

        var p = orig + t * dir;
        var val = textureSampleLevel(volume, tex_sampler, p, 0.0).r;
        // TODO: Sample opacity from colormap
        if (val > samples.y) {
            return 0.0;
        }
    }
    return 1.0;
}

// Estimates transmittance by multiplying the survival probabilities at each step.
fn ratio_tracking_transmittance(orig: float3,
                                dir: float3,
                                interval: float2,
                                rng: ptr<function, LCGRand>) -> f32 {
    var transmittance = 1.0;
    var t = interval.x;
    loop {
        t -= log(1.0 - lcg_randomf(rng)) / params.sigma_t_scale;
        if (t >= interval.y) {
            break;
        }

        var p = orig + t * dir;
        var val = textureSampleLevel(volume, tex_sampler, p, 0.0).r;
        // TODO: Sample from the opacity colormap
        transmittance *= (1.0 - val);
    }
    return transmittance;
}

// Spherical Harmonics basis functions (2nd order)
fn sh_eval_basis(dir: float3) -> SHCoefficients {
    let x = dir.x;
    let y = dir.y;
    let z = dir.z;

    var basis: SHCoefficients;
    // L00 (constant term)
    basis.L00 = float3(0.282095); // 1/(2*sqrt(π))

    // L1m1, L10, L11 (linear terms)
    basis.L1m1 = float3(0.488603 * y); // sqrt(3/(4π)) * y
    basis.L10  = float3(0.488603 * z); // sqrt(3/(4π)) * z
    basis.L11  = float3(0.488603 * x); // sqrt(3/(4π)) * x

    // L2m2, L2m1, L20, L21, L22 (quadratic terms)
    basis.L2m2 = float3(1.092548 * x * y); // 0.5 * sqrt(15/π) * x*y
    basis.L2m1 = float3(1.092548 * y * z); // 0.5 * sqrt(15/π) * y*z
    basis.L20  = float3(0.315392 * (3.0 * z * z - 1.0)); // 0.25 * sqrt(5/π) * (3z²-1)
    basis.L21  = float3(1.092548 * x * z); // 0.5 * sqrt(15/π) * x*z
    basis.L22  = float3(0.546274 * (x * x - y * y)); // 0.25 * sqrt(15/π) * (x²-y²)

    return basis;
}

// Computes the SH coefficients for radiance at a probe position by sampling random directions and tracing radiance.
fn compute_probe_radiance(probe_pos: float3, rng: ptr<function, LCGRand>) -> SHCoefficients {
    var sh_coeffs: SHCoefficients;

    // Initialize SH coefficients to zero
    sh_coeffs.L00 = float3(0.0);
    sh_coeffs.L1m1 = float3(0.0); sh_coeffs.L10 = float3(0.0); sh_coeffs.L11 = float3(0.0);
    sh_coeffs.L2m2 = float3(0.0); sh_coeffs.L2m1 = float3(0.0); sh_coeffs.L20 = float3(0.0); 
    sh_coeffs.L21 = float3(0.0); sh_coeffs.L22 = float3(0.0);

    // Sample radiance from random directions
    for (var i: i32 = 0; i < params.probe_samples; i += 1) {
        let dir = sample_spherical_direction(float2(lcg_randomf(rng), lcg_randomf(rng)));
        var radiance = trace_radiance(probe_pos, dir, rng);

        // Evaluate SH basis for this direction
        let basis = sh_eval_basis(dir);

        // Accumulate into SH coefficients (weighted by solid angle: 4π/params.probe_samples)
        let weight = 4.0 * M_PI / f32(params.probe_samples);
        sh_coeffs.L00 += radiance * basis.L00 * weight;
        sh_coeffs.L1m1 += radiance * basis.L1m1 * weight;
        sh_coeffs.L10 += radiance * basis.L10 * weight;
        sh_coeffs.L11 += radiance * basis.L11 * weight;
        sh_coeffs.L2m2 += radiance * basis.L2m2 * weight;
        sh_coeffs.L2m1 += radiance * basis.L2m1 * weight;
        sh_coeffs.L20 += radiance * basis.L20 * weight;
        sh_coeffs.L21 += radiance * basis.L21 * weight;
        sh_coeffs.L22 += radiance * basis.L22 * weight;
    }

    return sh_coeffs;
}

// Traces a ray from `orig` in direction `dir` and returns the radiance (lighting) from that ray. It uses `sample_woodcock` to find a scattering event and then estimates direct lighting.
fn trace_radiance(orig: float3, dir: float3, rng: ptr<function, LCGRand>) -> float3 {
    var t_interval = intersect_box(orig, dir);
    if (t_interval.x > t_interval.y) {
        return float3(0.0); // No volume intersection
    }

    let light_dir = params.light_dir.xyz;
    let light_emission = params.light_dir.w;
    let ambient_strength = 0.0;
    let volume_emission = 0.5;
    
    var illum = float3(0.0);
    var throughput = float3(1.0);

    var t = t_interval.x;
    var event = sample_woodcock(orig, dir, t_interval, &t, rng);
    if (!event.scattering_event) {
        illum = float3(0.1);
    } else {
        var pos = orig + dir * t;

        // Sample illumination from the direct light
        t_interval = intersect_box(pos, light_dir);
        // We're inside the volume
        t_interval.x = 0.0;

        var light_transmittance = delta_tracking_transmittance(pos, light_dir, t_interval, rng);
        illum += throughput * light_transmittance * float3(light_emission);

        illum += throughput * event.color * volume_emission;

        throughput *= event.color * event.transmittance * params.sigma_s_scale;
    }

    return illum;
}


// Evaluates the SH for a given direction using precomputed coefficients.
fn sh_eval(sh: SHCoefficients, dir: float3) -> float3 {
    let basis = sh_eval_basis(dir);
    return sh.L00 * basis.L00 +
           sh.L1m1 * basis.L1m1 + sh.L10 * basis.L10 + sh.L11 * basis.L11 +
           sh.L2m2 * basis.L2m2 + sh.L2m1 * basis.L2m1 + sh.L20 * basis.L20 +
           sh.L21 * basis.L21 + sh.L22 * basis.L22;
}

// Helper function to get probe index from grid coordinates
fn get_probe_index(ix: i32, iy: i32, iz: i32) -> i32 {
    return iz * params.probe_density * params.probe_density + 
           iy * params.probe_density + 
           ix;
}

// Helper function to clamp grid coordinates
fn clamp_probe_coord(coord: i32) -> i32 {
    return clamp(coord, 0, params.probe_density - 1);
}

// Uses trilinear interpolation of the SH coefficients from the 8 nearest probes to estimate radiance at a position and direction.
fn get_interpolated_radiance(pos: float3, dir: float3) -> float3 {
    // Calculate normalized position in probe grid space [0, 1] -> [0, density-1]
    let grid_pos = pos * f32(params.probe_density - 1);
    
    // Get the 8 surrounding probes
    let ix0 = i32(floor(grid_pos.x));
    let iy0 = i32(floor(grid_pos.y));
    let iz0 = i32(floor(grid_pos.z));
    let ix1 = clamp_probe_coord(ix0 + 1);
    let iy1 = clamp_probe_coord(iy0 + 1);
    let iz1 = clamp_probe_coord(iz0 + 1);
    
    // Get interpolation weights
    let fx = fract(grid_pos.x);
    let fy = fract(grid_pos.y);
    let fz = fract(grid_pos.z);
    
    // Get all 8 probes
    let p000 = probe_data_read[get_probe_index(ix0, iy0, iz0)].sh_coeffs;
    let p001 = probe_data_read[get_probe_index(ix0, iy0, iz1)].sh_coeffs;
    let p010 = probe_data_read[get_probe_index(ix0, iy1, iz0)].sh_coeffs;
    let p011 = probe_data_read[get_probe_index(ix0, iy1, iz1)].sh_coeffs;
    let p100 = probe_data_read[get_probe_index(ix1, iy0, iz0)].sh_coeffs;
    let p101 = probe_data_read[get_probe_index(ix1, iy0, iz1)].sh_coeffs;
    let p110 = probe_data_read[get_probe_index(ix1, iy1, iz0)].sh_coeffs;
    let p111 = probe_data_read[get_probe_index(ix1, iy1, iz1)].sh_coeffs;
    
    // Evaluate SH for each probe color
    let c000 = sh_eval(p000, dir);
    let c001 = sh_eval(p001, dir);
    let c010 = sh_eval(p010, dir);
    let c011 = sh_eval(p011, dir);
    let c100 = sh_eval(p100, dir);
    let c101 = sh_eval(p101, dir);
    let c110 = sh_eval(p110, dir);
    let c111 = sh_eval(p111, dir);
    
    // Trilinear interpolation
    let c00 = mix(c000, c100, fx);
    let c01 = mix(c001, c101, fx);
    let c10 = mix(c010, c110, fx);
    let c11 = mix(c011, c111, fx);
    
    let c0 = mix(c00, c10, fy);
    let c1 = mix(c01, c11, fy);
    
    return mix(c0, c1, fz);
}

/// Generates a 3D grid of probe positions within the unit cube [0,1]
/// Computes SH radiance at each probe and writes results to probe_data_write.
/// Automatically clears the dirty flag after all probes are updated.
@compute @workgroup_size(8, 8, 1)
fn init_probes(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let gx = i32(global_id.x);
    let gy = i32(global_id.y);
    let gz = i32(global_id.z);

    if (gx >= params.probe_density || gy >= params.probe_density || gz >= params.probe_density) {
        return; // Out of bounds
    }

    let idx = i32(gz * params.probe_density * params.probe_density + gy * params.probe_density + gx);
    let total_probes = params.probe_density * params.probe_density * params.probe_density;

    var rng = LCGRand();
    rng.state = u32(idx); // Deterministic RNG seed per probe

    let step = 1.0 / f32(params.probe_density - 1);
    let pos = vec3<f32>(f32(gx), f32(gy), f32(gz)) * step;

    probe_data_write[idx].position = pos;
    probe_data_write[idx].sh_coeffs = compute_probe_radiance(pos, &rng);
}

@fragment
fn fragment_main(in: VertexOutput) -> @location(0) float4 {
    var ray_dir = normalize(in.ray_dir);

    // Poiščemo intersekcijo žarka z volumnom (našim boundim boxom v katerem se nahaja volumen)
	var t_interval = intersect_box(in.transformed_eye, ray_dir);  
	if (t_interval.x > t_interval.y) {  // Če žarek ne vstopi v volumen ga ignoriramo
		discard;
	}
	t_interval.x = max(t_interval.x, 0.0);

    let pixel = int2(i32(in.position.x), i32(in.position.y));
    // Image will be no larger than 1280x720 so we can keep this fixed
    // for picking our RNG value
    var rng = get_rng(params.frame_id, pixel, int2(1280, 720));

    // This should just be 1 for the max density in scivis
    var inv_max_density = 1.0;

    let light_dir = params.light_dir.xyz;
    let light_emission = params.light_dir.w;
    let ambient_strength = 0.0;
    let volume_emission = 0.5;

    var illum = float3(0.0);
    var throughput = float3(1.0);
    var transmittance = 1.0;

    var had_any_event = false;
    var pos = in.transformed_eye;
    // Sample the next scattering event in the volume
    for (var i = 0; i < 4; i += 1) {
        var t = t_interval.x;
        var event = sample_woodcock(pos, ray_dir, t_interval, &t, &rng);

        if (!event.scattering_event) {
            // Illuminate with an "environment light"
            if (had_any_event) {
                illum += throughput * float3(ambient_strength);
            } else {
                illum = float3(0.1);
            }
            break;
        } else {
            had_any_event = true;

            // Update scattered ray position
            pos = pos + ray_dir * t;

            var radiance = get_interpolated_radiance(pos, ray_dir);
            illum += throughput * radiance;
            
            // Include emission from the volume for emission/absorption scivis model
            // Scaling the volume emission by the inverse of the opacity from the transfer function
            // can give some nice effects. Would be cool to provide control of this
            illum += throughput * event.color * volume_emission;// * (1.0 - event.transmittance);
            
            throughput *= event.color * event.transmittance * params.sigma_s_scale;

            // Scatter in a random direction to continue the ray
            ray_dir = sample_spherical_direction(float2(lcg_randomf(&rng), lcg_randomf(&rng)));
            t_interval = intersect_box(pos, ray_dir);
            if (t_interval.x > t_interval.y) {
                illum = float3(0.0, 1.0, 0.0);
                break;
            }
            // We're now inside the volume
            t_interval.x = 0.0;
        }
    }

    var color = float4(illum, 1.0);

    
    // Accumulate into the accumulation buffer for progressive accumulation 
    var accum_color = float4(0.0);
    if (params.frame_id > 0u) {
        accum_color = textureLoad(accum_buffer_in, pixel, 0);
    }
    accum_color += color;
    textureStore(accum_buffer_out, pixel, accum_color);

    color = accum_color / f32(params.frame_id + 1u);
    

    // TODO: background color also needs to be sRGB-mapped, otherwise this
    // causes the volume bounding box to show up incorrectly b/c of the
    // differing brightness
    color.r = linear_to_srgb(color.r);
    color.g = linear_to_srgb(color.g);
    color.b = linear_to_srgb(color.b);
    return color;
}

@vertex
fn probe_vertex_main(
    @builtin(instance_index) instanceIndex: u32,
    @builtin(vertex_index) vertexIndex: u32
) -> VertexOutput {
    var output : VertexOutput;
    let base_quad_size = 0.02;

    // Triangle for quad (two triangles)
    let points = array<vec2<f32>, 6>(
        vec2<f32>(-1.0, -1.0), // Bottom-left
        vec2<f32>( 1.0, -1.0), // Bottom-right
        vec2<f32>(-1.0,  1.0), // Top-left
        vec2<f32>(-1.0,  1.0), // Top-left (repeated)
        vec2<f32>( 1.0, -1.0), // Bottom-right (repeated)
        vec2<f32>( 1.0,  1.0)  // Top-right
    );

    let probe_pos = probe_data_read[instanceIndex].position;
    let probe_sh = probe_data_read[instanceIndex].sh_coeffs;
    
    let world_center = probe_pos * params.volume_scale.xyz;
    let probe_col = sh_eval(probe_sh, normalize(world_center - params.eye_pos.xyz));
    let probe_const_col = probe_data_read[instanceIndex].sh_coeffs.L00;

    // Compute view-aligned billboard axes
    //let cam_forward = normalize(params.eye_pos.xyz); // view direction
    let cam_forward = normalize(params.eye_pos.xyz - world_center);

    let cam_right = normalize(cross(cam_forward, vec3<f32>(0.0, 1.0, 0.0)));
    let cam_up = cross(cam_right, cam_forward);

    let distance = length(world_center - params.eye_pos.xyz);
    let scale_factor = clamp(1.0 / (distance + 0.1), 0.002, 0.8);

    // Apply screen-facing quad orientation
    let quad_offset = (cam_right * points[vertexIndex].x + cam_up * points[vertexIndex].y)
                      * base_quad_size * scale_factor;

    let worldPos = vec4<f32>(world_center + quad_offset, 1.0);
    output.position = params.proj_view * worldPos;
    
    let z = i32(instanceIndex) / (params.probe_density * params.probe_density);

    // Color map for 8 Z-layers
    let colors = array<vec3<f32>, 8>(
        vec3<f32>(1.0, 1.0, 1.0), // white
        vec3<f32>(1.0, 1.0, 0.0), // yellow
        vec3<f32>(1.0, 0.5, 0.0), // orange
        vec3<f32>(0.0, 1.0, 0.0), // green
        vec3<f32>(1.0, 0.0, 0.0), // red
        vec3<f32>(0.0, 0.0, 1.0), // blue
        vec3<f32>(1.0, 0.0, 1.0), // pink
        vec3<f32>(0.0, 0.0, 0.0)  // black
    );
    //output.color = colors[z];
    //output.color = probe_const_col;
    output.color = probe_col;

    return output;
}


@fragment
fn probe_fragment_main(input : VertexOutput) -> @location(0) vec4<f32> {
    return vec4<f32>(input.color, 1.0);
}

