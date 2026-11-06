#include "scene.slang"
#include "feet.slang"
#include "optics.slang"

Texture2D<float4> g_state;
RWTexture2D<float4> g_output;
RWTexture2D<float4> g_opticalData; // F, underwater path (m), transmitted ID, reflected ID.
uniform uint g_resolution;
uniform float g_extent;
uniform float2 g_domainCenter;
uniform float g_edgeWidth;
uniform float g_cameraSpan;
uniform uint g_domainDebug;
uniform float g_time;
uniform float2 g_player;
uniform FootData g_feet[8];
uniform uint g_footCount;
uniform float g_depth;
uniform float g_zoom;
uniform float2 g_cameraCenter;
uniform uint g_footDebug;
uniform float g_normalStrength;
uniform float g_detail;
uniform uint g_view;
uniform uint g_topDown;
uniform uint g_openWater;
uniform float g_ior;
uniform float g_roughness;
uniform float g_absorption;
uniform float3 g_sunDirection;
uniform uint g_refraction;
uniform uint g_reflection;
uniform uint g_floorGrid;

float4 stateTexel(int2 cell)
{
    if (any(cell < 0) || any(cell >= int(g_resolution))) return float4(0);
    return g_state[cell];
}

float4 sampleState(float2 p)
{
    float2 local = p - g_domainCenter;
    float edge = g_extent * 0.5 - max(abs(local.x), abs(local.y));
    if (edge <= 0) return float4(0);
    float2 texel = (local / g_extent + 0.5) * g_resolution - 0.5;
    int2 cell = int2(floor(texel));
    float2 t = frac(texel);
    float4 state = lerp(lerp(stateTexel(cell), stateTexel(cell + int2(1, 0)), t.x),
                        lerp(stateTexel(cell + int2(0, 1)), stateTexel(cell + 1), t.x), t.y);
    return state * smoothstep(0.0, g_edgeWidth, edge);
}

float depthAt(float2 p)
{
    return g_openWater != 0 ? g_depth : g_depth * smoothstep(0.0, 0.35, waterMargin(p));
}

float3 groundColor(float2 p)
{
    float n = noise2(p * 1.8) * 0.6 + noise2(p * 7.0) * 0.4;
    float3 sand = lerp(float3(0.20, 0.23, 0.18), float3(0.40, 0.43, 0.29), n);
    float margin = waterMargin(p);
    float moss = 1 - smoothstep(-2.4, -0.6, margin);
    float3 ground = lerp(sand, float3(0.065, 0.13, 0.095) * (0.8 + n * 0.7), moss);
    float2 tile = p * 0.9;
    tile.x += floor(tile.y) * 0.5;
    float2 edge = min(frac(tile), 1.0 - frac(tile));
    float joint = 1.0 - smoothstep(0.015, 0.04, min(edge.x, edge.y));
    float paving = (1 - smoothstep(-1.8, -0.9, margin)) * 0.45;
    ground *= 1.0 - joint * paving;
    if (g_floorGrid != 0 && depthAt(p) > 0)
    {
        float2 cell = p * 4.0; // 25 cm calibration tiles, fixed in world space.
        float2 border = min(frac(cell), 1 - frac(cell));
        float line = 1 - smoothstep(0.025, 0.065, min(border.x, border.y));
        float checker = frac((floor(cell.x) + floor(cell.y)) * 0.5) * 2;
        ground = lerp(float3(0.34, 0.40, 0.34), float3(0.17, 0.26, 0.25), checker);
        ground = lerp(ground, float3(0.025, 0.055, 0.06), line);
    }
    return ground;
}

float3 sky(float3 direction)
{
    float y = saturate(direction.y);
    float2 p = direction.xz / (y + 0.35) * 1.7;
    float clouds = noise2(p) * 0.60 + noise2(p * 2.1) * 0.27 + noise2(p * 4.3) * 0.13;
    clouds = smoothstep(0.46, 0.70, clouds);
    float3 clear = lerp(float3(0.60, 0.72, 0.77), float3(0.055, 0.18, 0.38), pow(y, 0.45));
    // Sun is excluded here; its energy is evaluated by the GGX direct-light term.
    return lerp(clear, float3(0.85, 0.88, 0.87), clouds * 0.85);
}

float3 footCoordinates(float3 p, FootData foot)
{
    float2 xz = footLocal(p.xz - foot.position.xz, foot.forward);
    return float3(xz.x, p.y - foot.position.y, xz.y);
}

float footHit(float3 origin, float3 direction, FootData foot)
{
    float2 d = footLocal(direction.xz, foot.forward);
    return ellipsoidHit(footCoordinates(origin, foot), float3(d.x, direction.y, d.y),
                        float3(foot.halfSize.x, foot.halfHeight, foot.halfSize.y));
}

float3 footNormal(float3 p, FootData foot)
{
    float3 radii = float3(foot.halfSize.x, foot.halfHeight, foot.halfSize.y);
    float3 n = footCoordinates(p, foot) / (radii * radii);
    float2 xz = n.x * float2(-foot.forward.y, foot.forward.x) + n.z * foot.forward;
    return normalize(float3(xz.x, n.y, xz.y));
}

float3 rockCenter(uint id)
{
    return id == 3 ? float3(-3.4, 0.35, 1.8) : float3(4.0, 0.25, -2.8);
}

float3 rockRadii(uint id)
{
    return id == 3 ? float3(1.12, 0.55, 0.82) : float3(0.65, 0.43, 0.98);
}

float3 footColor(uint index)
{
    if (index < 2) return index == 0 ? float3(0.95, 0.36, 0.07) : float3(0.06, 0.52, 0.85);
    float3 color = index < 4 ? float3(0.12, 0.82, 0.32)
                 : index < 6 ? float3(0.72, 0.20, 0.85) : float3(0.98, 0.74, 0.08);
    return color * (index % 2 == 0 ? 1.0 : 0.72);
}

struct SceneHit { float t; uint id; };

SceneHit traceSolids(float3 origin, float3 direction, float maxT)
{
    SceneHit hit;
    hit.t = maxT;
    hit.id = 0;
    float t;
    for (uint i = 0; i < g_footCount; ++i)
    {
        t = footHit(origin, direction, g_feet[i]);
        if (t > 0 && t < hit.t) { hit.t = t; hit.id = i < 2 ? i + 1 : i + 3; }
    }
    // Rocks sit inside the existing solid island footprints.
    if (g_openWater == 0)
    {
        for (uint id = 3; id <= 4; ++id)
        {
            t = ellipsoidHit(origin - rockCenter(id), direction, rockRadii(id));
            if (t > 0 && t < hit.t) { hit.t = t; hit.id = id; }
        }
    }
    return hit;
}

float3 shadeSolid(float3 p, float3 direction, uint id)
{
    float3 normal, base;
    if (id <= 2 || id >= 5)
    {
        uint index = id <= 2 ? id - 1 : id - 3;
        FootData foot = g_feet[index];
        normal = footNormal(p, foot);
        base = footColor(index);
        float3 local = footCoordinates(p, foot);
        base *= 0.35 + 0.65 * smoothstep(-0.06, -0.04, local.y);
        base *= 1 + 0.35 * smoothstep(0.07, 0.11, local.z);
    }
    else
    {
        float3 radii = rockRadii(id);
        normal = normalize((p - rockCenter(id)) / (radii * radii));
        float grain = noise2(p.xz * 9 + p.y * 3);
        base = lerp(float3(0.10, 0.14, 0.13), float3(0.28, 0.32, 0.28), grain);
    }
    float3 lit = base * (0.30 + 0.70 * saturate(dot(normal, g_sunDirection)));
    float3 halfway = g_sunDirection - direction;
    if (dot(halfway, halfway) > 1e-6)
        lit += pow(saturate(dot(normal, normalize(halfway))), 60) * ((id <= 2 || id >= 5) ? 0.25 : 0.04);
    return lit;
}

float terrainHit(float3 origin, float3 direction)
{
    // World-space height intersection, no screen-space depth buffer.
    float low = max(0.0, -origin.y / direction.y);
    float high = max(low, (-g_depth - origin.y) / direction.y);
    if (g_depth <= 1e-6) return low;
    float2 a = (origin + direction * low).xz;
    float2 b = (origin + direction * high).xz;
    // Conservative flat-region fast path. waterMargin has Lipschitz bound < 2.
    if (g_openWater != 0 || waterMargin(a) > 0.35 + 2 * length(b - a))
        return high;
    // Find the first bracket across the shallow sloping bank, then bisect it.
    float end = high;
    for (uint i = 1; i <= 12; ++i)
    {
        high = lerp(max(0.0, -origin.y / direction.y), end, float(i) / 12);
        float3 p = origin + direction * high;
        if (p.y + depthAt(p.xz) <= 0) break;
        low = high;
    }
    for (uint j = 0; j < 12; ++j)
    {
        float middle = (low + high) * 0.5;
        float3 p = origin + direction * middle;
        if (p.y + depthAt(p.xz) > 0) low = middle; else high = middle;
    }
    return (low + high) * 0.5;
}

float3 shadeBottom(float3 point)
{
    float3 color = groundColor(point.xz);
    float e = 0.015;
    float2 gradient = float2(depthAt(point.xz + float2(e, 0)) - depthAt(point.xz - float2(e, 0)),
                             depthAt(point.xz + float2(0, e)) - depthAt(point.xz - float2(0, e))) / (2 * e);
    float3 normal = normalize(float3(gradient.x, 1, gradient.y));
    float visibility = traceSolids(point + g_sunDirection * RAY_EPSILON, g_sunDirection, 40).id == 0 ? 1.0 : 0.3;
    color *= 0.3 + 0.7 * saturate(dot(normal, g_sunDirection)) * visibility;
    // Decorative caustics only: no claim of refractive light transport here.
    float caustic = pow(saturate(0.5 + 0.5 * sin(point.x * 7 + sin(point.z * 5 + g_time))), 14)
                  * pow(saturate(0.5 + 0.5 * sin(point.z * 6 - g_time * 0.6)), 5);
    return color + float3(0.12, 0.18, 0.12) * caustic * g_detail;
}

float3 transmittedRadiance(float3 surface, float3 direction, out float pathLength, out uint material)
{
    float3 origin = surface + direction * RAY_EPSILON;
    float terrainT = terrainHit(origin, direction);
    SceneHit hit = traceSolids(origin, direction, terrainT);
    float3 point = origin + direction * hit.t;
    float3 radiance = hit.id == 0 ? shadeBottom(point) : shadeSolid(point, direction, hit.id);
    pathLength = max(0.0, hit.t + RAY_EPSILON);
    material = hit.id;
    float3 transmission = beerLambert(float3(6.0, 1.4, 0.7) * g_absorption, pathLength);
    return radiance * transmission + float3(0.018, 0.13, 0.16) * (1 - transmission);
}

float3 reflectedRay(float3 surface, float3 direction, out uint material)
{
    direction.y = max(0.005, direction.y);
    direction = normalize(direction);
    float3 origin = surface + direction * RAY_EPSILON;
    SceneHit hit = traceSolids(origin, direction, 100);
    material = hit.id;
    return hit.id == 0 ? sky(direction) : shadeSolid(origin + direction * hit.t, direction, hit.id);
}

float3 reflectedRadiance(float3 surface, float3 direction, out uint material)
{
    float3 color = reflectedRay(surface, direction, material) * 0.4;
    float3 tangent = normalize(cross(direction, abs(direction.y) < 0.95 ? float3(0, 1, 0) : float3(1, 0, 0)));
    float3 bitangent = cross(direction, tangent);
    float spread = g_roughness * g_roughness * 2;
    uint unused;
    // Stable five-ray cone filter approximates rough local/environment reflection.
    color += 0.15 * reflectedRay(surface, normalize(direction + tangent * spread), unused);
    color += 0.15 * reflectedRay(surface, normalize(direction - tangent * spread), unused);
    color += 0.15 * reflectedRay(surface, normalize(direction + bitangent * spread), unused);
    color += 0.15 * reflectedRay(surface, normalize(direction - bitangent * spread), unused);
    return color;
}

float3 surfaceNormal(float2 p, float2 gradient, float3 incident)
{
    float2 detail = float2(cos(p.x * 7 + p.y * 4 + g_time * 1.6),
                          sin(p.y * 8 - p.x * 3 - g_time * 1.3)) * 0.009 * g_detail;
    float3 normal = normalize(float3(-gradient.x * g_normalStrength + detail.x, 1,
                                     -gradient.y * g_normalStrength + detail.y));
    // The interface is a reference plane. Keep secondary reflection in its air
    // hemisphere instead of leaking an unmodelled second water bounce.
    for (uint i = 0; i < 8 && reflect(incident, normal).y < 0.02; ++i)
        normal = normalize(normal + float3(0, 0.4, 0));
    return normal;
}

float3 footOverlay(float3 color, float2 p, FootData foot, float3 tint)
{
    float2 local = footLocal(p - foot.position.xz, foot.forward) / (foot.halfSize * 1.35);
    float outline = 1 - smoothstep(0.04, 0.12, abs(length(local) - 1));
    color = lerp(color, immersion(foot.position.y, foot.halfHeight) > 0.01 ? tint : float3(0.9, 0.65, 0.12), outline * 0.85);
    float2 target = abs(p - foot.target);
    float crossMark = (1 - smoothstep(0.012, 0.025, min(target.x, target.y)))
                    * (1 - smoothstep(0.09, 0.12, max(target.x, target.y)));
    return lerp(color, tint, crossMark * (foot.swinging != 0 ? 0.9 : 0));
}

[shader("compute")]
[numthreads(8, 8, 1)]
void render(uint3 tid : SV_DispatchThreadID)
{
    uint2 size;
    g_output.GetDimensions(size.x, size.y);
    if (any(tid.xy >= size)) return;
    float2 screen = (float2(tid.xy) + 0.5) / float2(size) * 2 - 1;
    screen.y = -screen.y;
    float aspect = float(size.x) / size.y;
    float halfWidth = g_cameraSpan * 0.59 * max(1.0, aspect / 1.65) / g_zoom;
    float3 incident = g_topDown != 0 ? float3(0, -1, 0) : normalize(float3(0, -1, -0.78));
    float3 up = float3(0, -incident.z, incident.y);
    float3 origin = -incident * 30 + float3(screen.x * halfWidth, 0, 0)
                  + up * screen.y * halfWidth / aspect + float3(g_cameraCenter.x, 0, g_cameraCenter.y);
    float planeT = -origin.y / incident.y;
    float3 surface = origin + incident * planeT;
    surface.y = 0;
    float2 p = surface.xz;
    bool wet = depthAt(p) > 1e-6;
    float4 state = sampleState(p);
    float dx = g_extent / g_resolution;
    float2 gradient = float2(sampleState(p + float2(dx, 0)).x - sampleState(p - float2(dx, 0)).x,
                             sampleState(p + float2(0, dx)).x - sampleState(p - float2(0, dx)).x) / (2 * dx);
    float3 color = groundColor(p);
    float4 opticalData = float4(0);
    if (wet && (g_view == 0 || g_view >= 4))
    {
        float3 normal = surfaceNormal(p, gradient, incident);
        float3 transmitted = g_refraction != 0 ? refract(incident, normal, 1.0 / g_ior) : incident;
        float path;
        uint transmittedID, reflectedID = 0;
        float3 transmission = transmittedRadiance(surface, transmitted, path, transmittedID);
        float fresnel = dielectricFresnel(saturate(dot(-incident, normal)), 1, g_ior);
        float3 reflection = float3(0);
        float directSun = 0;
        if (g_reflection != 0)
        {
            reflection = reflectedRadiance(surface, reflect(incident, normal), reflectedID);
            float visibility = traceSolids(surface + g_sunDirection * RAY_EPSILON, g_sunDirection, 40).id == 0 ? 1 : 0;
            directSun = waterSunBRDF(normal, -incident, g_sunDirection, g_roughness, g_ior) * visibility * 2.8;
        }
        color = g_reflection != 0 ? transmission * (1 - fresnel) + reflection * fresnel
                                  + float3(1, 0.92, 0.77) * directSun : transmission;
        float foam = smoothstep(0.035, 0.16, state.z) * smoothstep(0.06, 0.24, length(gradient)) * 0.45;
        color = lerp(color, float3(0.65, 0.85, 0.79), foam);
        if (g_view == 4) color = transmission;
        if (g_view == 5) color = reflection; // Unweighted reflection, for inspection.
        if (g_view == 6) color = float3(fresnel);
        opticalData = float4(fresnel, path, float(transmittedID), float(reflectedID));
    }
    if (g_view == 1)
        color = wet ? lerp(float3(0.025, 0.04, 0.065), state.x > 0 ? float3(0.1, 0.8, 0.65) : float3(0.95, 0.28, 0.1),
                           saturate(abs(state.x) * 60)) : float3(0.12, 0.13, 0.14);
    if (g_view == 2)
        color = wet ? float3(0.12 + max(0.0, state.y), 0.12 + state.z * 2, 0.12 + max(0.0, -state.y)) : float3(0.04);
    if (g_view == 3) color = wet ? float3(0.15, 0.55, 0.48) : float3(0.055);
    // Only geometry reached BEFORE the water plane may cover the optical result.
    // Submerged feet are exclusively shaded through the transmitted ray above.
    float primaryLimit = planeT;
    if (g_view >= 1 && g_view <= 3) primaryLimit += g_depth / -incident.y;
    SceneHit primary = traceSolids(origin, incident, primaryLimit);
    if (primary.id != 0)
    {
        color = shadeSolid(origin + incident * primary.t, incident, primary.id);
        opticalData = float4(0);
    }
    if (g_footDebug != 0 || g_view == 3)
    {
        for (uint i = 0; i < g_footCount; ++i)
            color = footOverlay(color, p, g_feet[i], footColor(i));
        float rootMark = 1 - smoothstep(0.01, 0.025, abs(length(p - g_player) - 0.07));
        color = lerp(color, float3(0.6), rootMark * 0.6);
    }
    if (g_domainDebug != 0)
    {
        float2 d = abs(p - g_domainCenter);
        float edge = g_extent * 0.5 - max(d.x, d.y);
        float line = 1 - smoothstep(0.025, 0.065, abs(edge));
        float inner = 1 - smoothstep(0.025, 0.065, abs(edge - g_edgeWidth));
        color = lerp(color, float3(0.1, 1.2, 1.4), line * 0.9);
        color = lerp(color, float3(1.2, 0.65, 0.04), inner * 0.8);
    }
    g_opticalData[tid.xy] = opticalData;
    if (g_view != 6)
    {
        color = max(color, float3(0));
        color = color / (color + 0.65);
        color = select(color <= 0.0031308, color * 12.92, 1.055 * pow(color, 1.0 / 2.4) - 0.055);
    }
    // Fresnel view intentionally writes raw [0,1] weights (no tone/gamma curve).
    g_output[tid.xy] = float4(saturate(color), 1);
}
