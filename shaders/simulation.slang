#include "scene.slang"
#include "feet.slang"

struct SimulationParams
{
    uint resolution;
    float extent;
    float2 center;
    float edgeWidth;
    float dt;
    float waveSpeed;
    float damping;
    float strength;
    float entryGain;
    float exitGain;
    float dragGain;
    float2 impulsePosition;
    float impulse;
    uint openWater;
};
uniform SimulationParams g;
uniform FootData g_feet[8];
uniform uint g_footCount;
uniform int2 g_shift;
Texture2D<float4> g_state;
Texture2D<float> g_mask;
RWTexture2D<float4> g_output;
RWTexture2D<float> g_maskOutput;

[shader("compute")]
[numthreads(8, 8, 1)]
void initialize(uint3 tid : SV_DispatchThreadID)
{
    if (any(tid.xy >= g.resolution)) return;
    float2 p = ((float2(tid.xy) + 0.5) / g.resolution - 0.5) * g.extent + g.center;
    g_output[tid.xy] = float4(0);
    g_maskOutput[tid.xy] = g.openWater != 0 || waterMargin(p) > 0 ? 1.0 : 0.0;
}

[shader("compute")]
[numthreads(8, 8, 1)]
void scroll(uint3 tid : SV_DispatchThreadID)
{
    if (any(tid.xy >= g.resolution)) return;
    int2 source = int2(tid.xy) + g_shift;
    float2 p = ((float2(tid.xy) + 0.5) / g.resolution - 0.5) * g.extent + g.center;
    bool wet = g.openWater != 0 || waterMargin(p) > 0;
    bool valid = all(source >= 0) && all(source < int(g.resolution));
    float4 state = float4(0);
    if (valid && wet) state = g_state[source];
    g_output[tid.xy] = state;
    g_maskOutput[tid.xy] = wet ? 1.0 : 0.0;
}

float neighborHeight(int2 cell, float center)
{
    // No normal flux through solid banks: mirrored height at dry neighbours.
    if (any(cell < 0) || any(cell >= int(g.resolution))) return 0;
    return g_mask[cell] > 0.5 ? g_state[cell].x : center;
}

[shader("compute")]
[numthreads(8, 8, 1)]
void step(uint3 tid : SV_DispatchThreadID)
{
    if (any(tid.xy >= g.resolution)) return;
    int2 cell = int2(tid.xy);
    if (g_mask[cell] < 0.5) { g_output[cell] = float4(0); return; }
    float dx = g.extent / g.resolution;
    float2 p = ((float2(cell) + 0.5) / g.resolution - 0.5) * g.extent + g.center;
    float4 state = g_state[cell];
    float laplacian = (neighborHeight(cell + int2(1, 0), state.x)
                    + neighborHeight(cell - int2(1, 0), state.x)
                    + neighborHeight(cell + int2(0, 1), state.x)
                    + neighborHeight(cell - int2(0, 1), state.x) - 4.0 * state.x) / (dx * dx);
    float forcing = 0;
    for (uint i = 0; i < g_footCount; ++i)
        forcing += footForcing(p, g_feet[i], g.dt, dx, g.entryGain, g.exitGain, g.dragGain);
    // Absorb waves near the outer RT edge; real shoreline remains reflective.
    float2 local = p - g.center;
    float edgeDistance = g.extent * 0.5 - max(abs(local.x), abs(local.y));
    float sponge = 8.0 * pow(saturate(1.0 - edgeDistance / g.edgeWidth), 2.0);
    float inputWeight = smoothstep(0.0, g.edgeWidth, edgeDistance);
    float attenuation = exp(-(g.damping + sponge) * g.dt);
    float q = dot(p - g.impulsePosition, p - g.impulsePosition) / 0.36;
    // Mexican-hat impulse has zero integral on an unbounded uniform domain.
    float impulse = g.impulse * (1.0 - q) * exp(-q);
    float velocity = (state.y + g.dt * g.waveSpeed * g.waveSpeed * laplacian
                      + (g.strength * forcing + impulse) * inputWeight) * attenuation;
    float height = (state.x + g.dt * velocity) * exp(-sponge * g.dt);
    // This is a local visual activity mask, not advected physical foam.
    float activity = max(state.z * exp(-1.8 * g.dt), saturate(abs(velocity) * 0.25));
    g_output[cell] = float4(height, velocity, activity, 0);
}
