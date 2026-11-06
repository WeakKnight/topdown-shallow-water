#include "scene.slang"
#include "feet.slang"

struct SimulationParams
{
    uint resolution;
    float extent;
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
Texture2D<float4> g_state;
Texture2D<float> g_mask;
RWTexture2D<float4> g_output;
RWTexture2D<float> g_maskOutput;

[shader("compute")]
[numthreads(8, 8, 1)]
void initialize(uint3 tid : SV_DispatchThreadID)
{
    if (any(tid.xy >= g.resolution)) return;
    float2 p = ((float2(tid.xy) + 0.5) / g.resolution - 0.5) * g.extent;
    g_output[tid.xy] = float4(0);
    g_maskOutput[tid.xy] = g.openWater != 0 || waterMargin(p) > 0 ? 1.0 : 0.0;
}

float neighborHeight(int2 cell, float center)
{
    // No normal flux through solid banks: mirrored height at dry neighbours.
    if (any(cell < 0) || any(cell >= int(g.resolution))) return center;
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
    float2 p = ((float2(cell) + 0.5) / g.resolution - 0.5) * g.extent;
    float4 state = g_state[cell];
    float laplacian = (neighborHeight(cell + int2(1, 0), state.x)
                    + neighborHeight(cell - int2(1, 0), state.x)
                    + neighborHeight(cell + int2(0, 1), state.x)
                    + neighborHeight(cell - int2(0, 1), state.x) - 4.0 * state.x) / (dx * dx);
    float forcing = 0;
    for (uint i = 0; i < g_footCount; ++i)
        forcing += footForcing(p, g_feet[i], g.dt, dx, g.entryGain, g.exitGain, g.dragGain);
    // Absorb waves near the outer RT edge; real shoreline remains reflective.
    float edgeDistance = min(min(p.x, p.y) + g.extent * 0.5,
                             g.extent * 0.5 - max(p.x, p.y));
    float sponge = 5.0 * pow(saturate(1.0 - edgeDistance / 1.8), 2.0);
    float attenuation = exp(-(g.damping + sponge) * g.dt);
    float q = dot(p - g.impulsePosition, p - g.impulsePosition) / 0.36;
    // Mexican-hat impulse has zero integral on an unbounded uniform domain.
    float impulse = g.impulse * (1.0 - q) * exp(-q);
    float velocity = (state.y + g.dt * g.waveSpeed * g.waveSpeed * laplacian
                      + g.strength * forcing + impulse) * attenuation;
    float height = state.x + g.dt * velocity;
    // This is a local visual activity mask, not advected physical foam.
    float activity = max(state.z * exp(-1.8 * g.dt), saturate(abs(velocity) * 0.25));
    g_output[cell] = float4(height, velocity, activity, 0);
}
