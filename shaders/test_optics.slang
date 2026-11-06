#include "optics.slang"
RWTexture2D<float4> g_results;

[shader("compute")]
[numthreads(1, 1, 1)]
void probe(uint3 tid : SV_DispatchThreadID)
{
    if (tid.x == 0)
        g_results[uint2(0, 0)] = float4(dielectricFresnel(1, 1, 1.333), dielectricFresnel(0, 1, 1.333),
                                       dielectricFresnel(0, 1, 1), dielectricFresnel(0.3, 1.333, 1));
    if (tid.x == 1)
        g_results[uint2(1, 0)] = float4(refract(float3(0.6, -0.8, 0), float3(0, 1, 0), 1.0 / 1.333), 1);
    if (tid.x == 2)
        g_results[uint2(2, 0)] = float4(ellipsoidHit(float3(0, 2, 0), float3(0, -1, 0), float3(1, 0.5, 1)),
            ellipsoidHit(float3(0, 0, 0), float3(0, -1, 0), float3(1, 0.5, 1)),
            ellipsoidHit(float3(2, 2, 0), float3(0, -1, 0), float3(1, 0.5, 1)),
            ellipsoidHit(float3(1, 2, 0), float3(0, -1, 0), float3(1, 0.5, 1)));
    if (tid.x == 3) g_results[uint2(3, 0)] = float4(beerLambert(float3(0.2, 1, 2), 0.3), 1);
    if (tid.x == 4) g_results[uint2(4, 0)] = float4(beerLambert(float3(0.2, 1, 2), 0.6), 1);
    if (tid.x == 5)
        g_results[uint2(5, 0)] = float4(waterSunBRDF(float3(0, 1, 0), float3(0, 1, 0), float3(0, 1, 0), 0.02, 1.333),
            waterSunBRDF(float3(0, 1, 0), float3(1, 0, 0), float3(0, 1, 0), 0.02, 1.333),
            waterSunBRDF(float3(0, 1, 0), float3(0, 1, 0), float3(0, -1, 0), 0.02, 1.333),
            waterSunBRDF(float3(0, 1, 0), float3(0, 1, 0), float3(0, 1, 0), 0.02, 1));
}
