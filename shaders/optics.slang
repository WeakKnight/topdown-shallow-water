// Shared optical/geometry primitives used by rendering and GPU validation.
static const float PI = 3.14159265359;
static const float RAY_EPSILON = 0.0001;

float dielectricFresnel(float cosIncident, float etaIncident, float etaTransmitted)
{
    if (abs(etaIncident - etaTransmitted) < 1e-6) return 0;
    float ci = saturate(cosIncident);
    float eta = etaIncident / etaTransmitted;
    float sinTransmitted2 = eta * eta * max(0.0, 1.0 - ci * ci);
    if (sinTransmitted2 >= 1.0) return 1.0;
    float ct = sqrt(max(0.0, 1.0 - sinTransmitted2));
    float rs = (etaIncident * ci - etaTransmitted * ct) / max(1e-8, etaIncident * ci + etaTransmitted * ct);
    float rp = (etaTransmitted * ci - etaIncident * ct) / max(1e-8, etaTransmitted * ci + etaIncident * ct);
    return saturate(0.5 * (rs * rs + rp * rp));
}

float3 beerLambert(float3 absorption, float distance)
{
    return exp(-max(absorption, float3(0)) * max(0.0, distance));
}

float ellipsoidHit(float3 localOrigin, float3 localDirection, float3 radii)
{
    float3 o = localOrigin / radii, d = localDirection / radii;
    float a = dot(d, d);
    if (a < 1e-12) return -1;
    // Closest approach avoids subtracting two large squared distances when
    // looking at a small foot from a camera tens of meters away.
    float centerT = -dot(o, d) / a;
    float3 closest = o + centerT * d;
    float radial2 = 1.0 - dot(closest, closest);
    if (radial2 < 0) return -1;
    float halfSpan = sqrt(radial2 / a);
    float nearT = centerT - halfSpan, farT = centerT + halfSpan;
    return nearT > RAY_EPSILON ? nearT : farT > RAY_EPSILON ? farT : -1;
}

float smithG1(float cosine, float alpha2)
{
    return 2 * cosine / max(1e-6, cosine + sqrt(alpha2 + (1 - alpha2) * cosine * cosine));
}

float waterSunBRDF(float3 normal, float3 view, float3 light, float roughness, float ior)
{
    float nv = saturate(dot(normal, view)), nl = saturate(dot(normal, light));
    if (nv < 1e-5 || nl < 1e-5) return 0;
    float3 halfway = normalize(light + view);
    float nh = saturate(dot(normal, halfway));
    // Finite-width sun approximation prevents singular subpixel highlights.
    float alpha = max(0.018, roughness * roughness);
    float a2 = alpha * alpha;
    float denominator = nh * nh * (a2 - 1) + 1;
    float distribution = a2 / max(1e-10, PI * denominator * denominator);
    float fresnel = dielectricFresnel(saturate(dot(view, halfway)), 1.0, ior);
    return distribution * smithG1(nv, a2) * smithG1(nl, a2) * fresnel / (4 * nv);
}
