struct FootData
{
    float3 previous;
    float3 position;
    float2 forward;
    float2 halfSize;
    float halfHeight;
    float phase;
    float2 target;
    uint swinging;
};

float2 footLocal(float2 delta, float2 forward)
{
    return float2(dot(delta, float2(-forward.y, forward.x)), dot(delta, forward));
}

float immersion(float height, float halfHeight)
{
    // Rest water level is zero. Using it avoids re-triggering an idle foot as
    // simulated waves pass; no GPU readback or animation-water feedback needed.
    return saturate((halfHeight - height) / (2 * halfHeight));
}

float footForcing(float2 p, FootData foot, float dt, float dx, float entryGain, float exitGain, float dragGain)
{
    float wasWet = immersion(foot.previous.y, foot.halfHeight);
    float isWet = immersion(foot.position.y, foot.halfHeight);
    float change = isWet - wasWet;
    float2 center = (foot.previous.xz + foot.position.xz) * 0.5;
    // A slightly enlarged, band-limited footprint remains legible from above.
    float2 size = max(foot.halfSize * 1.35, float2(dx * 1.5));
    float2 local = footLocal(p - center, foot.forward) / size;
    float q = dot(local, local);
    float verticalSpeed = abs(foot.position.y - foot.previous.y) / dt;
    // Integral over substeps is tied to immersion change, not frame count.
    // Center depresses on entry; the annulus rises. Exit has the opposite sign.
    float impact = -change * (change > 0 ? entryGain : exitGain) * min(2.5, 0.4 + verticalSpeed);
    float result = impact * (1 - q) * exp(-q);
    float2 travel = foot.position.xz - foot.previous.xz;
    uint samples = uint(clamp(ceil(length(travel) / (size.x * 0.5)), 1.0, 32.0));
    float drag = 0;
    for (uint i = 0; i < samples; ++i)
    {
        float2 mid = lerp(foot.previous.xz, foot.position.xz, (float(i) + 0.5) / samples);
        float2 offset = footLocal(p - mid, foot.forward) / size;
        float2 movement = footLocal(travel, foot.forward) / size;
        drag += 2 * dot(offset, movement) * exp(-dot(offset, offset)) / samples;
    }
    return result + drag * dragGain * (wasWet + isWet) * 0.5;
}
