// XZ world space, Y up. Positive signed margin means water.
float waterMargin(float2 p)
{
    float shore = (1.0 - length(p / float2(10.2, 8.5))) * 8.5;
    shore += 0.28 * sin(p.x * 1.1 + p.y * 0.7) + 0.18 * sin(p.y * 1.8);
    float islandA = length((p - float2(-3.4, 1.8)) / float2(1.25, 0.95)) - 1.0;
    float islandB = length((p - float2(4.0, -2.8)) / float2(0.75, 1.15)) - 1.0;
    return min(shore, min(islandA * 0.95, islandB * 0.75));
}

float hash21(float2 p)
{
    return frac(sin(dot(p, float2(127.1, 311.7))) * 43758.5453);
}

float noise2(float2 p)
{
    float2 i = floor(p), f = frac(p);
    f = f * f * (3.0 - 2.0 * f);
    return lerp(lerp(hash21(i), hash21(i + float2(1, 0)), f.x),
                lerp(hash21(i + float2(0, 1)), hash21(i + 1), f.x), f.y);
}
