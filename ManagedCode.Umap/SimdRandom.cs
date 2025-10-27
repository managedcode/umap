using System;
using System.Runtime.CompilerServices;

namespace ManagedCode.Umap;

internal static class SimdRandom
{
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void Zero(Span<int> values) => values.Clear();

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void Uniform(Span<float> values, float range, IProvideRandomValues random)
    {
        random.NextFloats(values);
        Simd.Multiply(values, 2 * range);
        Simd.Add(values, -range);
    }
}
