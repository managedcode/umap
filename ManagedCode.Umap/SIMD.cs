using System;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics;

namespace ManagedCode.Umap;

internal static class Simd
{
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float Magnitude(ReadOnlySpan<float> values) => MathF.Sqrt(DotProduct(values, values));

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float Euclidean(ReadOnlySpan<float> lhs, ReadOnlySpan<float> rhs)
    {
        if (lhs.Length != rhs.Length)
        {
            ThrowLengthMismatch();
        }

        ref float left = ref MemoryMarshal.GetReference(lhs);
        ref float right = ref MemoryMarshal.GetReference(rhs);

        int length = lhs.Length;
        int i = 0;
        float sum = 0f;

        if (Vector256.IsHardwareAccelerated && length >= Vector256<float>.Count)
        {
            Vector256<float> acc = Vector256<float>.Zero;
            for (; i <= length - Vector256<float>.Count; i += Vector256<float>.Count)
            {
                var diff = Vector256.LoadUnsafe(ref left, (uint)i) - Vector256.LoadUnsafe(ref right, (uint)i);
                acc += diff * diff;
            }
            sum += Vector256.Sum(acc);
        }

        if (Vector128.IsHardwareAccelerated && i <= length - Vector128<float>.Count)
        {
            Vector128<float> acc = Vector128<float>.Zero;
            for (; i <= length - Vector128<float>.Count; i += Vector128<float>.Count)
            {
                var diff = Vector128.LoadUnsafe(ref left, (uint)i) - Vector128.LoadUnsafe(ref right, (uint)i);
                acc += diff * diff;
            }
            sum += Vector128.Sum(acc);
        }

        for (; i < length; i++)
        {
            float diff = Unsafe.Add(ref left, i) - Unsafe.Add(ref right, i);
            sum += diff * diff;
        }

        return sum;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float DotProduct(ReadOnlySpan<float> lhs, ReadOnlySpan<float> rhs)
    {
        if (lhs.Length != rhs.Length)
        {
            ThrowLengthMismatch();
        }

        ref float left = ref MemoryMarshal.GetReference(lhs);
        ref float right = ref MemoryMarshal.GetReference(rhs);

        int length = lhs.Length;
        int i = 0;
        float sum = 0f;

        if (Vector256.IsHardwareAccelerated && length >= Vector256<float>.Count)
        {
            Vector256<float> acc = Vector256<float>.Zero;
            for (; i <= length - Vector256<float>.Count; i += Vector256<float>.Count)
            {
                acc += Vector256.LoadUnsafe(ref left, (uint)i) * Vector256.LoadUnsafe(ref right, (uint)i);
            }
            sum += Vector256.Sum(acc);
        }

        if (Vector128.IsHardwareAccelerated && i <= length - Vector128<float>.Count)
        {
            Vector128<float> acc = Vector128<float>.Zero;
            for (; i <= length - Vector128<float>.Count; i += Vector128<float>.Count)
            {
                acc += Vector128.LoadUnsafe(ref left, (uint)i) * Vector128.LoadUnsafe(ref right, (uint)i);
            }
            sum += Vector128.Sum(acc);
        }

        for (; i < length; i++)
        {
            sum += Unsafe.Add(ref left, i) * Unsafe.Add(ref right, i);
        }

        return sum;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void Add(Span<float> values, float scalar)
    {
        if (values.Length == 0)
        {
            return;
        }

        ref float start = ref MemoryMarshal.GetReference(values);
        int length = values.Length;
        int i = 0;

        if (Vector256.IsHardwareAccelerated && length >= Vector256<float>.Count)
        {
            Vector256<float> scalarVec = Vector256.Create(scalar);
            for (; i <= length - Vector256<float>.Count; i += Vector256<float>.Count)
            {
                var current = Vector256.LoadUnsafe(ref start, (uint)i);
                (current + scalarVec).StoreUnsafe(ref start, (uint)i);
            }
        }

        if (Vector128.IsHardwareAccelerated && i <= length - Vector128<float>.Count)
        {
            Vector128<float> scalarVec = Vector128.Create(scalar);
            for (; i <= length - Vector128<float>.Count; i += Vector128<float>.Count)
            {
                var current = Vector128.LoadUnsafe(ref start, (uint)i);
                (current + scalarVec).StoreUnsafe(ref start, (uint)i);
            }
        }

        for (; i < length; i++)
        {
            Unsafe.Add(ref start, i) += scalar;
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void Multiply(Span<float> values, float scalar)
    {
        if (values.Length == 0)
        {
            return;
        }

        ref float start = ref MemoryMarshal.GetReference(values);
        int length = values.Length;
        int i = 0;

        if (Vector256.IsHardwareAccelerated && length >= Vector256<float>.Count)
        {
            Vector256<float> scalarVec = Vector256.Create(scalar);
            for (; i <= length - Vector256<float>.Count; i += Vector256<float>.Count)
            {
                var current = Vector256.LoadUnsafe(ref start, (uint)i);
                (current * scalarVec).StoreUnsafe(ref start, (uint)i);
            }
        }

        if (Vector128.IsHardwareAccelerated && i <= length - Vector128<float>.Count)
        {
            Vector128<float> scalarVec = Vector128.Create(scalar);
            for (; i <= length - Vector128<float>.Count; i += Vector128<float>.Count)
            {
                var current = Vector128.LoadUnsafe(ref start, (uint)i);
                (current * scalarVec).StoreUnsafe(ref start, (uint)i);
            }
        }

        for (; i < length; i++)
        {
            Unsafe.Add(ref start, i) *= scalar;
        }
    }

    [MethodImpl(MethodImplOptions.NoInlining)]
    private static void ThrowLengthMismatch() => throw new ArgumentException("Vectors must have the same length.");
}
