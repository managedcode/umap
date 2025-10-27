using System;

namespace ManagedCode.Umap
{
    internal static class Utils
    {
        /// <summary>
        /// Creates an empty array.
        /// </summary>
        public static float[] Empty(int length) => length == 0 ? Array.Empty<float>() : new float[length];

        /// <summary>
        /// Creates an array filled with index values.
        /// </summary>
        public static float[] Range(int count)
        {
            if (count < 0)
            {
                throw new ArgumentOutOfRangeException(nameof(count));
            }

            if (count == 0)
            {
                return Array.Empty<float>();
            }

            var result = GC.AllocateUninitializedArray<float>(count);
            for (var i = 0; i < count; i++)
            {
                result[i] = i;
            }

            return result;
        }

        /// <summary>
        /// Creates an array filled with a specific value.
        /// </summary>
        public static float[] Filled(int count, float value)
        {
            if (count < 0)
            {
                throw new ArgumentOutOfRangeException(nameof(count));
            }

            if (count == 0)
            {
                return Array.Empty<float>();
            }

            var result = GC.AllocateUninitializedArray<float>(count);
            result.AsSpan().Fill(value);
            return result;
        }

        /// <summary>
        /// Returns the mean of an array.
        /// </summary>
        public static float Mean(float[] input) => Mean((ReadOnlySpan<float>)input);

        public static float Mean(ReadOnlySpan<float> input)
        {
            if (input.Length == 0)
            {
                return float.NaN;
            }

            double sum = 0d;
            foreach (var value in input)
            {
                sum += value;
            }

            return (float)(sum / input.Length);
        }

        /// <summary>
        /// Returns the maximum value of an array.
        /// </summary>
        public static float Max(float[] input) => Max((ReadOnlySpan<float>)input);

        public static float Max(ReadOnlySpan<float> input)
        {
            if (input.Length == 0)
            {
                throw new InvalidOperationException("Sequence contains no elements.");
            }

            var max = input[0];
            for (var i = 1; i < input.Length; i++)
            {
                if (input[i] > max)
                {
                    max = input[i];
                }
            }

            return max;
        }

        /// <summary>
        /// Generate nSamples many integers from 0 to poolSize such that no integer is selected twice.
        /// </summary>
        public static int[] RejectionSample(int nSamples, int poolSize, IProvideRandomValues random)
        {
            if (poolSize <= 0 || nSamples <= 0)
            {
                return Array.Empty<int>();
            }

            if (nSamples > poolSize)
            {
                nSamples = poolSize;
            }

            var result = GC.AllocateUninitializedArray<int>(nSamples);
            if (nSamples == poolSize)
            {
                for (var i = 0; i < nSamples; i++)
                {
                    result[i] = i;
                }

                return result;
            }

            Span<bool> taken = poolSize <= 1024
                ? stackalloc bool[poolSize]
                : new bool[poolSize];
            taken.Clear();

            var index = 0;
            while (index < nSamples)
            {
                var candidate = random.Next(0, poolSize);
                if (taken[candidate])
                {
                    continue;
                }

                taken[candidate] = true;
                result[index++] = candidate;
            }

            return result;
        }
    }
}
