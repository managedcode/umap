using System;
using System.Linq;
using Xunit;
using static ManagedCode.Umap.Tests.UnitTestData;

namespace ManagedCode.Umap.Tests
{
    public static class UmapTests
    {
        [Fact]
        public static void StepMethod2D()
        {
            var first = RunUmapProjection(dimensions: 2);
            var second = RunUmapProjection(dimensions: 2);

            AssertNestedFloatArraysEquivalent(first, second);
        }

        [Fact]
        public static void StepMethod3D()
        {
            var first = RunUmapProjection(dimensions: 3);
            var second = RunUmapProjection(dimensions: 3);

            AssertNestedFloatArraysEquivalent(first, second);
        }

        [Fact]
        public static void FindsNearestNeighbors()
        {
            var nNeighbors = 10;
            var umap = new Umap(random: new DeterministicRandomGenerator(42), numberOfNeighbors: nNeighbors);
            var (knnIndices, knnDistances) = umap.NearestNeighbors(TestData, progress => { });

            Assert.Equal(knnDistances.Length, TestData.Length);
            Assert.Equal(knnIndices.Length, TestData.Length);

            Assert.Equal(knnDistances[0].Length, nNeighbors);
            Assert.Equal(knnIndices[0].Length, nNeighbors);
        }

        [Fact]
        public static void FindsABParamsUsingLevenbergMarquardtForDefaultSettings()
        {
            const float expectedA = 1.5769434603113077f;
            const float expectedB = 0.8950608779109733f;

            var (a, b) = Umap.FindABParams(1, 0.1f);
            Assert.True(AreCloseEnough(a, expectedA));
            Assert.True(AreCloseEnough(b, expectedB));

            bool AreCloseEnough(float x, float y) => Math.Abs(x - y) < 0.01;
        }

        private static float[][] RunUmapProjection(int dimensions)
        {
            var umap = new Umap(
                random: new DeterministicRandomGenerator(42),
                dimensions: dimensions);

            var nEpochs = umap.InitializeFit(TestData);
            Assert.Equal(500, nEpochs);

            for (var i = 0; i < nEpochs; i++)
            {
                umap.Step();
            }

            return umap.GetEmbedding();
        }

        private static void AssertNestedFloatArraysEquivalent(float[][] expected, float[][] actual, float tolerance = 1e-5f)
        {
            Assert.Equal(expected.Length, actual.Length);
            foreach (var (expectedRow, actualRow) in expected.Zip(actual, (expectedRow, actualRow) => (expectedRow, actualRow)))
            {
                Assert.Equal(expectedRow.Length, actualRow.Length);
                foreach (var (expectedValue, actualValue) in expectedRow.Zip(actualRow, (expectedValue, actualValue) => (expectedValue, actualValue)))
                {
                    Assert.True(Math.Abs(expectedValue - actualValue) < tolerance);
                }
            }
        }
    }
}
