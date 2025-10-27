﻿using System;
using System.Collections.Generic;
using System.Text;

namespace ManagedCode.Umap
{
    /// <inheritdoc/>
    public class RawVectorArrayUmapDataPoint : IUmapDataPoint
    {

        /// <inheritdoc/>
        public RawVectorArrayUmapDataPoint(float[] data)
        {
            Data = data;
        }


        /// <inheritdoc/>
        public float[] Data { get; }

        /// <summary>
        /// Define an implicit conversion operator from <see cref="float[]"/>.
        /// </summary>
        public static implicit operator RawVectorArrayUmapDataPoint(float[] data) => new RawVectorArrayUmapDataPoint(data);

        /// <summary>
        /// Implicit conversation back to <see cref="float[]"/>.
        /// </summary>
        public static implicit operator float[](RawVectorArrayUmapDataPoint x) => x.Data;
    }
}
