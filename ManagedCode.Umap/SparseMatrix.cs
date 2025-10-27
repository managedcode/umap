using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;

namespace ManagedCode.Umap
{
    internal sealed class SparseMatrix
    {
        private readonly Dictionary<RowCol, float> _entries;
        public SparseMatrix(IEnumerable<int> rows, IEnumerable<int> cols, IEnumerable<float> values, (int rows, int cols) dims)
            : this(Materialize(rows, cols, values), dims)
        {
        }

        internal SparseMatrix(ReadOnlySpan<int> rows, ReadOnlySpan<int> cols, ReadOnlySpan<float> values, (int rows, int cols) dims)
        {
            if ((rows.Length != cols.Length) || (rows.Length != values.Length))
            {
                throw new ArgumentException($"The input lists {nameof(rows)}, {nameof(cols)} and {nameof(values)} must all have the same number of elements");
            }

            Dims = dims;
            _entries = new Dictionary<RowCol, float>(rows.Length);
            for (var i = 0; i < rows.Length; i++)
            {
                var row = rows[i];
                var col = cols[i];
                CheckDims(row, col);
                ref float slot = ref CollectionsMarshal.GetValueRefOrAddDefault(_entries, new RowCol(row, col), out _);
                slot = values[i];
            }
        }

        private SparseMatrix((int[] rows, int[] cols, float[] values) data, (int rows, int cols) dims)
            : this(data.rows.AsSpan(), data.cols.AsSpan(), data.values.AsSpan(), dims)
        {
        }

        private SparseMatrix(Dictionary<RowCol, float> entries, (int, int) dims)
        {
            Dims = dims;
            _entries = entries;
        }

        private static (int[] rows, int[] cols, float[] values) Materialize(IEnumerable<int> rows, IEnumerable<int> cols, IEnumerable<float> values)
        {
            var rowsArray = rows as int[] ?? rows.ToArray();
            var colsArray = cols as int[] ?? cols.ToArray();
            var valuesArray = values as float[] ?? values.ToArray();
            if ((rowsArray.Length != valuesArray.Length) || (colsArray.Length != valuesArray.Length))
            {
                throw new ArgumentException($"The input lists {nameof(rows)}, {nameof(cols)} and {nameof(values)} must all have the same number of elements");
            }

            return (rowsArray, colsArray, valuesArray);
        }

        public (int rows, int cols) Dims { get; }

        public void Set(int row, int col, float value)
        {
            CheckDims(row, col);
            ref float slot = ref CollectionsMarshal.GetValueRefOrAddDefault(_entries, new RowCol(row, col), out _);
            slot = value;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public float Get(int row, int col, float defaultValue = 0)
        {
            CheckDims(row, col);
            return _entries.TryGetValue(new RowCol(row, col), out var v) ? v : defaultValue;
        }

        public IEnumerable<(int row, int col, float value)> GetAll()
        {
            foreach (var kv in _entries)
            {
                yield return (kv.Key.Row, kv.Key.Col, kv.Value);
            }
        }

        public IEnumerable<int> GetRows() => _entries.Keys.Select(k => k.Row);
        public IEnumerable<int> GetCols() => _entries.Keys.Select(k => k.Col);
        public IEnumerable<float> GetValues() => _entries.Values;
        
        public void ForEach(Action<float, int, int> fn)
        {
            foreach (var kv in _entries)
            {
                fn(kv.Value, kv.Key.Row, kv.Key.Col);
            }
        }

        public SparseMatrix Map(Func<float, float> fn) => Map((value, row, col) => fn(value));

        public SparseMatrix Map(Func<float, int, int, float> fn)
        {
            var newEntries = new Dictionary<RowCol, float>(_entries.Count);
            foreach (var kv in _entries)
            {
                ref float slot = ref CollectionsMarshal.GetValueRefOrAddDefault(newEntries, kv.Key, out _);
                slot = fn(kv.Value, kv.Key.Row, kv.Key.Col);
            }
            return new SparseMatrix(newEntries, Dims);
        }

        public float[][] ToArray()
        {
            var output = Enumerable.Range(0, Dims.rows).Select(_ => new float[Dims.cols]).ToArray();
            foreach (var kv in _entries)
            {
                output[kv.Key.Row][kv.Key.Col] = kv.Value;
            }

            return output;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private void CheckDims(int row, int col)
        {
#if DEBUG
            if ((row >= Dims.rows) || (col >= Dims.cols))
            {
                throw new Exception("array index out of bounds");
            }
#endif
        }

        public SparseMatrix Transpose()
        {
            var dims = (Dims.cols, Dims.rows);
            var entries = new Dictionary<RowCol, float>(_entries.Count);
            foreach (var entry in _entries)
            {
                entries[new RowCol(entry.Key.Col, entry.Key.Row)] = entry.Value;
            }

            return new SparseMatrix(entries, dims);
        }

        /// <summary>
        /// Element-wise multiplication of two matrices
        /// </summary>
        public SparseMatrix PairwiseMultiply(SparseMatrix other)
        {
            var newEntries = new Dictionary<RowCol, float>(Math.Max(_entries.Count, other._entries.Count));
            foreach (var kv in _entries)
            {
                if (!other._entries.TryGetValue(kv.Key, out var v))
                {
                    continue;
                }

                ref float slot = ref CollectionsMarshal.GetValueRefOrAddDefault(newEntries, kv.Key, out _);
                slot = kv.Value * v;
            }
            return new SparseMatrix(newEntries, Dims);
        }

        /// <summary>
        /// Element-wise addition of two matrices
        /// </summary>
        public SparseMatrix Add(SparseMatrix other) => ElementWiseWith(other, (x, y) => x + y);

        /// <summary>
        /// Element-wise subtraction of two matrices
        /// </summary>
        public SparseMatrix Subtract(SparseMatrix other) => ElementWiseWith(other, (x, y) => x - y);

        /// <summary>
        /// Scalar multiplication of a matrix
        /// </summary>
        public SparseMatrix MultiplyScalar(float scalar) => Map((value, row, cols) => value * scalar);

        /// <summary>
        /// Helper function for element-wise operations
        /// </summary>
        private SparseMatrix ElementWiseWith(SparseMatrix other, Func<float, float, float> op)
        {
            var newEntries = new Dictionary<RowCol, float>(_entries.Count + other._entries.Count);
            foreach (var kv in _entries)
            {
                var key = kv.Key;
                other._entries.TryGetValue(key, out var otherValue);
                ref float slot = ref CollectionsMarshal.GetValueRefOrAddDefault(newEntries, key, out _);
                slot = op(kv.Value, otherValue);
            }

            foreach (var kv in other._entries)
            {
                ref float slot = ref CollectionsMarshal.GetValueRefOrAddDefault(newEntries, kv.Key, out var exists);
                if (exists)
                {
                    continue;
                }

                slot = op(0f, kv.Value);
            }
            return new SparseMatrix(newEntries, Dims);
        }

        /// <summary>
        /// Helper function for getting data, indices, and indptr arrays from a sparse matrix to follow csr matrix conventions. Super inefficient (and kind of defeats the purpose of this convention)
        /// but a lot of the ported python tree search logic depends on this data format.
        /// </summary>
        public (int[] indices, float[] values, int[] indptr) GetCSR()
        {
            var entries = new List<(float value, int row, int col)>();
            ForEach((value, row, col) => entries.Add((value, row, col)));
            entries.Sort((a, b) =>
            {
                if (a.row == b.row)
                {
                    return a.col - b.col;
                }

                return a.row - b.row;
            });

            var indices = new List<int>();
            var values = new List<float>();
            var indptr = new List<int>();
            var currentRow = -1;
            for (var i = 0; i < entries.Count; i++)
            {
                var (value, row, col) = entries[i];
                if (row != currentRow)
                {
                    currentRow = row;
                    indptr.Add(i);
                }
                indices.Add(col);
                values.Add(value);
            }
            return (indices.ToArray(), values.ToArray(), indptr.ToArray());
        }

        private struct RowCol : IEquatable<RowCol>
        {
            public RowCol(int row, int col)
            {
                Row = row;
                Col = col;
            }

            public int Row { get; }
            public int Col { get; }

            // 2019-06-24 DWR: Structs get default Equals and GetHashCode implementations but they can be slow - having these versions makes the code run much quicker
            // and it seems a good practice to throw in IEquatable<RowCol> to avoid boxing when Equals is called
            public bool Equals(RowCol other) => (other.Row == Row) && (other.Col == Col);
            public override bool Equals(object? obj) => (obj is RowCol rc) && rc.Equals(this);
            public override int GetHashCode() // Courtesy of https://stackoverflow.com/a/263416/3813189
            {
                unchecked // Overflow is fine, just wrap
                {
                    int hash = 17;
                    hash = hash * 23 + Row;
                    hash = hash * 23 + Col;
                    return hash;
                }
            }
        }
    }
}
