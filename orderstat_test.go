package bwarr

import (
	"math/rand"
	"sync"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// Tests for Rank, Select and CountRange live in this file. See orderstat.go.

// rankByScan returns the number of elements strictly less than elem using the
// public iteration API, as an independent reference for Rank.
func rankByScan(bwa *BWArr[int64], elem int64) int {
	n := 0
	bwa.Ascend(func(v int64) bool {
		if int64Cmp(v, elem) < 0 {
			n++
		}
		return true
	})
	return n
}

// fillInt64 inserts the given values and returns the array.
func fillInt64(values []int64) *BWArr[int64] {
	bwa := New(int64Cmp, len(values))
	for _, v := range values {
		bwa.Insert(v)
	}
	return bwa
}

func TestBWArr_Rank(t *testing.T) {
	t.Parallel()

	t.Run("empty array", func(t *testing.T) {
		t.Parallel()
		bwa := New(int64Cmp, 0)
		assert.Equal(t, 0, bwa.Rank(42))
	})

	t.Run("single element", func(t *testing.T) {
		t.Parallel()
		bwa := fillInt64([]int64{10})
		for _, elem := range []int64{-5, 9, 10, 11, 1000} {
			assert.Equalf(t, rankByScan(bwa, elem), bwa.Rank(elem), "Rank(%d)", elem)
		}
	})

	t.Run("all positions of a dense range", func(t *testing.T) {
		t.Parallel()
		const n = 300
		values := make([]int64, 0, n)
		for i := range n {
			values = append(values, int64(i))
		}
		bwa := fillInt64(values)
		for elem := int64(-2); elem <= n+2; elem++ {
			assert.Equalf(t, rankByScan(bwa, elem), bwa.Rank(elem), "Rank(%d)", elem)
		}
	})

	t.Run("duplicates counted per position", func(t *testing.T) {
		t.Parallel()
		bwa := fillInt64([]int64{5, 5, 5, 5, 1, 9, 9})
		assert.Equal(t, 0, bwa.Rank(1))
		assert.Equal(t, 1, bwa.Rank(5), "only strictly-smaller elements are counted")
		assert.Equal(t, 5, bwa.Rank(9))
		assert.Equal(t, 7, bwa.Rank(10))
		assert.Equal(t, 0, bwa.Rank(-100))
	})

	t.Run("with deleted elements", func(t *testing.T) {
		t.Parallel()
		bwa := fillInt64([]int64{1, 2, 3, 4, 5, 6, 7, 8, 9, 10})
		for _, v := range []int64{2, 5, 9, 10} {
			_, found := bwa.Delete(v)
			require.True(t, found)
		}
		for elem := int64(0); elem <= 12; elem++ {
			assert.Equalf(t, rankByScan(bwa, elem), bwa.Rank(elem), "Rank(%d) after deletions", elem)
		}
	})

	t.Run("random array with duplicates and deletions", func(t *testing.T) {
		t.Parallel()
		r := rand.New(rand.NewSource(2024))
		const n = 4000
		bwa := New(int64Cmp, n)
		values := make([]int64, 0, n)
		for range n {
			v := int64(r.Intn(500))
			bwa.Insert(v)
			values = append(values, v)
		}
		for range n / 4 {
			bwa.Delete(int64(r.Intn(500)))
		}
		for elem := int64(-3); elem <= 503; elem++ {
			assert.Equalf(t, rankByScan(bwa, elem), bwa.Rank(elem), "Rank(%d)", elem)
		}
	})

	t.Run("result matches Len and Ascend", func(t *testing.T) {
		t.Parallel()
		r := rand.New(rand.NewSource(99))
		bwa := New(int64Cmp, 0)
		for range 1000 {
			bwa.Insert(int64(r.Intn(50)))
		}
		for range 300 {
			bwa.Delete(int64(r.Intn(50)))
		}
		below := 0
		total := 0
		bwa.Ascend(func(v int64) bool {
			if v < 25 {
				below++
			}
			total++
			return true
		})
		assert.Equal(t, bwa.Len(), total)
		assert.Equal(t, below, bwa.Rank(25))
	})
}

// ascendInt64 returns all live elements in ascending order as a reference.
func ascendInt64(bwa *BWArr[int64]) []int64 {
	out := make([]int64, 0, bwa.Len())
	bwa.Ascend(func(v int64) bool {
		out = append(out, v)
		return true
	})
	return out
}

func TestBWArr_Select(t *testing.T) {
	t.Parallel()

	t.Run("empty array", func(t *testing.T) {
		t.Parallel()
		bwa := New(int64Cmp, 0)
		for _, rank := range []int{-1, 0, 1, 100} {
			v, ok := bwa.Select(rank)
			assert.Falsef(t, ok, "Select(%d) on empty array", rank)
			assert.Equal(t, int64(0), v)
		}
	})

	t.Run("single element", func(t *testing.T) {
		t.Parallel()
		bwa := fillInt64([]int64{42})
		v, ok := bwa.Select(0)
		assert.True(t, ok)
		assert.Equal(t, int64(42), v)
		_, ok = bwa.Select(1)
		assert.False(t, ok, "rank == Len() is out of range")
	})

	t.Run("every rank matches Ascend", func(t *testing.T) {
		t.Parallel()
		const n = 500
		values := make([]int64, 0, n)
		for i := range n {
			values = append(values, int64(i))
		}
		bwa := fillInt64(values)
		want := ascendInt64(bwa)
		require.Len(t, want, n)
		for rank := range n {
			got, ok := bwa.Select(rank)
			assert.Truef(t, ok, "Select(%d)", rank)
			assert.Equalf(t, want[rank], got, "Select(%d)", rank)
		}
	})

	t.Run("rank zero is the minimum and last rank the maximum", func(t *testing.T) {
		t.Parallel()
		bwa := fillInt64([]int64{7, 3, 9, 1, 5})
		first, ok := bwa.Select(0)
		require.True(t, ok)
		minElem, found := bwa.Min()
		require.True(t, found)
		assert.Equal(t, minElem, first)

		last, ok := bwa.Select(bwa.Len() - 1)
		require.True(t, ok)
		maxElem, found := bwa.Max()
		require.True(t, found)
		assert.Equal(t, maxElem, last)
	})

	t.Run("duplicates occupy separate positions", func(t *testing.T) {
		t.Parallel()
		bwa := fillInt64([]int64{5, 5, 5, 1, 9})
		for rank := 1; rank <= 3; rank++ {
			v, ok := bwa.Select(rank)
			assert.Truef(t, ok, "Select(%d)", rank)
			assert.Equalf(t, int64(5), v, "Select(%d) must return the duplicated value", rank)
		}
	})

	t.Run("out of range ranks", func(t *testing.T) {
		t.Parallel()
		bwa := fillInt64([]int64{1, 2, 3})
		for _, rank := range []int{-1, -100, 3, 4, 1000} {
			v, ok := bwa.Select(rank)
			assert.Falsef(t, ok, "Select(%d)", rank)
			assert.Equalf(t, int64(0), v, "Select(%d) must return the zero value", rank)
		}
	})

	t.Run("with deleted elements", func(t *testing.T) {
		t.Parallel()
		bwa := fillInt64([]int64{1, 2, 3, 4, 5, 6, 7, 8, 9, 10})
		for _, v := range []int64{1, 3, 4, 8, 10} {
			_, found := bwa.Delete(v)
			require.True(t, found)
		}
		want := ascendInt64(bwa)
		require.Len(t, want, bwa.Len())
		for rank := range want {
			got, ok := bwa.Select(rank)
			assert.Truef(t, ok, "Select(%d)", rank)
			assert.Equalf(t, want[rank], got, "Select(%d)", rank)
		}
		_, ok := bwa.Select(len(want))
		assert.False(t, ok, "rank == Len() is out of range")
	})

	t.Run("random array with duplicates and deletions", func(t *testing.T) {
		t.Parallel()
		r := rand.New(rand.NewSource(31337))
		const n = 3000
		bwa := New(int64Cmp, n)
		for range n {
			bwa.Insert(int64(r.Intn(250)))
		}
		for range n / 3 {
			bwa.Delete(int64(r.Intn(250)))
		}
		want := ascendInt64(bwa)
		require.Equal(t, bwa.Len(), len(want))
		for rank := range want {
			got, ok := bwa.Select(rank)
			assert.Truef(t, ok, "Select(%d)", rank)
			assert.Equalf(t, want[rank], got, "Select(%d)", rank)
		}
	})
}

// countRangeByScan returns the number of elements in [from, to) using AscendRange.
func countRangeByScan(bwa *BWArr[int64], from, to int64) int {
	n := 0
	bwa.AscendRange(from, to, func(int64) bool {
		n++
		return true
	})
	return n
}

func TestBWArr_CountRange(t *testing.T) {
	t.Parallel()

	t.Run("empty array", func(t *testing.T) {
		t.Parallel()
		bwa := New(int64Cmp, 0)
		assert.Equal(t, 0, bwa.CountRange(-10, 10))
		assert.Equal(t, 0, bwa.CountRange(0, 0))
	})

	t.Run("single element", func(t *testing.T) {
		t.Parallel()
		bwa := fillInt64([]int64{10})
		for _, r := range [][2]int64{{0, 10}, {10, 11}, {9, 11}, {-5, 5}, {11, 20}} {
			assert.Equalf(t, countRangeByScan(bwa, r[0], r[1]), bwa.CountRange(r[0], r[1]), "[%d,%d)", r[0], r[1])
		}
	})

	t.Run("dense range all boundaries", func(t *testing.T) {
		t.Parallel()
		const n = 300
		values := make([]int64, 0, n)
		for i := range n {
			values = append(values, int64(i))
		}
		bwa := fillInt64(values)
		for from := int64(-2); from <= n+1; from++ {
			for to := int64(-2); to <= n+1; to++ {
				assert.Equalf(t, countRangeByScan(bwa, from, to), bwa.CountRange(from, to), "[%d,%d)", from, to)
			}
		}
	})

	t.Run("duplicates counted per position", func(t *testing.T) {
		t.Parallel()
		bwa := fillInt64([]int64{1, 2, 2, 2, 2, 5, 8, 8})
		assert.Equal(t, 5, bwa.CountRange(2, 8), "four 2s plus 5")
		assert.Equal(t, 4, bwa.CountRange(2, 5), "only the four 2s")
		assert.Equal(t, 3, bwa.CountRange(5, 9), "5 and both 8s")
	})

	t.Run("lower bound not less than upper bound", func(t *testing.T) {
		t.Parallel()
		bwa := fillInt64([]int64{1, 2, 3, 4, 5})
		assert.Equal(t, 0, bwa.CountRange(3, 3))
		assert.Equal(t, 0, bwa.CountRange(4, 2))
		assert.Equal(t, 0, bwa.CountRange(0, -10))
	})

	t.Run("range outside the array", func(t *testing.T) {
		t.Parallel()
		bwa := fillInt64([]int64{10, 20, 30})
		assert.Equal(t, 0, bwa.CountRange(-100, 0))
		assert.Equal(t, 0, bwa.CountRange(31, 100))
		assert.Equal(t, 0, bwa.CountRange(100, 200))
		assert.Equal(t, 2, bwa.CountRange(20, 31), "20 and 30")
	})

	t.Run("with deleted elements", func(t *testing.T) {
		t.Parallel()
		bwa := fillInt64([]int64{1, 2, 3, 4, 5, 6, 7, 8, 9, 10})
		for _, v := range []int64{1, 4, 5, 10} {
			_, found := bwa.Delete(v)
			require.True(t, found)
		}
		for from := int64(-1); from <= 11; from++ {
			for to := int64(-1); to <= 11; to++ {
				assert.Equalf(t, countRangeByScan(bwa, from, to), bwa.CountRange(from, to), "[%d,%d)", from, to)
			}
		}
	})

	t.Run("random array with duplicates and deletions", func(t *testing.T) {
		t.Parallel()
		r := rand.New(rand.NewSource(4242))
		const n = 4000
		bwa := New(int64Cmp, n)
		for range n {
			bwa.Insert(int64(r.Intn(400)))
		}
		for range n / 3 {
			bwa.Delete(int64(r.Intn(400)))
		}
		for range 500 {
			from := int64(r.Intn(420)) - 10
			to := int64(r.Intn(420)) - 10
			assert.Equalf(t, countRangeByScan(bwa, from, to), bwa.CountRange(from, to), "[%d,%d)", from, to)
		}
	})
}

// assertOrderStatsConsistent checks all three queries against Ascend/AscendRange
// for every element position of bwa.
func assertOrderStatsConsistent(t *testing.T, bwa *BWArr[int64]) {
	t.Helper()
	want := ascendInt64(bwa)
	require.Equal(t, bwa.Len(), len(want), "Len must match the Ascend count")

	firstIndexOf := make(map[int64]int, len(want))
	for i, elem := range want {
		if _, seen := firstIndexOf[elem]; !seen {
			firstIndexOf[elem] = i
		}
	}

	for rank, elem := range want {
		got, ok := bwa.Select(rank)
		assert.Truef(t, ok, "Select(%d) must succeed", rank)
		assert.Equalf(t, elem, got, "Select(%d)", rank)
		assert.Equalf(t, firstIndexOf[elem], bwa.Rank(elem), "Rank(%d) must count strictly smaller elements", elem)
		assert.Equalf(t, countRangeByScan(bwa, elem, elem+1), bwa.CountRange(elem, elem+1), "CountRange(%d, %d)", elem, elem+1)
	}

	_, ok := bwa.Select(len(want))
	assert.False(t, ok, "Select(Len()) must be out of range")

	if len(want) == 0 {
		assert.Equal(t, 0, bwa.Rank(0))
		assert.Equal(t, 0, bwa.CountRange(0, 1))
		return
	}

	aboveMax := want[len(want)-1] + 1
	assert.Equal(t, len(want), bwa.Rank(aboveMax), "Rank above the maximum must be Len")
	assert.Equal(t, 0, bwa.CountRange(aboveMax, aboveMax+100))
	assert.Equal(t, 0, bwa.Rank(want[0]-1), "Rank below the minimum must be 0")
}

func TestBWArr_OrderStatsEmptyAndBounds(t *testing.T) {
	t.Parallel()

	t.Run("empty array never panics", func(t *testing.T) {
		t.Parallel()
		bwa := New(int64Cmp, 0)
		assert.Equal(t, 0, bwa.Rank(0))
		assert.Equal(t, 0, bwa.Rank(-1))
		v, ok := bwa.Select(0)
		assert.False(t, ok)
		assert.Equal(t, int64(0), v)
		assert.Equal(t, 0, bwa.CountRange(0, 1))
		assert.Equal(t, 0, bwa.CountRange(-1, 1))
		assert.Equal(t, 0, bwa.CountRange(1, 0))
	})

	t.Run("cleared array behaves as empty", func(t *testing.T) {
		t.Parallel()
		bwa := fillInt64([]int64{3, 1, 2})
		bwa.Clear(false)
		assert.Equal(t, 0, bwa.Rank(2))
		_, ok := bwa.Select(0)
		assert.False(t, ok)
		assert.Equal(t, 0, bwa.CountRange(0, 10))
	})

	t.Run("single-element boundary ranks", func(t *testing.T) {
		t.Parallel()
		bwa := fillInt64([]int64{5})
		assert.Equal(t, 0, bwa.Rank(5))
		assert.Equal(t, 1, bwa.Rank(6))
		v, ok := bwa.Select(0)
		assert.True(t, ok)
		assert.Equal(t, int64(5), v)
	})
}

func TestBWArr_OrderStatsAfterMutations(t *testing.T) {
	t.Parallel()

	t.Run("after insert", func(t *testing.T) {
		t.Parallel()
		bwa := New(int64Cmp, 0)
		for _, v := range []int64{10, 30, 20} {
			bwa.Insert(v)
			assertOrderStatsConsistent(t, bwa)
		}
	})

	t.Run("after replace or insert", func(t *testing.T) {
		t.Parallel()
		bwa := fillInt64([]int64{1, 3, 5, 7})
		_, found := bwa.ReplaceOrInsert(3)
		require.True(t, found)
		assertOrderStatsConsistent(t, bwa)
		_, found = bwa.ReplaceOrInsert(4)
		require.False(t, found)
		assertOrderStatsConsistent(t, bwa)
	})

	t.Run("after delete", func(t *testing.T) {
		t.Parallel()
		bwa := fillInt64([]int64{1, 2, 3, 4, 5, 6})
		for _, v := range []int64{3, 6, 1} {
			_, found := bwa.Delete(v)
			require.True(t, found)
			assertOrderStatsConsistent(t, bwa)
		}
	})

	t.Run("after delete min and max", func(t *testing.T) {
		t.Parallel()
		bwa := fillInt64([]int64{5, 1, 4, 2, 3})
		for range 5 {
			_, found := bwa.DeleteMin()
			require.True(t, found)
			assertOrderStatsConsistent(t, bwa)
		}
		bwa = fillInt64([]int64{5, 1, 4, 2, 3})
		for range 5 {
			_, found := bwa.DeleteMax()
			require.True(t, found)
			assertOrderStatsConsistent(t, bwa)
		}
	})

	t.Run("after clear and refill", func(t *testing.T) {
		t.Parallel()
		bwa := fillInt64([]int64{1, 2, 3, 4, 5})
		bwa.Clear(true)
		assertOrderStatsConsistent(t, bwa)
		for _, v := range []int64{9, 7, 8} {
			bwa.Insert(v)
		}
		assertOrderStatsConsistent(t, bwa)
	})

	t.Run("clone of stats matches and diverges independently", func(t *testing.T) {
		t.Parallel()
		bwa := fillInt64([]int64{1, 2, 2, 3, 8, 5})
		bwa.Delete(8)
		clone := bwa.Clone()
		assertOrderStatsConsistent(t, bwa)
		assertOrderStatsConsistent(t, clone)

		// Mutating the clone must not change the original.
		clone.Delete(2)
		clone.Insert(100)
		assertOrderStatsConsistent(t, clone)
		assertOrderStatsConsistent(t, bwa)
		assert.Equal(t, 5, bwa.Len(), "original must keep its length")
		assert.Equal(t, 5, clone.Len(), "clone lost one and gained one")

		// Mutating the original must not change the clone either.
		before := ascendInt64(clone)
		bwa.Insert(42)
		bwa.Delete(1)
		assert.Equal(t, before, ascendInt64(clone), "clone must be independent of the original")
	})
}

func TestBWArr_OrderStatsConcurrentReads(t *testing.T) {
	t.Parallel()

	// The read path takes no locks and must be stable for concurrent readers:
	// all of them see the same segments, so all must return the same answers.
	bwa := New(int64Cmp, 0)
	const n = 2000
	r := rand.New(rand.NewSource(8080))
	for range n {
		bwa.Insert(int64(r.Intn(300)))
	}
	for range n / 4 {
		bwa.Delete(int64(r.Intn(300)))
	}

	want := ascendInt64(bwa)
	require.NotEmpty(t, want)

	// One reference answer per rank, snapshotted before the goroutines start.
	refRanks := make([]int, len(want))
	refCounts := make([]int, len(want))
	for rank, elem := range want {
		refRanks[rank] = bwa.Rank(elem)
		refCounts[rank] = bwa.CountRange(elem, elem+1)
	}

	const readers = 8
	var wg sync.WaitGroup
	wg.Add(readers)
	for range readers {
		go func() {
			defer wg.Done()
			for rank, elem := range want {
				got, ok := bwa.Select(rank)
				if !ok || got != elem {
					t.Errorf("Select(%d) = (%d, %v), want (%d, true)", rank, got, ok, elem)
					return
				}
				if gotRank := bwa.Rank(elem); gotRank != refRanks[rank] {
					t.Errorf("Rank(%d) = %d, want %d", elem, gotRank, refRanks[rank])
					return
				}
				if gotCount := bwa.CountRange(elem, elem+1); gotCount != refCounts[rank] {
					t.Errorf("CountRange(%d, %d) = %d, want %d", elem, elem+1, gotCount, refCounts[rank])
					return
				}
			}
		}()
	}
	wg.Wait()
}
