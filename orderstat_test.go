package bwarr

import (
	"math/rand"
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

func TestBWArr_Select(t *testing.T) {
	t.Parallel()
	// Cases are added incrementally together with the implementation.
	_ = New(int64Cmp, 0)
}

func TestBWArr_CountRange(t *testing.T) {
	t.Parallel()
	_ = New(int64Cmp, 0)
}
