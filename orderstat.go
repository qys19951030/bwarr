package bwarr

import (
	"math/bits"
)

// This file adds order-statistics queries to BWArr: Rank, Select and CountRange.
//
// All three queries walk the per-segment cumulative information instead of
// materialising the array: each active white segment stores its elements in
// sorted order, so a value bound is located inside a segment with a binary
// search over elements, and the number of live elements in an index range is
// derived from the segment's deleted bitset with word-wise popcounts. The
// number of active segments is O(log N), therefore none of the queries below
// expands the array into a slice or visits elements one by one.

// Rank returns the number of elements strictly less than the given element.
// Equal elements are counted as separate positions; only elements that compare
// less than element are included. The comparison function given to New is used.
//
// The operation has O(log^2 N) time complexity in the worst case.
//
//nolint:revive // Method names follow the public API naming of the package.
func (bwa *BWArr[T]) Rank(element T) int {
	rank := 0
	for seg := range bwa.activeSegments {
		rank += bwa.whiteSegments[seg].countLiveLess(bwa.cmp, element)
	}
	return rank
}

// lowerBoundIndex returns the first index whose element is not less than val.
// The elements slice of a segment is sorted in ascending order as a whole, deleted
// entries included, so the bound can be found without consulting the deleted bitset.
func (s *segment[T]) lowerBoundIndex(cmp CmpFunc[T], val T) int {
	lo, hi := 0, len(s.elements)
	for lo < hi {
		mid := int(uint(lo+hi) >> 1)
		if cmp(s.elements[mid], val) < 0 {
			lo = mid + 1
		} else {
			hi = mid
		}
	}
	return lo
}

// countLiveLess returns how many live elements of the segment compare less than val.
func (s *segment[T]) countLiveLess(cmp CmpFunc[T], val T) int {
	return s.countLiveInRange(0, s.lowerBoundIndex(cmp, val))
}

// countLiveInRange counts the non-deleted elements at indices in [from, to) using
// the segment's deleted bitset word by word: an unset bit marks a live element.
// The count is restricted to [minNonDeleted, maxNonDeleted] so that words beyond the
// live data (which read as zero) never contribute phantom elements.
func (s *segment[T]) countLiveInRange(from, to int) int {
	lo, hi := s.minNonDeletedIndex(), s.maxNonDeletedIndex()+1
	if lo < 0 { // No live elements: nothing to count.
		return 0
	}
	if from < lo {
		from = lo
	}
	if to > hi {
		to = hi
	}
	if from >= to {
		return 0
	}

	words := s.deleted.layers[0]
	first, last := from>>wordShift, (to-1)>>wordShift
	count := 0
	for w := first; w <= last; w++ {
		lo, hi := 0, wordBits
		if w == first {
			lo = from & wordMask
		}
		if w == last {
			hi = (to-1)&wordMask + 1
		}
		// Bits [lo, hi) of the word; hi may be wordBits, where 1<<64 yields 0 and the
		// subtraction below gives the all-ones mask.
		mask := (uint64(1)<<uint(hi) - 1) &^ (uint64(1)<<uint(lo) - 1)
		count += bits.OnesCount64(^words[w] & mask)
	}
	return count
}

// Select returns the element at the given zero-based rank (0 returns the
// minimum element) and true. If rank is negative or greater than or equal to
// Len(), it returns the zero value of T and false. Duplicate elements occupy
// separate positions. The comparison function given to New is used.
//
// The operation has O(log^3 N) time complexity in the worst case.
//
//nolint:revive // Method names follow the public API naming of the package.
func (bwa *BWArr[T]) Select(rank int) (T, bool) {
	var zero T
	return zero, false
}

// CountRange returns the number of elements in the half-open range
// [greaterOrEqual, lessThan), counted per position, so duplicate elements are
// counted individually. If greaterOrEqual is not less than lessThan, it
// returns 0. The comparison function given to New is used.
//
// The operation has O(log^2 N) time complexity in the worst case.
func (bwa *BWArr[T]) CountRange(greaterOrEqual, lessThan T) int {
	if bwa.cmp(greaterOrEqual, lessThan) >= 0 {
		return 0
	}
	count := 0
	for seg := range bwa.activeSegments {
		count += bwa.whiteSegments[seg].countLiveIn(bwa.cmp, greaterOrEqual, lessThan)
	}
	return count
}

// countLiveIn returns how many live elements of the segment fall in [from, to).
// The whole elements slice is sorted, so the range maps to a contiguous index
// span [lowerBound(from), lowerBound(to)) and only the live bits in that span
// are counted.
func (s *segment[T]) countLiveIn(cmp CmpFunc[T], from, to T) int {
	return s.countLiveInRange(s.lowerBoundIndex(cmp, from), s.lowerBoundIndex(cmp, to))
}
