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
	return s.lowerBound(cmp, val, 0, len(s.elements))
}

// lowerBound returns the first index in [lo, hi) whose element is not less than val.
func (s *segment[T]) lowerBound(cmp CmpFunc[T], val T, lo, hi int) int {
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

// upperBound returns the first index in [lo, hi) whose element is greater than val.
func (s *segment[T]) upperBound(cmp CmpFunc[T], val T, lo, hi int) int {
	for lo < hi {
		mid := int(uint(lo+hi) >> 1)
		if cmp(s.elements[mid], val) <= 0 {
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
// The operation combines binary searches inside the O(log N) segments with
// word-wise popcounts over their candidate windows; it never expands the array.
//
//nolint:revive // Method names follow the public API naming of the package.
func (bwa *BWArr[T]) Select(rank int) (T, bool) {
	var zero T
	if rank < 0 || rank >= bwa.Len() {
		return zero, false
	}

	// Each active segment is sorted on its own, but the segments interleave by
	// value, so a global rank cannot be mapped to a single segment. Instead every
	// active segment keeps an index window [lo, hi) of candidates and the number
	// of live elements below that window (before). A pivot value splits all
	// windows at once and the widest one at least halves, so the search ends.
	ranges := bwa.newSelectRanges()
	for {
		pivot, ok := bwa.widestPivot(ranges)
		if !ok { // Defensive: the requested rank is valid, so candidates must remain.
			return zero, false
		}
		less, notGreater := 0, 0
		for i := range ranges {
			r := &ranges[i]
			seg := &bwa.whiteSegments[r.seg]
			less += r.before + seg.countLiveInRange(r.lo, seg.lowerBound(bwa.cmp, pivot, r.lo, r.hi))
			notGreater += r.before + seg.countLiveInRange(r.lo, seg.upperBound(bwa.cmp, pivot, r.lo, r.hi))
		}

		switch {
		case rank < less: // The answer is strictly below the pivot: drop the upper part.
			for i := range ranges {
				r := &ranges[i]
				r.hi = bwa.whiteSegments[r.seg].lowerBound(bwa.cmp, pivot, r.lo, r.hi)
			}
		case rank >= notGreater: // The answer is strictly above the pivot: drop the lower part.
			for i := range ranges {
				r := &ranges[i]
				seg := &bwa.whiteSegments[r.seg]
				newLo := seg.upperBound(bwa.cmp, pivot, r.lo, r.hi)
				r.before += seg.countLiveInRange(r.lo, newLo)
				r.lo = newLo
			}
		default: // less <= rank < notGreater: a live element equal to the pivot sits at rank.
			return pivot, true
		}
	}
}

// selectRange is one active segment's live candidate window for Select.
type selectRange struct {
	seg    int
	lo, hi int // Half-open index window; elements outside it cannot be the answer.
	before int // Live elements before lo, which are all below the window's values.
}

// newSelectRanges collects the candidate window of every active segment.
func (bwa *BWArr[T]) newSelectRanges() []selectRange {
	ranges := make([]selectRange, 0, len(bwa.whiteSegments))
	for seg := range bwa.activeSegments {
		ranges = append(ranges, selectRange{seg: seg, lo: 0, hi: len(bwa.whiteSegments[seg].elements)})
	}
	return ranges
}

// widestPivot returns the middle value of the widest candidate window and true,
// or the zero value and false when no window has candidates left. That window
// at least halves on either narrow path, which drives the search to termination.
func (bwa *BWArr[T]) widestPivot(ranges []selectRange) (T, bool) {
	var zero T
	widest := -1
	for i := range ranges {
		if ranges[i].lo >= ranges[i].hi {
			continue
		}
		if widest < 0 || ranges[i].hi-ranges[i].lo > ranges[widest].hi-ranges[widest].lo {
			widest = i
		}
	}
	if widest < 0 {
		return zero, false
	}
	r := ranges[widest]
	return bwa.whiteSegments[r.seg].elements[(r.lo+r.hi)>>1], true
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
