package bwarr

import (
	"testing"
)

// Tests for Rank, Select and CountRange live in this file. See orderstat.go.

func TestBWArr_Rank(t *testing.T) {
	t.Parallel()
	// Cases are added incrementally together with the implementation.
	_ = New(int64Cmp, 0)
}

func TestBWArr_Select(t *testing.T) {
	t.Parallel()
	_ = New(int64Cmp, 0)
}

func TestBWArr_CountRange(t *testing.T) {
	t.Parallel()
	_ = New(int64Cmp, 0)
}
