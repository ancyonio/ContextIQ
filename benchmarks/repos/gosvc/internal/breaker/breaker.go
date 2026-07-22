// Package breaker implements a rolling-window circuit breaker.
//
// Rather than counting consecutive failures, the breaker keeps a bucketed
// window of outcomes and trips when the failure ratio over that window
// crosses a threshold. That behaves far better under partial degradation,
// where a dependency answers nine calls in ten and a consecutive-failure
// counter would never fire.
package breaker

import (
	"context"
	"errors"
	"sync"
	"time"
)

// State is the position of the breaker in its lifecycle.
type State int

// The three states, in the order they are normally visited.
const (
	StateClosed State = iota
	StateOpen
	StateHalfOpen
)

// Window shape: ten buckets covering one minute.
const (
	BucketCount    = 10
	BucketDuration = 6 * time.Second
	MinimumSamples = 20
	HalfOpenProbes = 3
)

// ErrOpen is returned instead of calling through when the breaker is open.
var ErrOpen = errors.New("circuit is open; call was not attempted")

// String renders a State for logs and the health endpoint.
func (s State) String() string {
	switch s {
	case StateOpen:
		return "open"
	case StateHalfOpen:
		return "half-open"
	default:
		return "closed"
	}
}

// bucket accumulates outcomes for one slice of the rolling window.
type bucket struct {
	startedAt time.Time
	successes int
	failures  int
}

// Snapshot describes the breaker for the health endpoint.
type Snapshot struct {
	State        string
	Failures     int
	Successes    int
	FailureRatio float64
	OpenedAt     time.Time
}

// Breaker guards one downstream dependency.
type Breaker struct {
	mu           sync.Mutex
	buckets      [BucketCount]bucket
	cursor       int
	state        State
	openedAt     time.Time
	probesLeft   int
	cooldown     time.Duration
	failureRatio float64
	clock        func() time.Time
}

// New builds a breaker that trips above failureRatio and stays open for
// cooldown before letting probe traffic through.
func New(failureRatio float64, cooldown time.Duration) *Breaker {
	if failureRatio <= 0 || failureRatio > 1 {
		failureRatio = 0.5
	}
	b := &Breaker{
		state:        StateClosed,
		cooldown:     cooldown,
		failureRatio: failureRatio,
		clock:        time.Now,
	}
	now := b.clock()
	for i := range b.buckets {
		b.buckets[i].startedAt = now
	}
	return b
}

// rotate advances the cursor and zeroes buckets that have aged out.
//
// Called with the lock held. Buckets older than the whole window are reset so
// a burst of failures an hour ago cannot keep the breaker open today.
func (b *Breaker) rotate(now time.Time) {
	current := &b.buckets[b.cursor]
	if now.Sub(current.startedAt) < BucketDuration {
		return
	}
	steps := int(now.Sub(current.startedAt) / BucketDuration)
	if steps > BucketCount {
		steps = BucketCount
	}
	for i := 0; i < steps; i++ {
		b.cursor = (b.cursor + 1) % BucketCount
		b.buckets[b.cursor] = bucket{startedAt: now}
	}
}

// totals sums the window. Called with the lock held.
func (b *Breaker) totals() (successes, failures int) {
	for _, bk := range b.buckets {
		successes += bk.successes
		failures += bk.failures
	}
	return successes, failures
}

// Allow reports whether a call may proceed, moving to half-open when the
// cooldown has elapsed so a small number of probes can test the water.
func (b *Breaker) Allow() bool {
	b.mu.Lock()
	defer b.mu.Unlock()
	now := b.clock()
	b.rotate(now)
	switch b.state {
	case StateOpen:
		if now.Sub(b.openedAt) >= b.cooldown {
			b.state = StateHalfOpen
			b.probesLeft = HalfOpenProbes
			return true
		}
		return false
	case StateHalfOpen:
		if b.probesLeft <= 0 {
			return false
		}
		b.probesLeft--
		return true
	default:
		return true
	}
}

// RecordSuccess books a good outcome, closing the breaker after a probe.
func (b *Breaker) RecordSuccess() {
	b.mu.Lock()
	defer b.mu.Unlock()
	now := b.clock()
	b.rotate(now)
	b.buckets[b.cursor].successes++
	if b.state == StateHalfOpen {
		b.state = StateClosed
		for i := range b.buckets {
			b.buckets[i] = bucket{startedAt: now}
		}
	}
}

// RecordFailure books a bad outcome and trips the breaker when the failure
// ratio over the window crosses the threshold.
//
// The window must hold at least MinimumSamples observations first, otherwise
// two failures on a quiet service would take the dependency out of rotation.
func (b *Breaker) RecordFailure() {
	b.mu.Lock()
	defer b.mu.Unlock()
	now := b.clock()
	b.rotate(now)
	b.buckets[b.cursor].failures++
	if b.state == StateHalfOpen {
		b.state = StateOpen
		b.openedAt = now
		return
	}
	successes, failures := b.totals()
	total := successes + failures
	if total < MinimumSamples {
		return
	}
	if float64(failures)/float64(total) >= b.failureRatio {
		b.state = StateOpen
		b.openedAt = now
	}
}

// Do runs fn through the breaker, recording the outcome.
//
// A context cancellation is deliberately not counted as a dependency failure:
// the caller giving up says nothing about the health of the downstream.
func (b *Breaker) Do(ctx context.Context, fn func(context.Context) error) error {
	if !b.Allow() {
		return ErrOpen
	}
	err := fn(ctx)
	switch {
	case err == nil:
		b.RecordSuccess()
	case errors.Is(err, context.Canceled), errors.Is(err, context.DeadlineExceeded):
		// caller-initiated; leave the window untouched
	default:
		b.RecordFailure()
	}
	return err
}

// Snapshot exposes the breaker's internals for the health endpoint.
func (b *Breaker) Snapshot() Snapshot {
	b.mu.Lock()
	defer b.mu.Unlock()
	successes, failures := b.totals()
	ratio := 0.0
	if total := successes + failures; total > 0 {
		ratio = float64(failures) / float64(total)
	}
	return Snapshot{
		State:        b.state.String(),
		Failures:     failures,
		Successes:    successes,
		FailureRatio: ratio,
		OpenedAt:     b.openedAt,
	}
}
