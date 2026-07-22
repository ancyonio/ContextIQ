// Package pool implements a bounded, health-checked connection pool for the
// downstream store.
//
// The pool hands out at most Size connections; callers block (with context
// cancellation honoured) when every connection is busy. Idle connections are
// recycled after MaxIdleTime because a connection that has been sitting
// behind a NAT gateway for an hour is usually dead without knowing it.
package pool

import (
	"context"
	"errors"
	"fmt"
	"sync"
	"time"
)

// Tunables that are not worth putting in the config file.
const (
	MaxIdleTime      = 5 * time.Minute
	HealthCheckEvery = 30 * time.Second
	acquireSpinGuard = 250 * time.Millisecond
)

// ErrPoolClosed is returned once Close has been called.
var ErrPoolClosed = errors.New("pool is closed")

// ErrAcquireTimeout is returned when the caller's context expired while
// waiting for a free connection.
var ErrAcquireTimeout = errors.New("timed out waiting for a connection")

// Conn is the minimal behaviour the pool needs from a connection.
type Conn interface {
	Ping(ctx context.Context) error
	Close() error
	ID() string
}

// Dialer opens a new connection to the downstream store.
type Dialer func(ctx context.Context) (Conn, error)

// entry pairs a connection with the bookkeeping the pool keeps about it.
type entry struct {
	conn     Conn
	lastUsed time.Time
	uses     int64
}

// Stats is a point-in-time snapshot used by the health endpoint.
type Stats struct {
	Size      int
	Idle      int
	InUse     int
	Dialed    int64
	Discarded int64
	Waited    int64
}

// Pool is a fixed-capacity connection pool.
type Pool struct {
	mu        sync.Mutex
	idle      []*entry
	dialer    Dialer
	size      int
	inUse     int
	closed    bool
	dialed    int64
	discarded int64
	waited    int64
	slots     chan struct{}
	clock     func() time.Time
}

// New builds a pool of the given size. A size below one is treated as one.
func New(size int, dialer Dialer) *Pool {
	if size < 1 {
		size = 1
	}
	slots := make(chan struct{}, size)
	for i := 0; i < size; i++ {
		slots <- struct{}{}
	}
	return &Pool{
		idle:   make([]*entry, 0, size),
		dialer: dialer,
		size:   size,
		slots:  slots,
		clock:  time.Now,
	}
}

// Acquire returns a connection, dialing a fresh one when none are idle.
//
// A slot is taken from the semaphore first, so the pool can never exceed its
// configured size no matter how many goroutines call in at once. If the
// caller's context is cancelled while waiting, the slot is left untouched and
// ErrAcquireTimeout is returned.
func (p *Pool) Acquire(ctx context.Context) (Conn, error) {
	select {
	case <-p.slots:
	case <-ctx.Done():
		p.mu.Lock()
		p.waited++
		p.mu.Unlock()
		return nil, fmt.Errorf("%w: %v", ErrAcquireTimeout, ctx.Err())
	}

	p.mu.Lock()
	if p.closed {
		p.mu.Unlock()
		p.slots <- struct{}{}
		return nil, ErrPoolClosed
	}
	for len(p.idle) > 0 {
		last := p.idle[len(p.idle)-1]
		p.idle = p.idle[:len(p.idle)-1]
		if p.clock().Sub(last.lastUsed) < MaxIdleTime {
			p.inUse++
			p.mu.Unlock()
			return last.conn, nil
		}
		p.discarded++
		_ = last.conn.Close()
	}
	p.inUse++
	p.dialed++
	p.mu.Unlock()

	conn, err := p.dialer(ctx)
	if err != nil {
		p.mu.Lock()
		p.inUse--
		p.mu.Unlock()
		p.slots <- struct{}{}
		return nil, fmt.Errorf("dial failed: %w", err)
	}
	return conn, nil
}

// Release puts a connection back. Passing broken=true discards it instead.
func (p *Pool) Release(conn Conn, broken bool) {
	if conn == nil {
		return
	}
	p.mu.Lock()
	p.inUse--
	if p.closed || broken {
		p.discarded++
		p.mu.Unlock()
		_ = conn.Close()
		p.slots <- struct{}{}
		return
	}
	p.idle = append(p.idle, &entry{conn: conn, lastUsed: p.clock(), uses: 1})
	p.mu.Unlock()
	p.slots <- struct{}{}
}

// Reap closes idle connections that have outlived MaxIdleTime.
//
// Returns how many were discarded so the caller can log a non-zero sweep.
func (p *Pool) Reap() int {
	cutoff := p.clock().Add(-MaxIdleTime)
	p.mu.Lock()
	kept := p.idle[:0]
	var stale []Conn
	for _, e := range p.idle {
		if e.lastUsed.Before(cutoff) {
			stale = append(stale, e.conn)
			p.discarded++
			continue
		}
		kept = append(kept, e)
	}
	p.idle = kept
	p.mu.Unlock()
	for _, conn := range stale {
		_ = conn.Close()
	}
	return len(stale)
}

// HealthCheck pings every idle connection and drops the ones that fail.
func (p *Pool) HealthCheck(ctx context.Context) error {
	p.mu.Lock()
	snapshot := make([]*entry, len(p.idle))
	copy(snapshot, p.idle)
	p.mu.Unlock()

	var firstErr error
	for _, e := range snapshot {
		checkCtx, cancel := context.WithTimeout(ctx, acquireSpinGuard)
		err := e.conn.Ping(checkCtx)
		cancel()
		if err == nil {
			continue
		}
		if firstErr == nil {
			firstErr = err
		}
		p.drop(e)
	}
	return firstErr
}

// drop removes one entry from the idle list and closes it.
func (p *Pool) drop(target *entry) {
	p.mu.Lock()
	kept := p.idle[:0]
	for _, e := range p.idle {
		if e != target {
			kept = append(kept, e)
		}
	}
	p.idle = kept
	p.discarded++
	p.mu.Unlock()
	_ = target.conn.Close()
}

// Stats returns a snapshot of the pool counters.
func (p *Pool) Stats() Stats {
	p.mu.Lock()
	defer p.mu.Unlock()
	return Stats{
		Size:      p.size,
		Idle:      len(p.idle),
		InUse:     p.inUse,
		Dialed:    p.dialed,
		Discarded: p.discarded,
		Waited:    p.waited,
	}
}

// Close discards every idle connection and refuses further acquisitions.
func (p *Pool) Close() error {
	p.mu.Lock()
	if p.closed {
		p.mu.Unlock()
		return ErrPoolClosed
	}
	p.closed = true
	stale := p.idle
	p.idle = nil
	p.mu.Unlock()
	for _, e := range stale {
		_ = e.conn.Close()
	}
	return nil
}
