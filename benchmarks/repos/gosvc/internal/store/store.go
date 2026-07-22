// Package store is the persistence layer for shipment records.
//
// Every query runs on a connection borrowed from the pool and guarded by the
// circuit breaker, so a store outage degrades into fast local errors instead
// of a pile of goroutines blocked on a dead socket.
package store

import (
	"context"
	"errors"
	"fmt"
	"strings"
	"time"

	"gosvc/internal/breaker"
	"gosvc/internal/pool"
)

// ErrNotFound is returned when a lookup by key matches nothing.
var ErrNotFound = errors.New("record not found")

// ErrConflict is returned when an optimistic update loses the race.
var ErrConflict = errors.New("record was modified concurrently")

// MaxBatchSize caps how many records one write may carry.
const MaxBatchSize = 500

// Shipment is the aggregate this service owns.
type Shipment struct {
	ID         string
	OrderID    string
	Carrier    string
	Status     string
	Version    int64
	Weight     float64
	UpdatedAt  time.Time
	Attributes map[string]string
}

// Querier is the subset of a database connection the store needs.
//
// The pool hands back a pool.Conn; the adapter below asserts it up to this
// richer interface, which keeps the pool package free of SQL concerns.
type Querier interface {
	QueryRow(ctx context.Context, sql string, args ...any) Row
	Exec(ctx context.Context, sql string, args ...any) (int64, error)
}

// Row is a single result row.
type Row interface {
	Scan(dest ...any) error
}

// Store reads and writes shipments.
type Store struct {
	pool    *pool.Pool
	breaker *breaker.Breaker
	timeout time.Duration
}

// New wires a store over a pool and a breaker.
func New(p *pool.Pool, b *breaker.Breaker, timeout time.Duration) *Store {
	return &Store{pool: p, breaker: b, timeout: timeout}
}

// withConn borrows a connection, runs fn through the breaker and returns the
// connection, marking it broken when the failure looks like a transport error.
func (s *Store) withConn(ctx context.Context, fn func(context.Context, Querier) error) error {
	callCtx, cancel := context.WithTimeout(ctx, s.timeout)
	defer cancel()

	conn, err := s.pool.Acquire(callCtx)
	if err != nil {
		return fmt.Errorf("acquire: %w", err)
	}
	querier, ok := conn.(Querier)
	if !ok {
		s.pool.Release(conn, true)
		return errors.New("pooled connection does not implement Querier")
	}
	runErr := s.breaker.Do(callCtx, func(inner context.Context) error {
		return fn(inner, querier)
	})
	s.pool.Release(conn, isTransportFailure(runErr))
	return runErr
}

// isTransportFailure decides whether a connection should be thrown away
// rather than returned to the idle list.
func isTransportFailure(err error) bool {
	if err == nil {
		return false
	}
	if errors.Is(err, ErrNotFound) || errors.Is(err, ErrConflict) {
		return false
	}
	message := strings.ToLower(err.Error())
	return strings.Contains(message, "connection") ||
		strings.Contains(message, "broken pipe") ||
		strings.Contains(message, "reset by peer")
}

// Get loads one shipment by primary key.
func (s *Store) Get(ctx context.Context, id string) (*Shipment, error) {
	var found Shipment
	err := s.withConn(ctx, func(inner context.Context, q Querier) error {
		row := q.QueryRow(inner,
			`SELECT id, order_id, carrier, status, version, weight, updated_at
			   FROM shipments WHERE id = $1`, id)
		scanErr := row.Scan(&found.ID, &found.OrderID, &found.Carrier,
			&found.Status, &found.Version, &found.Weight, &found.UpdatedAt)
		if scanErr != nil {
			if strings.Contains(scanErr.Error(), "no rows") {
				return fmt.Errorf("%w: shipment %s", ErrNotFound, id)
			}
			return scanErr
		}
		return nil
	})
	if err != nil {
		return nil, err
	}
	return &found, nil
}

// UpdateStatus advances a shipment using optimistic concurrency.
//
// The WHERE clause pins the version we read, so two workers processing the
// same event cannot both apply their change; the loser gets ErrConflict and
// retries against fresh state.
func (s *Store) UpdateStatus(ctx context.Context, id, status string, expectedVersion int64) error {
	return s.withConn(ctx, func(inner context.Context, q Querier) error {
		affected, err := q.Exec(inner,
			`UPDATE shipments SET status = $1, version = version + 1,
			        updated_at = now()
			  WHERE id = $2 AND version = $3`,
			status, id, expectedVersion)
		if err != nil {
			return err
		}
		if affected == 0 {
			return fmt.Errorf("%w: shipment %s at version %d",
				ErrConflict, id, expectedVersion)
		}
		return nil
	})
}

// UpsertBatch writes many shipments in a single statement.
//
// Multi-row VALUES beats a statement per record by roughly an order of
// magnitude on this workload, which matters because the consumer flushes on
// every commit interval.
func (s *Store) UpsertBatch(ctx context.Context, shipments []*Shipment) (int64, error) {
	if len(shipments) == 0 {
		return 0, nil
	}
	if len(shipments) > MaxBatchSize {
		return 0, fmt.Errorf("batch of %d exceeds %d", len(shipments), MaxBatchSize)
	}
	placeholders := make([]string, 0, len(shipments))
	args := make([]any, 0, len(shipments)*5)
	for i, sh := range shipments {
		base := i * 5
		placeholders = append(placeholders, fmt.Sprintf(
			"($%d, $%d, $%d, $%d, $%d)",
			base+1, base+2, base+3, base+4, base+5))
		args = append(args, sh.ID, sh.OrderID, sh.Carrier, sh.Status, sh.Weight)
	}
	statement := `INSERT INTO shipments (id, order_id, carrier, status, weight)
	              VALUES ` + strings.Join(placeholders, ", ") + `
	              ON CONFLICT (id) DO UPDATE SET
	                status = EXCLUDED.status,
	                carrier = EXCLUDED.carrier,
	                version = shipments.version + 1,
	                updated_at = now()`

	var written int64
	err := s.withConn(ctx, func(inner context.Context, q Querier) error {
		affected, execErr := q.Exec(inner, statement, args...)
		written = affected
		return execErr
	})
	return written, err
}

// Ping proves the store is reachable, for the readiness probe.
func (s *Store) Ping(ctx context.Context) error {
	return s.withConn(ctx, func(inner context.Context, q Querier) error {
		var one int
		return q.QueryRow(inner, "SELECT 1").Scan(&one)
	})
}
