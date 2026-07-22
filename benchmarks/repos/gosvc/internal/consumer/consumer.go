// Package consumer runs the event loop that drains the shipment topic.
//
// Delivery is at-least-once, so every handler in this package must be safe to
// run twice on the same message. Offsets are committed on an interval rather
// than per message: committing every message costs a round trip per record,
// and re-processing a few seconds of traffic after a crash is cheap when the
// handlers are idempotent.
package consumer

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"time"

	"gosvc/internal/logging"
	"gosvc/internal/store"
)

// Loop tuning.
const (
	MaxPollRecords   = 200
	RetryBaseDelay   = 200 * time.Millisecond
	RetryMaxDelay    = 10 * time.Second
	MaxDeliveryTries = 5
)

// ErrPoisonMessage marks a record that will never succeed and should be
// routed to the dead-letter topic instead of retried forever.
var ErrPoisonMessage = errors.New("message cannot be processed")

// Message is one record read from the topic.
type Message struct {
	Topic     string
	Partition int32
	Offset    int64
	Key       []byte
	Value     []byte
	Timestamp time.Time
	Tries     int
}

// Reader is the broker client the loop consumes from.
type Reader interface {
	Fetch(ctx context.Context, max int) ([]Message, error)
	Commit(ctx context.Context, offsets map[int32]int64) error
	Close() error
}

// DeadLetterWriter accepts records the loop has given up on.
type DeadLetterWriter interface {
	Publish(ctx context.Context, msg Message, reason string) error
}

// shipmentEvent is the payload carried on the topic.
type shipmentEvent struct {
	ShipmentID string  `json:"shipment_id"`
	OrderID    string  `json:"order_id"`
	Carrier    string  `json:"carrier"`
	Status     string  `json:"status"`
	Weight     float64 `json:"weight"`
	Version    int64   `json:"version"`
}

// Metrics counts what the loop did, for the health endpoint.
type Metrics struct {
	Fetched    int64
	Processed  int64
	Retried    int64
	DeadLetter int64
	Commits    int64
}

// Consumer owns the poll/handle/commit loop for one topic.
type Consumer struct {
	reader     Reader
	store      *store.Store
	dead       DeadLetterWriter
	log        *logging.Logger
	interval   time.Duration
	pending    map[int32]int64
	metrics    Metrics
	lastCommit time.Time
	clock      func() time.Time
}

// New builds a consumer. commitInterval controls how often offsets are
// flushed back to the broker.
func New(reader Reader, st *store.Store, dead DeadLetterWriter,
	log *logging.Logger, commitInterval time.Duration) *Consumer {
	return &Consumer{
		reader:     reader,
		store:      st,
		dead:       dead,
		log:        log,
		interval:   commitInterval,
		pending:    make(map[int32]int64),
		lastCommit: time.Now(),
		clock:      time.Now,
	}
}

// backoffFor returns the delay before redelivering a message, doubling with
// each attempt and capped so a stubborn record cannot stall the partition.
func backoffFor(tries int) time.Duration {
	delay := RetryBaseDelay
	for i := 1; i < tries; i++ {
		delay *= 2
		if delay >= RetryMaxDelay {
			return RetryMaxDelay
		}
	}
	return delay
}

// decode turns a record's bytes into an event, rejecting anything malformed
// as poison so it lands in the dead-letter topic on the first pass.
func decode(msg Message) (shipmentEvent, error) {
	var event shipmentEvent
	if err := json.Unmarshal(msg.Value, &event); err != nil {
		return event, fmt.Errorf("%w: invalid json: %v", ErrPoisonMessage, err)
	}
	if event.ShipmentID == "" || event.Status == "" {
		return event, fmt.Errorf("%w: shipment_id and status are required",
			ErrPoisonMessage)
	}
	return event, nil
}

// Handle applies one record. Safe to call twice for the same offset.
func (c *Consumer) Handle(ctx context.Context, msg Message) error {
	event, err := decode(msg)
	if err != nil {
		return err
	}
	updateErr := c.store.UpdateStatus(ctx, event.ShipmentID, event.Status, event.Version)
	if errors.Is(updateErr, store.ErrConflict) {
		// Another worker already applied this transition; that is the
		// expected outcome of a duplicate delivery, not a failure.
		c.log.Debug("skipping already-applied event", logging.Fields{
			"shipment": event.ShipmentID,
			"offset":   msg.Offset,
		})
		return nil
	}
	if errors.Is(updateErr, store.ErrNotFound) {
		_, insertErr := c.store.UpsertBatch(ctx, []*store.Shipment{{
			ID:      event.ShipmentID,
			OrderID: event.OrderID,
			Carrier: event.Carrier,
			Status:  event.Status,
			Weight:  event.Weight,
		}})
		return insertErr
	}
	return updateErr
}

// RunOnce fetches a batch, handles each record and commits when due.
//
// Returns the number of records fetched so the caller can idle when the topic
// is quiet rather than spinning on an empty poll.
func (c *Consumer) RunOnce(ctx context.Context) (int, error) {
	messages, err := c.reader.Fetch(ctx, MaxPollRecords)
	if err != nil {
		return 0, fmt.Errorf("fetch: %w", err)
	}
	c.metrics.Fetched += int64(len(messages))
	for _, msg := range messages {
		if handleErr := c.handleWithRetries(ctx, msg); handleErr != nil {
			return len(messages), handleErr
		}
		if msg.Offset > c.pending[msg.Partition] {
			c.pending[msg.Partition] = msg.Offset
		}
	}
	if err := c.maybeCommit(ctx, false); err != nil {
		return len(messages), err
	}
	return len(messages), nil
}

// handleWithRetries retries a record with growing delays, then dead-letters it.
func (c *Consumer) handleWithRetries(ctx context.Context, msg Message) error {
	for attempt := 1; attempt <= MaxDeliveryTries; attempt++ {
		err := c.Handle(ctx, msg)
		if err == nil {
			c.metrics.Processed++
			return nil
		}
		if errors.Is(err, ErrPoisonMessage) || attempt == MaxDeliveryTries {
			c.metrics.DeadLetter++
			c.log.Error("routing record to dead letter", err, logging.Fields{
				"partition": msg.Partition,
				"offset":    msg.Offset,
				"attempts":  attempt,
			})
			return c.dead.Publish(ctx, msg, err.Error())
		}
		c.metrics.Retried++
		select {
		case <-time.After(backoffFor(attempt)):
		case <-ctx.Done():
			return ctx.Err()
		}
	}
	return nil
}

// maybeCommit flushes offsets when the interval has elapsed or force is set.
func (c *Consumer) maybeCommit(ctx context.Context, force bool) error {
	if len(c.pending) == 0 {
		return nil
	}
	if !force && c.clock().Sub(c.lastCommit) < c.interval {
		return nil
	}
	if err := c.reader.Commit(ctx, c.pending); err != nil {
		return fmt.Errorf("commit: %w", err)
	}
	c.metrics.Commits++
	c.lastCommit = c.clock()
	c.pending = make(map[int32]int64)
	return nil
}

// Run polls until the context is cancelled, then flushes offsets one last time.
//
// The final commit is what keeps a graceful restart from replaying the whole
// commit interval; a hard kill still replays, which the idempotent handlers
// absorb.
func (c *Consumer) Run(ctx context.Context) error {
	c.log.Info("consumer loop started", logging.Fields{
		"commit_interval": c.interval.String(),
		"max_poll":        MaxPollRecords,
	})
	for {
		select {
		case <-ctx.Done():
			flushCtx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
			defer cancel()
			if err := c.maybeCommit(flushCtx, true); err != nil {
				c.log.Error("final commit failed", err, nil)
			}
			return c.reader.Close()
		default:
		}
		fetched, err := c.RunOnce(ctx)
		if err != nil {
			c.log.Error("poll cycle failed", err, nil)
			time.Sleep(RetryBaseDelay)
			continue
		}
		if fetched == 0 {
			time.Sleep(RetryBaseDelay)
		}
	}
}

// Metrics returns a copy of the loop counters.
func (c *Consumer) Metrics() Metrics {
	return c.metrics
}
