// Command gosvc is the shipment service entry point.
//
// It builds the dependency graph in one place, starts the consumer loop and
// the RPC surface, and coordinates a shutdown that drains both without losing
// in-flight work.
package main

import (
	"context"
	"errors"
	"os"
	"os/signal"
	"sync"
	"syscall"
	"time"

	"gosvc/internal/breaker"
	"gosvc/internal/config"
	"gosvc/internal/consumer"
	"gosvc/internal/logging"
	"gosvc/internal/pool"
	"gosvc/internal/server"
	"gosvc/internal/store"
)

// ShutdownGrace bounds how long we wait for the loops to finish.
const ShutdownGrace = 20 * time.Second

// dependencies is everything main assembles and later tears down.
type dependencies struct {
	cfg      *config.Config
	log      *logging.Logger
	pool     *pool.Pool
	breaker  *breaker.Breaker
	store    *store.Store
	service  *server.Service
	consumer *consumer.Consumer
}

// build assembles the object graph from a validated configuration.
func build(cfg *config.Config, dial pool.Dialer, reader consumer.Reader,
	dead consumer.DeadLetterWriter) *dependencies {
	log := logging.New(os.Stdout, logging.ParseLevel(cfg.LogLevel), logging.Fields{
		"service": cfg.ServiceName,
		"env":     cfg.Env,
	})
	connections := pool.New(cfg.PoolSize, dial)
	guard := breaker.New(cfg.FailureRatio, cfg.BreakerCooldown)
	records := store.New(connections, guard, cfg.RequestTimeout)
	return &dependencies{
		cfg:      cfg,
		log:      log,
		pool:     connections,
		breaker:  guard,
		store:    records,
		service:  server.NewService(records, connections, guard, log),
		consumer: consumer.New(reader, records, dead, log, cfg.CommitInterval),
	}
}

// startMaintenance runs the periodic pool sweep until the context is done.
//
// Idle connections are reaped and the survivors pinged, so a connection that
// died quietly is discovered by the sweeper rather than by a customer request.
func startMaintenance(ctx context.Context, deps *dependencies, wg *sync.WaitGroup) {
	wg.Add(1)
	go func() {
		defer wg.Done()
		ticker := time.NewTicker(pool.HealthCheckEvery)
		defer ticker.Stop()
		for {
			select {
			case <-ctx.Done():
				return
			case <-ticker.C:
				if reaped := deps.pool.Reap(); reaped > 0 {
					deps.log.Info("reaped idle connections", logging.Fields{
						"count": reaped,
					})
				}
				if err := deps.pool.HealthCheck(ctx); err != nil {
					deps.log.Warn("pool health check found a bad connection",
						logging.Fields{"error": err.Error()})
				}
			}
		}
	}()
}

// startConsumer runs the event loop in its own goroutine.
func startConsumer(ctx context.Context, deps *dependencies, wg *sync.WaitGroup) {
	wg.Add(1)
	go func() {
		defer wg.Done()
		if err := deps.consumer.Run(ctx); err != nil && !errors.Is(err, context.Canceled) {
			deps.log.Error("consumer loop exited", err, nil)
		}
	}()
}

// awaitSignal blocks until the process is asked to stop.
func awaitSignal(ctx context.Context) {
	signals := make(chan os.Signal, 1)
	signal.Notify(signals, syscall.SIGINT, syscall.SIGTERM)
	defer signal.Stop(signals)
	select {
	case <-signals:
	case <-ctx.Done():
	}
}

// run is main's testable body: it returns an exit code instead of exiting.
func run(dial pool.Dialer, reader consumer.Reader, dead consumer.DeadLetterWriter) int {
	cfg, err := config.Load()
	if err != nil {
		logging.New(os.Stderr, logging.LevelError, nil).
			Error("configuration is invalid", err, nil)
		return 78
	}
	deps := build(cfg, dial, reader, dead)
	deps.log.Info("starting", logging.Fields(cfg.Redacted()))

	ctx, cancel := context.WithCancel(context.Background())
	var wg sync.WaitGroup
	startMaintenance(ctx, deps, &wg)
	startConsumer(ctx, deps, &wg)

	awaitSignal(ctx)
	deps.log.Info("shutdown requested", logging.Fields{
		"grace": ShutdownGrace.String(),
	})
	cancel()

	finished := make(chan struct{})
	go func() {
		wg.Wait()
		close(finished)
	}()
	select {
	case <-finished:
		deps.log.Info("loops drained cleanly", logging.Fields{
			"consumer": deps.consumer.Metrics().Processed,
		})
	case <-time.After(ShutdownGrace):
		deps.log.Warn("grace period elapsed with work still running", nil)
	}
	if err := deps.pool.Close(); err != nil {
		deps.log.Warn("pool close reported an error", logging.Fields{
			"error": err.Error(),
		})
	}
	return 0
}

func main() {
	os.Exit(run(defaultDialer, defaultReader{}, defaultDeadLetter{}))
}

// defaultDialer opens a connection to the configured store. The real build
// links this against the database driver; the signature is the contract.
var defaultDialer pool.Dialer = func(ctx context.Context) (pool.Conn, error) {
	return nil, errors.New("no driver linked into this build")
}

// defaultReader is the broker client used when none is injected.
type defaultReader struct{}

// Fetch returns no records; the real build swaps in the broker client.
func (defaultReader) Fetch(ctx context.Context, max int) ([]consumer.Message, error) {
	<-ctx.Done()
	return nil, ctx.Err()
}

// Commit acknowledges offsets back to the broker.
func (defaultReader) Commit(ctx context.Context, offsets map[int32]int64) error {
	return nil
}

// Close releases the broker connection.
func (defaultReader) Close() error { return nil }

// defaultDeadLetter drops poison records with a log line.
type defaultDeadLetter struct{}

// Publish records that a message was abandoned.
func (defaultDeadLetter) Publish(ctx context.Context, msg consumer.Message, reason string) error {
	return nil
}
