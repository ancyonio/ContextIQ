// Package config loads the service configuration from the process
// environment exactly once at start-up.
//
// Nothing else in gosvc reads os.Getenv directly: a single Load call produces
// an immutable Config that is threaded through the constructors. That makes
// the deployable surface auditable and lets tests build a Config literal
// without touching global state.
package config

import (
	"errors"
	"fmt"
	"os"
	"strconv"
	"strings"
	"time"
)

// Defaults applied when the corresponding variable is unset.
const (
	DefaultListenAddr      = ":9090"
	DefaultPoolSize        = 16
	DefaultDialTimeout     = 3 * time.Second
	DefaultRequestTimeout  = 10 * time.Second
	DefaultConsumerGroup   = "gosvc-workers"
	DefaultCommitInterval  = 5 * time.Second
	DefaultBreakerCooldown = 30 * time.Second
)

// ErrMissing is returned when a mandatory variable has not been supplied.
var ErrMissing = errors.New("required environment variable is not set")

// Config is the fully resolved runtime configuration.
type Config struct {
	Env             string
	ListenAddr      string
	ServiceName     string
	LogLevel        string
	DatabaseDSN     string
	PoolSize        int
	DialTimeout     time.Duration
	RequestTimeout  time.Duration
	BreakerCooldown time.Duration
	FailureRatio    float64
	Brokers         []string
	Topic           string
	ConsumerGroup   string
	CommitInterval  time.Duration
}

// Validate reports the first inconsistency it finds in an assembled Config.
//
// Load already rejects unparsable values, so this catches the combinations
// that are individually fine but nonsensical together.
func (c *Config) Validate() error {
	if c.PoolSize < 1 {
		return fmt.Errorf("pool size must be positive, got %d", c.PoolSize)
	}
	if c.DialTimeout > c.RequestTimeout {
		return fmt.Errorf(
			"dial timeout %s exceeds request timeout %s",
			c.DialTimeout, c.RequestTimeout,
		)
	}
	if c.FailureRatio <= 0 || c.FailureRatio > 1 {
		return fmt.Errorf("failure ratio must be in (0,1], got %v", c.FailureRatio)
	}
	if len(c.Brokers) == 0 {
		return fmt.Errorf("%w: BROKERS", ErrMissing)
	}
	return nil
}

// Redacted returns a map safe to write to logs: the DSN password is removed.
func (c *Config) Redacted() map[string]any {
	return map[string]any{
		"env":          c.Env,
		"service":      c.ServiceName,
		"listen":       c.ListenAddr,
		"database":     maskDSN(c.DatabaseDSN),
		"pool_size":    c.PoolSize,
		"brokers":      len(c.Brokers),
		"topic":        c.Topic,
		"group":        c.ConsumerGroup,
		"log_level":    c.LogLevel,
		"dial_timeout": c.DialTimeout.String(),
	}
}

// maskDSN strips the credential portion of a connection string.
func maskDSN(dsn string) string {
	at := strings.LastIndex(dsn, "@")
	if at < 0 {
		return dsn
	}
	scheme := strings.Index(dsn, "://")
	if scheme < 0 {
		return "***" + dsn[at:]
	}
	return dsn[:scheme+3] + "***" + dsn[at:]
}

// mustGet returns the value of key or an ErrMissing-wrapped error.
func mustGet(key string) (string, error) {
	value := strings.TrimSpace(os.Getenv(key))
	if value == "" {
		return "", fmt.Errorf("%w: %s", ErrMissing, key)
	}
	return value, nil
}

// getInt reads an optional integer variable.
func getInt(key string, fallback int) (int, error) {
	raw := strings.TrimSpace(os.Getenv(key))
	if raw == "" {
		return fallback, nil
	}
	parsed, err := strconv.Atoi(raw)
	if err != nil {
		return 0, fmt.Errorf("%s: %q is not an integer: %w", key, raw, err)
	}
	return parsed, nil
}

// getDuration reads an optional Go duration such as "250ms" or "3s".
func getDuration(key string, fallback time.Duration) (time.Duration, error) {
	raw := strings.TrimSpace(os.Getenv(key))
	if raw == "" {
		return fallback, nil
	}
	parsed, err := time.ParseDuration(raw)
	if err != nil {
		return 0, fmt.Errorf("%s: %q is not a duration: %w", key, raw, err)
	}
	return parsed, nil
}

// getFloat reads an optional float variable.
func getFloat(key string, fallback float64) (float64, error) {
	raw := strings.TrimSpace(os.Getenv(key))
	if raw == "" {
		return fallback, nil
	}
	parsed, err := strconv.ParseFloat(raw, 64)
	if err != nil {
		return 0, fmt.Errorf("%s: %q is not a number: %w", key, raw, err)
	}
	return parsed, nil
}

// getList splits a comma separated variable, trimming and dropping blanks.
func getList(key string) []string {
	raw := strings.TrimSpace(os.Getenv(key))
	if raw == "" {
		return nil
	}
	parts := strings.Split(raw, ",")
	out := make([]string, 0, len(parts))
	for _, part := range parts {
		if trimmed := strings.TrimSpace(part); trimmed != "" {
			out = append(out, trimmed)
		}
	}
	return out
}

// Load reads every supported variable and returns a validated Config.
//
// DATABASE_DSN and TOPIC are mandatory; everything else has a default that is
// documented next to the constant it comes from.
func Load() (*Config, error) {
	dsn, err := mustGet("DATABASE_DSN")
	if err != nil {
		return nil, err
	}
	topic, err := mustGet("TOPIC")
	if err != nil {
		return nil, err
	}
	poolSize, err := getInt("POOL_SIZE", DefaultPoolSize)
	if err != nil {
		return nil, err
	}
	dial, err := getDuration("DIAL_TIMEOUT", DefaultDialTimeout)
	if err != nil {
		return nil, err
	}
	request, err := getDuration("REQUEST_TIMEOUT", DefaultRequestTimeout)
	if err != nil {
		return nil, err
	}
	cooldown, err := getDuration("BREAKER_COOLDOWN", DefaultBreakerCooldown)
	if err != nil {
		return nil, err
	}
	commit, err := getDuration("COMMIT_INTERVAL", DefaultCommitInterval)
	if err != nil {
		return nil, err
	}
	ratio, err := getFloat("FAILURE_RATIO", 0.5)
	if err != nil {
		return nil, err
	}
	cfg := &Config{
		Env:             valueOr(os.Getenv("ENV"), "development"),
		ListenAddr:      valueOr(os.Getenv("LISTEN_ADDR"), DefaultListenAddr),
		ServiceName:     valueOr(os.Getenv("SERVICE_NAME"), "gosvc"),
		LogLevel:        valueOr(os.Getenv("LOG_LEVEL"), "info"),
		DatabaseDSN:     dsn,
		PoolSize:        poolSize,
		DialTimeout:     dial,
		RequestTimeout:  request,
		BreakerCooldown: cooldown,
		FailureRatio:    ratio,
		Brokers:         getList("BROKERS"),
		Topic:           topic,
		ConsumerGroup:   valueOr(os.Getenv("CONSUMER_GROUP"), DefaultConsumerGroup),
		CommitInterval:  commit,
	}
	if err := cfg.Validate(); err != nil {
		return nil, err
	}
	return cfg, nil
}

// valueOr returns fallback when value is empty.
func valueOr(value, fallback string) string {
	if strings.TrimSpace(value) == "" {
		return fallback
	}
	return value
}
