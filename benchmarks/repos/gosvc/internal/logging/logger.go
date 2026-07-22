// Package logging provides the structured, leveled logger the rest of the
// service uses.
//
// Records are emitted as one JSON object per line so the log shipper can
// index fields without a grok pattern. A logger carries immutable context
// (service name, request id, partition) that is merged into every record it
// writes, which is what makes correlating a request across goroutines cheap.
package logging

import (
	"encoding/json"
	"fmt"
	"io"
	"os"
	"sort"
	"strings"
	"sync"
	"time"
)

// Level is a syslog-style severity.
type Level int

// The severities the service emits, in increasing order of urgency.
const (
	LevelDebug Level = iota
	LevelInfo
	LevelWarn
	LevelError
)

// TimestampKey is the field name every record uses for its clock reading.
const TimestampKey = "ts"

// Fields is a bag of structured attributes attached to a record.
type Fields map[string]any

// String renders a Level the way it appears in the emitted JSON.
func (l Level) String() string {
	switch l {
	case LevelDebug:
		return "debug"
	case LevelInfo:
		return "info"
	case LevelWarn:
		return "warn"
	case LevelError:
		return "error"
	default:
		return "unknown"
	}
}

// ParseLevel maps a configuration string onto a Level, defaulting to info.
func ParseLevel(name string) Level {
	switch strings.ToLower(strings.TrimSpace(name)) {
	case "debug", "trace":
		return LevelDebug
	case "warn", "warning":
		return LevelWarn
	case "error", "fatal":
		return LevelError
	default:
		return LevelInfo
	}
}

// Logger writes JSON records at or above its minimum level.
//
// The mutex guards the writer only; the field map is copied on every With so
// a derived logger can be handed to another goroutine safely.
type Logger struct {
	mu       sync.Mutex
	out      io.Writer
	minLevel Level
	base     Fields
	clock    func() time.Time
}

// New builds a logger writing to out.
func New(out io.Writer, minLevel Level, base Fields) *Logger {
	if out == nil {
		out = os.Stdout
	}
	fields := Fields{}
	for k, v := range base {
		fields[k] = v
	}
	return &Logger{out: out, minLevel: minLevel, base: fields, clock: time.Now}
}

// With returns a copy of the logger carrying additional permanent fields.
//
// Deriving rather than mutating is what lets a request handler add a trace id
// without every other goroutine suddenly logging it too.
func (l *Logger) With(extra Fields) *Logger {
	merged := Fields{}
	for k, v := range l.base {
		merged[k] = v
	}
	for k, v := range extra {
		merged[k] = v
	}
	return &Logger{out: l.out, minLevel: l.minLevel, base: merged, clock: l.clock}
}

// Enabled reports whether a record at level would be written.
func (l *Logger) Enabled(level Level) bool {
	return level >= l.minLevel
}

// log renders and writes one record, dropping it when the level is too low.
func (l *Logger) log(level Level, msg string, extra Fields) {
	if !l.Enabled(level) {
		return
	}
	record := make(map[string]any, len(l.base)+len(extra)+3)
	for k, v := range l.base {
		record[k] = v
	}
	for k, v := range extra {
		record[k] = v
	}
	record[TimestampKey] = l.clock().UTC().Format(time.RFC3339Nano)
	record["level"] = level.String()
	record["msg"] = msg

	encoded, err := json.Marshal(record)
	if err != nil {
		encoded = []byte(fmt.Sprintf(
			`{"%s":%q,"level":"error","msg":"log encoding failed"}`,
			TimestampKey, l.clock().UTC().Format(time.RFC3339Nano),
		))
	}
	l.mu.Lock()
	defer l.mu.Unlock()
	_, _ = l.out.Write(append(encoded, '\n'))
}

// Debug writes a record at debug level.
func (l *Logger) Debug(msg string, extra Fields) { l.log(LevelDebug, msg, extra) }

// Info writes a record at info level.
func (l *Logger) Info(msg string, extra Fields) { l.log(LevelInfo, msg, extra) }

// Warn writes a record at warn level.
func (l *Logger) Warn(msg string, extra Fields) { l.log(LevelWarn, msg, extra) }

// Error writes a record at error level, attaching err when it is non-nil.
func (l *Logger) Error(msg string, err error, extra Fields) {
	fields := Fields{}
	for k, v := range extra {
		fields[k] = v
	}
	if err != nil {
		fields["error"] = err.Error()
	}
	l.log(LevelError, msg, fields)
}

// FieldNames lists the permanent field names, sorted, for diagnostics.
func (l *Logger) FieldNames() []string {
	names := make([]string, 0, len(l.base))
	for k := range l.base {
		names = append(names, k)
	}
	sort.Strings(names)
	return names
}
