// Package server exposes the RPC surface of the shipment service.
//
// The transport is deliberately abstracted behind small request/response
// structs so the same handlers serve both the generated gRPC stubs and the
// JSON gateway used by internal tooling. Handlers do three things: validate
// the request, call the store, and translate errors into status codes.
package server

import (
	"context"
	"errors"
	"fmt"
	"strings"
	"time"

	"gosvc/internal/breaker"
	"gosvc/internal/logging"
	"gosvc/internal/pool"
	"gosvc/internal/store"
)

// Code mirrors the canonical gRPC status codes this service returns.
type Code int

// The subset of status codes the handlers use.
const (
	CodeOK Code = iota
	CodeInvalidArgument
	CodeNotFound
	CodeAborted
	CodeUnavailable
	CodeInternal
)

// MaxTrackingIDLength bounds the identifier a caller may send.
const MaxTrackingIDLength = 64

// Status is the error envelope every handler returns on failure.
type Status struct {
	Code    Code
	Message string
	Details map[string]string
}

// Error makes Status satisfy the error interface.
func (s *Status) Error() string {
	return fmt.Sprintf("rpc error: code=%d message=%s", s.Code, s.Message)
}

// GetShipmentRequest asks for one shipment by identifier.
type GetShipmentRequest struct {
	ShipmentID string
	TraceID    string
}

// GetShipmentResponse carries the record back to the caller.
type GetShipmentResponse struct {
	ShipmentID string
	OrderID    string
	Carrier    string
	Status     string
	Version    int64
	UpdatedAt  time.Time
}

// AdvanceShipmentRequest moves a shipment to its next status.
type AdvanceShipmentRequest struct {
	ShipmentID  string
	NextStatus  string
	FromVersion int64
	TraceID     string
}

// HealthResponse is what the readiness probe returns.
type HealthResponse struct {
	Healthy bool
	Store   bool
	Breaker breaker.Snapshot
	Pool    pool.Stats
	Uptime  string
}

// Service holds the dependencies every handler needs.
type Service struct {
	store     *store.Store
	pool      *pool.Pool
	breaker   *breaker.Breaker
	log       *logging.Logger
	startedAt time.Time
}

// NewService wires the handlers over their dependencies.
func NewService(st *store.Store, p *pool.Pool, b *breaker.Breaker,
	log *logging.Logger) *Service {
	return &Service{
		store:     st,
		pool:      p,
		breaker:   b,
		log:       log,
		startedAt: time.Now(),
	}
}

// validateShipmentID rejects identifiers that are empty, oversized or contain
// characters that have no business in a key.
func validateShipmentID(id string) *Status {
	trimmed := strings.TrimSpace(id)
	if trimmed == "" {
		return &Status{Code: CodeInvalidArgument, Message: "shipment id is required"}
	}
	if len(trimmed) > MaxTrackingIDLength {
		return &Status{
			Code:    CodeInvalidArgument,
			Message: fmt.Sprintf("shipment id may not exceed %d characters", MaxTrackingIDLength),
		}
	}
	for _, r := range trimmed {
		isAllowed := (r >= 'a' && r <= 'z') || (r >= 'A' && r <= 'Z') ||
			(r >= '0' && r <= '9') || r == '-' || r == '_'
		if !isAllowed {
			return &Status{
				Code:    CodeInvalidArgument,
				Message: "shipment id contains an unsupported character",
				Details: map[string]string{"shipment_id": trimmed},
			}
		}
	}
	return nil
}

// statusFromError maps a domain or infrastructure error onto a status code.
//
// The mapping matters operationally: Unavailable tells the caller's client
// library that a retry is worthwhile, while Aborted signals a lost optimistic
// update the caller should redo against fresh state.
func statusFromError(err error) *Status {
	switch {
	case err == nil:
		return nil
	case errors.Is(err, store.ErrNotFound):
		return &Status{Code: CodeNotFound, Message: "shipment does not exist"}
	case errors.Is(err, store.ErrConflict):
		return &Status{Code: CodeAborted, Message: "shipment was modified concurrently"}
	case errors.Is(err, breaker.ErrOpen):
		return &Status{Code: CodeUnavailable, Message: "downstream store is unavailable"}
	case errors.Is(err, pool.ErrAcquireTimeout):
		return &Status{Code: CodeUnavailable, Message: "no connection available"}
	case errors.Is(err, context.DeadlineExceeded):
		return &Status{Code: CodeUnavailable, Message: "request deadline exceeded"}
	default:
		return &Status{Code: CodeInternal, Message: "unexpected server error"}
	}
}

// GetShipment returns one shipment.
func (s *Service) GetShipment(ctx context.Context, req *GetShipmentRequest) (*GetShipmentResponse, error) {
	if bad := validateShipmentID(req.ShipmentID); bad != nil {
		return nil, bad
	}
	log := s.log.With(logging.Fields{"trace": req.TraceID, "rpc": "GetShipment"})
	record, err := s.store.Get(ctx, req.ShipmentID)
	if err != nil {
		log.Error("lookup failed", err, logging.Fields{"shipment": req.ShipmentID})
		return nil, statusFromError(err)
	}
	log.Debug("lookup served", logging.Fields{"shipment": record.ID})
	return &GetShipmentResponse{
		ShipmentID: record.ID,
		OrderID:    record.OrderID,
		Carrier:    record.Carrier,
		Status:     record.Status,
		Version:    record.Version,
		UpdatedAt:  record.UpdatedAt,
	}, nil
}

// AdvanceShipment moves a shipment forward, honouring optimistic versioning.
func (s *Service) AdvanceShipment(ctx context.Context, req *AdvanceShipmentRequest) (*GetShipmentResponse, error) {
	if bad := validateShipmentID(req.ShipmentID); bad != nil {
		return nil, bad
	}
	if strings.TrimSpace(req.NextStatus) == "" {
		return nil, &Status{Code: CodeInvalidArgument, Message: "next status is required"}
	}
	log := s.log.With(logging.Fields{"trace": req.TraceID, "rpc": "AdvanceShipment"})
	if err := s.store.UpdateStatus(ctx, req.ShipmentID, req.NextStatus, req.FromVersion); err != nil {
		log.Warn("advance rejected", logging.Fields{
			"shipment": req.ShipmentID,
			"reason":   err.Error(),
		})
		return nil, statusFromError(err)
	}
	return s.GetShipment(ctx, &GetShipmentRequest{
		ShipmentID: req.ShipmentID,
		TraceID:    req.TraceID,
	})
}

// Health reports whether the service is ready to take traffic.
//
// A tripped breaker is reported as unhealthy on purpose: routing traffic to
// an instance that can only return errors is worse than removing it from the
// load balancer until the dependency recovers.
func (s *Service) Health(ctx context.Context) *HealthResponse {
	snapshot := s.breaker.Snapshot()
	storeUp := s.store.Ping(ctx) == nil
	return &HealthResponse{
		Healthy: storeUp && snapshot.State != "open",
		Store:   storeUp,
		Breaker: snapshot,
		Pool:    s.pool.Stats(),
		Uptime:  time.Since(s.startedAt).Round(time.Second).String(),
	}
}
