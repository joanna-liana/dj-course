package shipments

import "time"

type ShipmentStatus string

const (
	Scheduled  ShipmentStatus = "SCHEDULED"
	InProgress ShipmentStatus = "IN_PROGRESS"
	Completed  ShipmentStatus = "COMPLETED"
	Cancelled  ShipmentStatus = "CANCELLED"
)

type Shipment struct {
	ID               int
	OrderID          int
	DriverID         int
	VehicleID        int
	Status           ShipmentStatus
	ScheduledStart   time.Time
	ScheduledEnd     time.Time
	ActualStart      *time.Time
	ActualEnd        *time.Time
	PickupLocation   string
	DeliveryLocation string
	DistanceKm       float64
	Notes            string
}
