package shipments

import (
	"fmt"
	"math"
	"math/rand"
	"strconv"
	"strings"
	"time"

	"tms-data-generator/generator/drivers"
	"tms-data-generator/generator/transportation_orders"
	"tms-data-generator/generator/vehicles"
)

type timeRange struct {
	start time.Time
	end   time.Time
}

func GenerateShipments(
	orders []transportation_orders.TransportationOrder,
	driversList []drivers.Driver,
	vehiclesList []vehicles.Vehicle,
) []Shipment {
	shipments := make([]Shipment, 0, int(float64(len(orders))*0.7))
	shipmentID := 1

	assignedDrivers := make(map[int][]timeRange)
	assignedVehicles := make(map[int][]timeRange)

	activeDrivers := make([]drivers.Driver, 0)
	for _, d := range driversList {
		if d.Status == drivers.Active || d.Status == drivers.OnRoute {
			activeDrivers = append(activeDrivers, d)
		}
	}

	for _, order := range orders {
		if shouldAssignShipment(order.Status) && rand.Float64() < 0.70 {
			startTime, endTime := calculateTimeSlot(order)

			driver, vehicle, found := selectAvailableResources(
				activeDrivers,
				vehiclesList,
				startTime,
				endTime,
				assignedDrivers,
				assignedVehicles,
			)

			if found {
				shipment := Shipment{
					ID:               shipmentID,
					OrderID:          order.ID,
					DriverID:         driver.ID,
					VehicleID:        vehicle.ID,
					Status:           mapOrderStatusToShipmentStatus(order.Status),
					ScheduledStart:   startTime,
					ScheduledEnd:     endTime,
					ActualStart:      getActualStart(order.Status, startTime),
					ActualEnd:        getActualEnd(order.Status, startTime, endTime),
					PickupLocation:   generatePickupLocation(),
					DeliveryLocation: order.ShippingAddress,
					DistanceKm:       calculateDistance(),
					Notes:            generateNotes(),
				}

				shipments = append(shipments, shipment)
				shipmentID++

				assignedDrivers[driver.ID] = append(assignedDrivers[driver.ID], timeRange{startTime, endTime})
				assignedVehicles[vehicle.ID] = append(assignedVehicles[vehicle.ID], timeRange{startTime, endTime})
			}
		}
	}

	return shipments
}

func shouldAssignShipment(status transportation_orders.OrderStatus) bool {
	return status == transportation_orders.OrderDelivered ||
		status == transportation_orders.OrderInTransit ||
		status == transportation_orders.OrderReadyForPickup
}

func mapOrderStatusToShipmentStatus(orderStatus transportation_orders.OrderStatus) ShipmentStatus {
	switch orderStatus {
	case transportation_orders.OrderDelivered:
		return Completed
	case transportation_orders.OrderInTransit:
		if rand.Float64() < 0.80 {
			return InProgress
		}
		return Scheduled
	case transportation_orders.OrderReadyForPickup:
		return Scheduled
	default:
		return Scheduled
	}
}

func calculateTimeSlot(order transportation_orders.TransportationOrder) (time.Time, time.Time) {
	durationHours := 2 + rand.Intn(7)
	offsetHours := rand.Intn(24)

	switch order.Status {
	case transportation_orders.OrderDelivered:
		startTime := order.OrderDate.Add(time.Duration(offsetHours) * time.Hour)
		endTime := startTime.Add(time.Duration(durationHours) * time.Hour)
		return startTime, endTime

	case transportation_orders.OrderInTransit:
		startTime := order.OrderDate.Add(time.Duration(offsetHours) * time.Hour)
		endTime := startTime.Add(time.Duration(durationHours) * time.Hour)
		return startTime, endTime

	case transportation_orders.OrderReadyForPickup:
		now := time.Now()
		offsetHours := 1 + rand.Intn(48)
		startTime := now.Add(time.Duration(offsetHours) * time.Hour)
		endTime := startTime.Add(time.Duration(durationHours) * time.Hour)
		return startTime, endTime

	default:
		startTime := order.OrderDate.Add(time.Duration(offsetHours) * time.Hour)
		endTime := startTime.Add(time.Duration(durationHours) * time.Hour)
		return startTime, endTime
	}
}

func getActualStart(status transportation_orders.OrderStatus, scheduledStart time.Time) *time.Time {
	if status == transportation_orders.OrderDelivered {
		actualStart := scheduledStart.Add(time.Duration(rand.Intn(120)) * time.Minute)
		return &actualStart
	}
	if status == transportation_orders.OrderInTransit {
		actualStart := scheduledStart.Add(time.Duration(rand.Intn(120)) * time.Minute)
		return &actualStart
	}
	return nil
}

func getActualEnd(status transportation_orders.OrderStatus, scheduledStart time.Time, scheduledEnd time.Time) *time.Time {
	if status == transportation_orders.OrderDelivered {
		actualEnd := scheduledEnd.Add(time.Duration(rand.Intn(120)) * time.Minute)
		return &actualEnd
	}
	return nil
}

func selectAvailableResources(
	activeDrivers []drivers.Driver,
	vehiclesList []vehicles.Vehicle,
	startTime time.Time,
	endTime time.Time,
	assignedDrivers map[int][]timeRange,
	assignedVehicles map[int][]timeRange,
) (drivers.Driver, vehicles.Vehicle, bool) {
	shuffledDrivers := make([]drivers.Driver, len(activeDrivers))
	copy(shuffledDrivers, activeDrivers)
	rand.Shuffle(len(shuffledDrivers), func(i, j int) {
		shuffledDrivers[i], shuffledDrivers[j] = shuffledDrivers[j], shuffledDrivers[i]
	})

	shuffledVehicles := make([]vehicles.Vehicle, len(vehiclesList))
	copy(shuffledVehicles, vehiclesList)
	rand.Shuffle(len(shuffledVehicles), func(i, j int) {
		shuffledVehicles[i], shuffledVehicles[j] = shuffledVehicles[j], shuffledVehicles[i]
	})

	for _, driver := range shuffledDrivers {
		if !hasConflict(assignedDrivers[driver.ID], startTime, endTime) {
			for _, vehicle := range shuffledVehicles {
				if !hasConflict(assignedVehicles[vehicle.ID], startTime, endTime) {
					return driver, vehicle, true
				}
			}
		}
	}

	var emptyDriver drivers.Driver
	var emptyVehicle vehicles.Vehicle
	return emptyDriver, emptyVehicle, false
}

func hasConflict(assignments []timeRange, newStart time.Time, newEnd time.Time) bool {
	for _, assignment := range assignments {
		if checkTimeOverlap(assignment.start, assignment.end, newStart, newEnd) {
			return true
		}
	}
	return false
}

func checkTimeOverlap(start1 time.Time, end1 time.Time, start2 time.Time, end2 time.Time) bool {
	return start1.Before(end2) && start2.Before(end1)
}

func generatePickupLocation() string {
	locations := []string{
		"Central Warehouse A",
		"Distribution Hub North",
		"Logistics Center East",
		"Regional Hub South",
		"West Coast Distribution",
		"Central Valley Warehouse",
		"Northeast Logistics Park",
		"Midwest Transport Center",
	}
	return locations[rand.Intn(len(locations))]
}

func calculateDistance() float64 {
	return 10.0 + math.Round(rand.Float64()*490)
}

func generateNotes() string {
	if rand.Float64() < 0.80 {
		return ""
	}

	noteTemplates := []string{
		"Fragile items - handle with care",
		"Refrigerated transport required",
		"Customer requested morning delivery",
		"Call customer 30 minutes before arrival",
		"Signature required for delivery",
		"Leave at front desk if no answer",
		"Heavy items - two-person delivery",
		"Priority shipment - expedite",
	}

	return noteTemplates[rand.Intn(len(noteTemplates))]
}

func formatNullableTime(t *time.Time) string {
	if t == nil {
		return "NULL"
	}
	return fmt.Sprintf("'%s'", t.Format("2006-01-02 15:04:05"))
}

func GenerateInsertStatements(shipments []Shipment) string {
	if len(shipments) == 0 {
		return ""
	}

	var sb strings.Builder
	sb.Grow(len(shipments) * 200)
	sb.WriteString("INSERT INTO shipments (id, order_id, driver_id, vehicle_id, status, scheduled_start, scheduled_end, actual_start, actual_end, pickup_location, delivery_location, distance_km, notes) VALUES\n")

	for i, s := range shipments {
		sb.WriteString("    (")
		sb.WriteString(strconv.Itoa(s.ID))
		sb.WriteString(", ")
		sb.WriteString(strconv.Itoa(s.OrderID))
		sb.WriteString(", ")
		sb.WriteString(strconv.Itoa(s.DriverID))
		sb.WriteString(", ")
		sb.WriteString(strconv.Itoa(s.VehicleID))
		sb.WriteString(", '")
		sb.WriteString(string(s.Status))
		sb.WriteString("', '")
		sb.WriteString(s.ScheduledStart.Format("2006-01-02 15:04:05"))
		sb.WriteString("', '")
		sb.WriteString(s.ScheduledEnd.Format("2006-01-02 15:04:05"))
		sb.WriteString("', ")
		sb.WriteString(formatNullableTime(s.ActualStart))
		sb.WriteString(", ")
		sb.WriteString(formatNullableTime(s.ActualEnd))
		sb.WriteString(", '")
		sb.WriteString(escapeSQL(s.PickupLocation))
		sb.WriteString("', '")
		sb.WriteString(escapeSQL(s.DeliveryLocation))
		sb.WriteString("', ")
		sb.WriteString(strconv.FormatFloat(s.DistanceKm, 'f', 2, 64))
		sb.WriteString(", ")

		if s.Notes == "" {
			sb.WriteString("NULL")
		} else {
			sb.WriteString("'")
			sb.WriteString(escapeSQL(s.Notes))
			sb.WriteString("'")
		}

		sb.WriteString(")")
		if i < len(shipments)-1 {
			sb.WriteString(",\n")
		} else {
			sb.WriteString(";\n")
		}
	}

	return sb.String()
}

func escapeSQL(value string) string {
	return strings.ReplaceAll(value, "'", "''")
}
