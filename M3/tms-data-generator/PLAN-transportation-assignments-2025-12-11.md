# Implementation Plan: Transportation Assignments & Availability Tracking

## Overview

Add a transportation assignment entity that links orders to drivers and vehicles with time slots, enabling availability queries. The system will track which driver/vehicle is assigned to which order, with start/end times to prevent scheduling conflicts.

## Analysis

### Current State

**Existing Domain Structure:**
- **Vehicles** (50): Basic fleet data (make/model/year/fuel capacity)
- **Drivers** (20): Employee records with status (ACTIVE, ON_ROUTE, RESTING, OFF_DUTY, SICK_LEAVE)
- **Customers** (500): Three types (INDIVIDUAL/BUSINESS/VIP)
- **Transportation Orders** (1000): Delivery jobs with status tracking and timeline events

**Generation Pattern:**
- Each domain has 3 files: `model.go`, `{domain}.go` (no separate file for insert statements)
- Concurrent generation using goroutines + sync.WaitGroup in `generator.go`
- Configuration constants in `generator/config/count.go`
- Single INSERT statements per table with string builders
- Referential integrity maintained by passing dependent entities as parameters

**Current Order Status Flow:**
- PENDING → PROCESSING → IN_TRANSIT/READY_FOR_PICKUP → DELIVERED/CANCELLED
- Orders have timeline events tracking state transitions
- 90% of historical orders (>2 weeks) are DELIVERED/CANCELLED
- 30% of recent orders are in progress

### Affected Components

**New Domain:**
- `generator/shipments/` (new directory)
  - `model.go` - Shipment struct and enums
  - `shipments.go` - Generation logic and SQL statements

**Modified Files:**
- `generator/config/count.go` - Add SHIPMENTS constant
- `generator/generator.go` - Wire shipments into generation pipeline
- `schema/create-tms-schema.sql` - Add shipments table and indices

**Constraints:**
- Driver availability: Status must be ACTIVE or ON_ROUTE during assignment
- Vehicle availability: Only one assignment per vehicle at a given time
- Time allocation: 70% of orders should have assignments (realistic for completed/in-transit orders)

### Dependencies

**Data Dependencies:**
- Shipments depend on: orders, drivers, vehicles
- Generation order: vehicles/drivers/customers (parallel) → orders → shipments

**Library Usage:**
- `github.com/brianvoe/gofakeit/v6` for realistic time slots and status generation
- Standard Go `time` package for temporal calculations

## Naming Decision

**Recommended Entity Name: `Shipment`**

**Rationale:**
- **Shipment** is standard logistics terminology for the physical movement of goods
- Distinguishes between:
  - **Order**: Customer request (what needs to be delivered)
  - **Shipment**: Execution plan (how/when it will be delivered with assigned resources)
- Common in TMS systems (FreightWaves, SAP TMS, Oracle Transportation Management)
- Clearer than "Assignment" (too generic) or "Trip" (implies passenger transport)

**Alternative Names Considered:**
- ❌ `TransportationAssignment`: Too verbose, redundant with "transportation_orders"
- ❌ `OrderAssignment`: Ambiguous (what is being assigned?)
- ❌ `Trip`: Better suited for passenger transportation
- ❌ `Delivery`: Conflicts with "delivery" status in orders
- ✅ `Shipment`: Industry standard, clear purpose

## Implementation Strategy

### Phase 1: Create Shipments Domain Structure

**Objective:** Establish the shipments domain with model definitions and enums

**Files to Create:**

#### `generator/shipments/model.go`
```go
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
```

**Key Design Decisions:**
- `ScheduledStart/End`: Planned time slot (always present)
- `ActualStart/End`: Realized times (nullable, only for in-progress/completed)
- `PickupLocation`: Typically warehouse/distribution center
- `DeliveryLocation`: From order's shipping address
- `DistanceKm`: For realistic time slot calculation (speed-based estimation)
- `Status`: Derived from order status but independent (allows scheduling future shipments)

---

### Phase 2: Implement Shipment Generation Logic

**Objective:** Generate realistic shipments with non-conflicting time slots

**Files to Create:**

#### `generator/shipments/shipments.go`

**Core Functions:**

1. **`GenerateShipments(orders, drivers, vehicles)`**
   - Iterate through orders with suitable statuses (IN_TRANSIT, READY_FOR_PICKUP, DELIVERED)
   - Assign 70% of eligible orders to shipments
   - Select available driver/vehicle pairs using conflict detection
   - Calculate realistic time slots based on order dates and delivery expectations

2. **`calculateTimeSlot(order)`**
   - For DELIVERED orders: actual slot between order date and delivery event
   - For IN_TRANSIT orders: started in past, ends in future
   - For READY_FOR_PICKUP: scheduled for near future
   - Duration: 2-8 hours based on distance estimation

3. **`selectAvailableResources(drivers, vehicles, startTime, endTime, assignedShipments)`**
   - Filter drivers with ACTIVE/ON_ROUTE status
   - Check existing assignments for time conflicts
   - Use simple collision detection: no overlap between [start1, end1] and [start2, end2]
   - Return first available driver/vehicle pair

4. **`checkTimeOverlap(start1, end1, start2, end2)`**
   - Returns true if time ranges overlap
   - Logic: `start1 < end2 && start2 < end1`

5. **`generatePickupLocation()`**
   - Random selection from realistic distribution centers
   - Examples: "Central Warehouse A", "Distribution Hub North", "Logistics Center East"

6. **`calculateDistance()`**
   - Random distance 10-500 km
   - Use order's shipping address as hint (longer distance for interstate)

7. **`GenerateInsertStatements(shipments)`**
   - Standard SQL generation following existing pattern
   - Handle NULL for ActualStart/ActualEnd using conditional logic

**Algorithm Pseudocode:**
```
shipments = []
assignedDrivers = map[driverID][]timeRange
assignedVehicles = map[vehicleID][]timeRange

for each order where status IN (DELIVERED, IN_TRANSIT, READY_FOR_PICKUP):
    if random() < 0.70:  // 70% assignment rate
        startTime, endTime = calculateTimeSlot(order)

        for each driver in shuffled(activeDrivers):
            if !hasTimeConflict(assignedDrivers[driver.ID], startTime, endTime):
                for each vehicle in shuffled(vehicles):
                    if !hasTimeConflict(assignedVehicles[vehicle.ID], startTime, endTime):
                        shipment = createShipment(order, driver, vehicle, startTime, endTime)
                        shipments.append(shipment)

                        assignedDrivers[driver.ID].append([startTime, endTime])
                        assignedVehicles[vehicle.ID].append([startTime, endTime])

                        break out of both loops

return shipments
```

**Realistic Time Slot Generation Strategy:**

| Order Status      | Scheduled Start                | Scheduled End                    | Actual Times                        |
|-------------------|--------------------------------|----------------------------------|-------------------------------------|
| DELIVERED         | order_date + 0-12h             | scheduled_start + 2-8h           | ✅ Both populated, close to scheduled |
| IN_TRANSIT        | order_date + 0-24h             | scheduled_start + 3-10h          | ✅ ActualStart populated, end NULL  |
| READY_FOR_PICKUP  | now + 1-48h                    | scheduled_start + 2-6h           | ❌ Both NULL (not started)          |

**Status Mapping:**
- DELIVERED order → COMPLETED shipment (100%)
- IN_TRANSIT order → IN_PROGRESS shipment (80%), SCHEDULED (20%)
- READY_FOR_PICKUP order → SCHEDULED shipment (100%)
- PROCESSING/PENDING order → No shipment (not yet assigned)
- CANCELLED order → CANCELLED shipment if previously created (10%)

---

### Phase 3: Integrate into Generation Pipeline

**Objective:** Wire shipments generation into the concurrent pipeline

**Files to Modify:**

#### `generator/config/count.go`
```go
const (
	VEHICLES              = 50
	DRIVERS               = 20
	TRANSPORTATION_ORDERS = 1000
	CUSTOMERS             = 500
	// No constant needed - shipments count derived from orders (70% coverage)
)
```

**Rationale:** Shipments count is dynamic (70% of eligible orders), not a fixed configuration value.

#### `generator/generator.go`

**Changes:**

1. **Import shipments package** (line ~16):
```go
import (
	// ... existing imports
	"tms-data-generator/generator/shipments"
)
```

2. **Add shipments variable** (line ~50):
```go
var vehiclesStatements string
var driversStatements string
var customersStatements string
var shipmentsStatements string  // NEW
```

3. **Generate shipments after orders** (after line ~106):
```go
// Phase 6: Generate shipments (depends on orders, drivers, vehicles)
startShipments := time.Now()
fmt.Println("Generating shipments...", time.Now())
driversList := drivers.GenerateDrivers(config.DRIVERS)
vehiclesList := vehicles.GenerateVehicles(config.VEHICLES)
shipmentsList := shipments.GenerateShipments(ordersList, driversList, vehiclesList)
shipmentsStatements = shipments.GenerateInsertStatements(shipmentsList)
fmt.Println("done generating shipments", time.Now(), time.Since(startShipments))
```

**Note:** Shipments must be generated AFTER orders but requires re-generating drivers/vehicles lists (or passing them through from Phase 1).

**Optimization Option:** Cache generated entities to avoid regeneration:
```go
// Phase 1: Store generated entities
var driversList []drivers.Driver
var vehiclesList []vehicles.Vehicle

// In goroutines, populate these instead of just statements
```

4. **Append to output** (line ~130):
```go
sb.WriteString(vehiclesStatements)
sb.WriteString(driversStatements)
sb.WriteString(customersStatements)
sb.WriteString(ordersStatements)
sb.WriteString(timelineStatements)
sb.WriteString(itemsStatements)
sb.WriteString(shipmentsStatements)  // NEW - must be after vehicles/drivers/orders
```

---

### Phase 4: Database Schema Updates

**Objective:** Add shipments table with proper constraints and indices

**Files to Modify:**

#### `schema/create-tms-schema.sql`

**Changes:**

1. **Add DROP statement** (after line 3):
```sql
DROP TABLE IF EXISTS shipments;
DROP TABLE IF EXISTS order_items;
DROP TABLE IF EXISTS order_timeline_events;
```

**Rationale:** Shipments must be dropped before transportation_orders due to foreign key.

2. **Add shipments table** (after line 73, before indices):
```sql
CREATE TABLE shipments (
    id INT PRIMARY KEY,
    order_id INT NOT NULL,
    driver_id INT NOT NULL,
    vehicle_id INT NOT NULL,
    status VARCHAR(20) NOT NULL,
    scheduled_start TIMESTAMP NOT NULL,
    scheduled_end TIMESTAMP NOT NULL,
    actual_start TIMESTAMP,
    actual_end TIMESTAMP,
    pickup_location VARCHAR(255) NOT NULL,
    delivery_location VARCHAR(255) NOT NULL,
    distance_km DECIMAL(6,2) NOT NULL,
    notes TEXT,
    FOREIGN KEY (order_id) REFERENCES transportation_orders(id),
    FOREIGN KEY (driver_id) REFERENCES drivers(id),
    FOREIGN KEY (vehicle_id) REFERENCES vehicles(id),
    CONSTRAINT check_scheduled_times CHECK (scheduled_end > scheduled_start),
    CONSTRAINT check_actual_times CHECK (actual_end IS NULL OR actual_start IS NOT NULL),
    CONSTRAINT check_actual_after_scheduled CHECK (actual_start IS NULL OR actual_start >= scheduled_start)
);
```

**Constraints Explained:**
- `check_scheduled_times`: End must be after start (data quality)
- `check_actual_times`: Can't have actual end without actual start
- `check_actual_after_scheduled`: Actual start can't be before scheduled (realistic)

3. **Add indices** (after line 78):
```sql
CREATE INDEX idx_shipments_order ON shipments(order_id);
CREATE INDEX idx_shipments_driver ON shipments(driver_id);
CREATE INDEX idx_shipments_vehicle ON shipments(vehicle_id);
CREATE INDEX idx_shipments_status ON shipments(status);
CREATE INDEX idx_shipments_scheduled ON shipments(scheduled_start, scheduled_end);
```

**Index Rationale:**
- `idx_shipments_order`: Lookup shipment for an order
- `idx_shipments_driver`: Find all shipments for a driver
- `idx_shipments_vehicle`: Find all shipments for a vehicle
- `idx_shipments_status`: Filter by shipment status
- `idx_shipments_scheduled`: **Key for availability queries** - find conflicts in time ranges

---

### Phase 5: Implementation Details & Edge Cases

**Objective:** Document implementation nuances and testing approach

#### Conflict Detection Algorithm

**Time Overlap Logic:**
```go
func checkTimeOverlap(start1, end1, start2, end2 time.Time) bool {
    return start1.Before(end2) && start2.Before(end1)
}
```

**Examples:**
- Shipment A: 08:00-12:00
- Shipment B: 10:00-14:00 → **CONFLICT** (overlap 10:00-12:00)
- Shipment C: 12:00-16:00 → **NO CONFLICT** (exactly adjacent, 12:00 is boundary)
- Shipment D: 06:00-08:00 → **NO CONFLICT** (completely before)

**Data Structure for Tracking:**
```go
type timeRange struct {
    start time.Time
    end   time.Time
}

assignedDrivers := make(map[int][]timeRange)   // driverID -> list of assigned time ranges
assignedVehicles := make(map[int][]timeRange)  // vehicleID -> list of assigned time ranges
```

**Conflict Check:**
```go
func hasConflict(assignments []timeRange, newStart, newEnd time.Time) bool {
    for _, assignment := range assignments {
        if checkTimeOverlap(assignment.start, assignment.end, newStart, newEnd) {
            return true
        }
    }
    return false
}
```

#### Handling Assignment Failures

**Scenario:** No available driver/vehicle for a time slot

**Strategy:**
1. Shuffle drivers and vehicles to randomize selection (avoid always assigning first resources)
2. If no match found after checking all combinations, skip that order
3. Log skipped orders (for debugging if assignment rate drops below 50%)
4. Realistic outcome: Some orders may not have shipments (outsourced, cancelled before assignment, etc.)

**Expected Coverage:**
- Target: 70% of eligible orders assigned
- Actual: 55-75% (variance due to conflict constraints)
- If <50%: Indicates resource shortage (increase DRIVERS or VEHICLES constants)

#### Null Handling in SQL Generation

**Pattern for Nullable Fields:**
```go
func formatNullableTime(t *time.Time) string {
    if t == nil {
        return "NULL"
    }
    return fmt.Sprintf("'%s'", t.Format("2006-01-02 15:04:05"))
}

// In GenerateInsertStatements:
sb.WriteString(fmt.Sprintf("    (%d, %d, %d, %d, '%s', '%s', '%s', %s, %s, '%s', '%s', %.2f, '%s')",
    shipment.ID,
    shipment.OrderID,
    shipment.DriverID,
    shipment.VehicleID,
    shipment.Status,
    shipment.ScheduledStart.Format("2006-01-02 15:04:05"),
    shipment.ScheduledEnd.Format("2006-01-02 15:04:05"),
    formatNullableTime(shipment.ActualStart),   // NULL or timestamp
    formatNullableTime(shipment.ActualEnd),     // NULL or timestamp
    // ...
))
```

#### Sample Data Characteristics

**For 1000 orders with default config:**
- Eligible orders (DELIVERED/IN_TRANSIT/READY_FOR_PICKUP): ~920 orders
- Target shipments (70% of eligible): ~640 shipments
- Drivers: 20 (avg ~32 shipments per driver)
- Vehicles: 50 (avg ~13 shipments per vehicle)

**Status Distribution:**
- COMPLETED: ~75% (historical delivered orders)
- IN_PROGRESS: ~15% (current in-transit orders)
- SCHEDULED: ~8% (upcoming ready-for-pickup orders)
- CANCELLED: ~2% (cancelled orders with prior assignments)

#### Notes Field Generation

**Use Cases:**
- Special handling instructions: "Fragile items", "Refrigerated transport required"
- Route notes: "Avoid highway construction on I-95"
- Delivery instructions: "Call customer 30 min before arrival"
- Empty for most shipments (realistic - only ~20% have notes)

**Implementation:**
```go
func generateNotes() string {
    if rand.Float64() < 0.80 {  // 80% no notes
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
```

---

## Database Schema for Availability Queries

### Enabled Query Patterns

**1. Find available drivers at a specific time:**
```sql
SELECT d.id, d.first_name, d.last_name, d.status
FROM drivers d
WHERE d.status IN ('ACTIVE', 'ON_ROUTE')
  AND d.id NOT IN (
      SELECT driver_id
      FROM shipments
      WHERE status IN ('SCHEDULED', 'IN_PROGRESS')
        AND scheduled_start < '2025-12-15 14:00:00'
        AND scheduled_end > '2025-12-15 10:00:00'
  );
```

**2. Find available vehicles in a time range:**
```sql
SELECT v.id, v.make, v.model
FROM vehicles v
WHERE v.id NOT IN (
    SELECT vehicle_id
    FROM shipments
    WHERE status IN ('SCHEDULED', 'IN_PROGRESS')
      AND scheduled_start < :end_time
      AND scheduled_end > :start_time
);
```

**3. Get driver workload/schedule:**
```sql
SELECT
    s.scheduled_start,
    s.scheduled_end,
    s.status,
    o.order_number,
    s.pickup_location,
    s.delivery_location
FROM shipments s
JOIN transportation_orders o ON s.order_id = o.id
WHERE s.driver_id = :driver_id
  AND s.scheduled_start >= CURRENT_DATE
ORDER BY s.scheduled_start;
```

**4. Vehicle utilization report:**
```sql
SELECT
    v.id,
    v.make,
    v.model,
    COUNT(s.id) as total_shipments,
    SUM(s.distance_km) as total_distance,
    AVG(TIMESTAMPDIFF(HOUR, s.scheduled_start, s.scheduled_end)) as avg_duration_hours
FROM vehicles v
LEFT JOIN shipments s ON v.id = s.vehicle_id
WHERE s.scheduled_start >= DATE_SUB(CURRENT_DATE, INTERVAL 30 DAY)
GROUP BY v.id, v.make, v.model
ORDER BY total_shipments DESC;
```

**5. Find scheduling conflicts (validation query):**
```sql
SELECT
    s1.id as shipment1_id,
    s2.id as shipment2_id,
    s1.driver_id,
    s1.scheduled_start as start1,
    s1.scheduled_end as end1,
    s2.scheduled_start as start2,
    s2.scheduled_end as end2
FROM shipments s1
JOIN shipments s2 ON s1.driver_id = s2.driver_id AND s1.id < s2.id
WHERE s1.scheduled_start < s2.scheduled_end
  AND s2.scheduled_start < s1.scheduled_end;
```

---

## Testing Strategy

### Unit Testing (Go)

**No existing test files in the project** - tests are not required for this generator, but if added:

**Test Coverage:**
1. `TestCheckTimeOverlap` - Verify overlap detection logic
2. `TestCalculateTimeSlot` - Validate time slot generation for each order status
3. `TestSelectAvailableResources` - Ensure conflict detection works
4. `TestGenerateShipments` - Integration test for full generation
5. `TestNullableTimeSQLFormatting` - Verify NULL handling in SQL output

### Manual Validation

**After Running Generator:**

1. **Check assignment rate:**
```sql
SELECT
    (SELECT COUNT(*) FROM shipments) as shipment_count,
    (SELECT COUNT(*) FROM transportation_orders WHERE status IN ('DELIVERED', 'IN_TRANSIT', 'READY_FOR_PICKUP')) as eligible_orders,
    ROUND((SELECT COUNT(*) FROM shipments) * 100.0 / (SELECT COUNT(*) FROM transportation_orders WHERE status IN ('DELIVERED', 'IN_TRANSIT', 'READY_FOR_PICKUP')), 1) as assignment_percentage;
```
Expected: 55-75%

2. **Verify no conflicts exist:**
```sql
-- Should return 0 rows
SELECT COUNT(*) FROM (
    SELECT s1.id as shipment1_id, s2.id as shipment2_id
    FROM shipments s1
    JOIN shipments s2 ON s1.driver_id = s2.driver_id AND s1.id < s2.id
    WHERE s1.scheduled_start < s2.scheduled_end
      AND s2.scheduled_start < s1.scheduled_end
) conflicts;
```

3. **Check status consistency:**
```sql
SELECT
    s.status as shipment_status,
    o.status as order_status,
    COUNT(*) as count
FROM shipments s
JOIN transportation_orders o ON s.order_id = o.id
GROUP BY s.status, o.status
ORDER BY s.status, o.status;
```

Expected mappings:
- COMPLETED shipment → DELIVERED order
- IN_PROGRESS shipment → IN_TRANSIT order
- SCHEDULED shipment → READY_FOR_PICKUP order

4. **Validate time constraints:**
```sql
-- All should return 0
SELECT COUNT(*) FROM shipments WHERE scheduled_end <= scheduled_start;
SELECT COUNT(*) FROM shipments WHERE actual_end IS NOT NULL AND actual_start IS NULL;
SELECT COUNT(*) FROM shipments WHERE actual_start < scheduled_start;
```

---

## Risks and Considerations

### Resource Constraint Risk

**Issue:** With 20 drivers and 1000 orders, if time slots frequently overlap, assignment rate may drop below target.

**Mitigation:**
- Current config should be sufficient (50 vehicles, 20 drivers)
- Orders span 1 year, so time distribution is spread out
- Historical orders (>2 weeks) are mostly completed, freeing up resources
- Randomization in time slot generation reduces clustering

**Monitoring:** If assignment rate < 50%, increase DRIVERS or VEHICLES in config.

### Time Zone Handling

**Current Approach:** All timestamps use server local time (no explicit timezone in schema)

**Risk:** Queries spanning time zones may be ambiguous

**Recommendation:**
- For production: Use `TIMESTAMP WITH TIME ZONE` in PostgreSQL
- For generator: Document assumption that all times are UTC
- Add comment in schema: `-- All timestamps in UTC`

### Performance on Large Datasets

**Current Scale:** 1000 orders → ~640 shipments (manageable)

**Scaling Concerns:**
- Conflict detection is O(n²) in worst case (n = shipments per resource)
- At 10,000 orders → ~7,000 shipments, checking conflicts becomes slow

**Optimization Strategy (if needed):**
- Use sorted time ranges and binary search for overlap checks
- Pre-bin shipments by date (check only same-day conflicts)
- Current implementation sufficient for <5000 orders

### Data Realism Trade-offs

**Simplifications:**
- Pickup locations are generic strings (not actual addresses)
- Distance calculated randomly (not from real coordinates)
- Driver breaks/rest periods not modeled
- Vehicle maintenance windows not considered

**Justification:** Generator is for testing/demo, not simulation. Prioritize generation speed and data variety over perfect realism.

---

## Validation Checklist

- [ ] `generator/shipments/model.go` created with Shipment struct and enums
- [ ] `generator/shipments/shipments.go` created with generation and SQL logic
- [ ] `generator/config/count.go` reviewed (no new constant needed)
- [ ] `generator/generator.go` updated with shipments generation phase
- [ ] `schema/create-tms-schema.sql` updated with shipments table and indices
- [ ] DROP statements reordered to handle foreign keys correctly
- [ ] Run `task run` successfully generates SQL file
- [ ] Validate assignment rate (55-75% of eligible orders)
- [ ] Verify no scheduling conflicts exist (SQL query returns 0)
- [ ] Check shipment status aligns with order status
- [ ] Test availability queries return expected results
- [ ] Confirm file size increase is reasonable (~30-40% larger)

---

## Questions for Review

### 1. Entity Naming
**Approved:** "Shipment" as the entity name
- Alternative if you prefer: "Dispatch", "Assignment", "Route"
- Impact: Just naming convention, no logic changes needed

### 2. Assignment Coverage
**Current Plan:** 70% of eligible orders assigned to shipments
- **Question:** Should this be configurable or hardcoded?
- **Options:**
  - A) Keep at 70% (simple, realistic)
  - B) Add config constant `SHIPMENT_ASSIGNMENT_RATE = 0.70`
  - C) Make it 100% (every eligible order has shipment)

**Recommendation:** Keep at 70% hardcoded (simple, realistic for test data)

### 3. Location Data Detail
**Current Plan:** Pickup locations are generic strings like "Central Warehouse A"

- **Question:** Do you want actual addresses using gofakeit.Address()?
- **Trade-off:**
  - Generic strings: Faster, clearer that it's test data
  - Real addresses: More realistic, better for map-based UI testing

**Recommendation:** Use generic strings (faster, sufficient for SQL queries)

### 4. Resource Caching vs. Regeneration
**Current Approach:** Re-generate drivers/vehicles lists for shipments phase

- **Question:** Should we cache generated entities to avoid regeneration?
- **Options:**
  - A) Regenerate (current, simple but duplicates work)
  - B) Cache in variables and pass to shipments (efficient, more complex)

**Recommendation:** Start with regeneration (simpler), optimize if performance issue detected

### 5. Assignment Algorithm Complexity
**Current Plan:** Simple sequential search for available resources

- **Question:** If conflicts cause low assignment rate, should we implement smarter scheduling?
- **Options:**
  - A) Keep simple (current plan)
  - B) Add time slot adjustment (try nearby times if conflict)
  - C) Add priority-based assignment (assign important orders first)

**Recommendation:** Keep simple (current plan), adjust only if <50% assignment rate observed

### 6. Database Type Compatibility
**Current Schema:** Uses MySQL/PostgreSQL compatible syntax

- **Question:** What database will this be imported into?
- **Impact:**
  - PostgreSQL: Current schema works perfectly
  - MySQL: Need to adjust CHECK constraints (not fully supported before 8.0.16)
  - SQLite: Remove CHECK constraints entirely

**Recommendation:** Assume PostgreSQL (most feature-complete), document MySQL adjustments if needed

---

## Next Steps

**Immediate Actions:**
1. ✅ Review this plan and approve naming/design decisions
2. Implement Phase 1 (model.go) - foundational structs
3. Implement Phase 2 (shipments.go) - generation logic
4. Implement Phase 3 (config + generator.go) - pipeline integration
5. Implement Phase 4 (schema.sql) - database schema
6. Test generation: `task run`
7. Validate output with SQL queries
8. Adjust assignment rate or time slot logic if needed

**Estimated Implementation Time:**
- Phase 1: 20 minutes (straightforward struct definitions)
- Phase 2: 60 minutes (core logic, conflict detection, SQL generation)
- Phase 3: 15 minutes (wiring into pipeline)
- Phase 4: 10 minutes (schema updates)
- Testing/Validation: 30 minutes
- **Total: ~2.5 hours of implementation**

**Success Criteria:**
- ✅ `task run` completes without errors
- ✅ Generated SQL imports successfully
- ✅ 55-75% of eligible orders have shipments
- ✅ Zero scheduling conflicts detected
- ✅ Availability queries return correct results
- ✅ File size is reasonable (~250-280 KB vs current ~200 KB)
