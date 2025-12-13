# Vehicle Selection Feature Implementation Plan

## Overview
Add vehicle selection to parking game with 6 vehicle types: Sedan, SUV, Compact, Sports Car, Motorcycle, Scooter. Each with distinct physics and dimensions. One-time selection before level 1, persisted to localStorage.

## Game Flow
```
TITLE_SCREEN → [Start] → VEHICLE_SELECTION → [Confirm] → RUNNING (Level 0)
                              ↑
                              └─ "Change Vehicle" button from UI panel
```

## Vehicle Definitions

### Physical Properties (VEHICLE_TYPES object)
Each type has: `width`, `length`, `wheelBase`, `maxSpeed`, `acceleration`, `maxSteerAngle`, `brakingForce`, `tireGrip`, `turboMultiplier`

| Type | Dimensions | Max Speed | Accel | Handling | Turbo Multiplier | Description |
|------|-----------|-----------|-------|----------|------------------|-------------|
| **Sedan** | 44×90 | 18.0 | 0.05 | 0.65° | 1.4x | Balanced (current default) |
| **SUV** | 50×115 | 16.0 | 0.04 | 0.55° | 1.3x | Large & stable, harder parking |
| **Compact** | 40×80 | 17.0 | 0.055 | 0.75° | 1.5x | Nimble & easy |
| **Sports** | 42×95 | 22.0 | 0.15 | 0.70° | 1.8x | Fast & responsive |
| **Bike** | 20×50 | 20.0 | 0.12 | 0.85° | 1.7x | Ultra-compact |
| **Scooter** | 18×45 | 12.0 | 0.08 | 0.90° | 1.6x | Slowest but tiny |

### Turbo System
**Activation:** Hold Shift key while driving
**Effect:** Multiplies max speed and acceleration by `turboMultiplier`
**Feedback:**
- Visual: Particle trails behind vehicle, screen shake effect, speed lines
- Audio: Turbo sound effect, engine pitch increase
- UI: Turbo indicator (glowing TURBO text when active)

## UI Design: Selection Screen

```
┌─────────────────────────────┐
│  WYBIERZ POJAZD / CHOOSE    │
├─────────────────────────────┤
│  [←]  VEHICLE NAME  [→]     │
│                             │
│   [Live Vehicle Preview]    │
│                             │
│  Stats:                     │
│  • Speed: ████████░░        │
│  • Acceleration: ██████░░░  │
│  • Handling: ███████░░░     │
│  • Size: ██████████         │
│                             │
│  "Description text"         │
├─────────────────────────────┤
│    [POTWIERDŹ/CONFIRM]      │
└─────────────────────────────┘
```

**Controls:**
- Arrow Left/Right: Navigate vehicles
- Enter: Confirm selection
- Mouse: Click arrows/confirm button

## Implementation Steps

### Phase 1: Core Data (30 min)
**File:** `index.html` (near CONFIG object, ~line 224)

1. Define `VEHICLE_TYPES` object with all 6 vehicle configs
2. Add `selectedVehicle` property to Game constructor (default: `localStorage.getItem('selectedVehicle') || 'sedan'`)
3. Add `saveVehicleSelection(type)` helper function

### Phase 2: PlayerCar Refactoring (45 min)
**File:** `index.html` (PlayerCar class, ~line 900)

1. Modify `constructor(x, y, angleDeg, vehicleType = 'sedan')`
   - Store `this.vehicleType = vehicleType`

2. Update `reset(x, y, angleDeg)` method:
   ```js
   const vType = VEHICLE_TYPES[this.vehicleType];
   this.w = vType.width;
   this.l = vType.length;
   this.wheelBase = vType.wheelBase;
   this.vehicleMaxSpeed = vType.maxSpeed;
   this.vehicleAcceleration = vType.acceleration;
   this.vehicleMaxSteerAngle = vType.maxSteerAngle;
   this.vehicleBrakingForce = vType.brakingForce;
   this.vehicleTireGrip = vType.tireGrip;
   ```

3. Replace CONFIG references in `updateSimplePhysics()` and `updateWinterPhysics()`:
   - `CONFIG.maxSpeed` → `this.vehicleMaxSpeed`
   - `CONFIG.acceleration` → `this.vehicleAcceleration`
   - `CONFIG.maxSteerAngle` → `this.vehicleMaxSteerAngle`
   - etc.

4. Update `Game.loadLevel()` to pass vehicle type:
   ```js
   this.player = new PlayerCar(ld.start.x, ld.start.y, ld.start.angle, this.selectedVehicle);
   ```

### Phase 3: Selection Screen Rendering (60 min)
**File:** `index.html` (Game class, add after `drawTitleScreen()`, ~line 2482)

1. Add state constant: `VEHICLE_SELECTION`

2. Add properties to Game constructor:
   ```js
   this.selectionIndex = 0;
   this.selectionVehicles = Object.keys(VEHICLE_TYPES); // ['sedan', 'suv', ...]
   this.leftArrowBounds = null;
   this.rightArrowBounds = null;
   this.confirmButtonBounds = null;
   ```

3. Create `drawVehicleSelectionScreen()` method:
   - Background gradient (reuse title screen style)
   - Title text: "WYBIERZ POJAZD"
   - Navigation arrows (← →) with bounds tracking
   - Vehicle preview: Create temp `PlayerCar` with current `vehicleType`, render at center
   - Stats bars: Calculate percentages, render horizontal bars
   - Confirm button with bounds tracking

4. Helper method `getVehicleStatPercent(vehicle, stat)`:
   ```js
   const ranges = {
     maxSpeed: { min: 12, max: 22 },
     acceleration: { min: 0.04, max: 0.15 },
     maxSteerAngle: { min: 0.55, max: 0.90 },
     size: { min: 810, max: 5750 }
   };
   // Calculate percentage within range
   ```

5. Update `Game.draw()` to handle VEHICLE_SELECTION state:
   ```js
   if (this.state === 'VEHICLE_SELECTION') {
     this.drawVehicleSelectionScreen();
   }
   ```

### Phase 4: Selection Screen Interaction (30 min)
**File:** `index.html` (Event handlers, ~line 2853)

1. Add keyboard handlers:
   ```js
   if (game.state === 'VEHICLE_SELECTION') {
     if (key === 'ArrowLeft') {
       game.selectionIndex = (game.selectionIndex - 1 + N) % N;
     }
     if (key === 'ArrowRight') {
       game.selectionIndex = (game.selectionIndex + 1) % N;
     }
     if (key === 'Enter') {
       const selected = game.selectionVehicles[game.selectionIndex];
       game.selectedVehicle = selected;
       localStorage.setItem('selectedVehicle', selected);
       game.startGame();
     }
   }
   ```

2. Add mouse click handlers:
   ```js
   if (game.state === 'VEHICLE_SELECTION') {
     // Check if click in leftArrowBounds → decrement index
     // Check if click in rightArrowBounds → increment index
     // Check if click in confirmButtonBounds → save & start
   }
   ```

3. Add mousemove handler for hover effects (optional polish)

### Phase 5: Game Flow Integration (20 min)
**File:** `index.html` (Game class methods)

1. Modify `startGame()` method:
   ```js
   startGame() {
     if (!localStorage.getItem('selectedVehicle')) {
       this.state = 'VEHICLE_SELECTION';
       return;
     }
     this.currentLevelIndex = 0;
     this.loadLevel(0);
     this.state = 'RUNNING';
   }
   ```

2. Add "Change Vehicle" button to UI panel:
   - Add button after existing UI buttons
   - Click handler sets `state = 'VEHICLE_SELECTION'`
   - Reloads current level after selection

### Phase 6: Visual Polish (30 min)

1. Vehicle preview animation:
   - Scale pulse: `1.0 + Math.sin(time) * 0.05`
   - Slight rotation wobble (optional)

2. Stats bar styling:
   - Gradient fills (green→yellow→red based on value)
   - Smooth rendering with rounded ends
   - Label text beside bars

3. Transition animations:
   - Fade in/out between states (optional)
   - Button hover highlights

4. Adjust wheel rendering for bike/scooter:
   - Scale down wheel size proportionally
   - Adjust headlight positions

### Phase 7: Turbo System Implementation (60 min)

**A. Input Handling**
**File:** `index.html` (Event handlers, ~line 2853)

1. Track Shift key state:
   ```js
   const keyState = {
     up: false,
     down: false,
     left: false,
     right: false,
     handbrake: false,
     turbo: false  // NEW
   };
   ```

2. Add Shift key handlers:
   ```js
   document.addEventListener('keydown', (e) => {
     if (e.key === 'Shift') {
       keyState.turbo = true;
       e.preventDefault(); // Prevent browser default behavior
     }
   });

   document.addEventListener('keyup', (e) => {
     if (e.key === 'Shift') {
       keyState.turbo = false;
     }
   });
   ```

**B. Physics Integration**
**File:** `index.html` (PlayerCar.update methods, ~line 1000)

1. Add turbo state to PlayerCar:
   ```js
   constructor(x, y, angleDeg, vehicleType = 'sedan') {
     // ... existing code
     this.isTurboActive = false;
   }
   ```

2. Modify `updateSimplePhysics()` and `updateWinterPhysics()`:
   ```js
   update(input, deltaTime) {
     const vType = VEHICLE_TYPES[this.vehicleType];
     this.isTurboActive = input.turbo;

     const turboBoost = this.isTurboActive ? vType.turboMultiplier : 1.0;
     const effectiveMaxSpeed = this.vehicleMaxSpeed * turboBoost;
     const effectiveAccel = this.vehicleAcceleration * turboBoost;

     // Use effectiveMaxSpeed and effectiveAccel in physics calculations
   }
   ```

**C. Visual Effects**
**File:** `index.html` (PlayerCar.draw and Game class, ~line 950, 2700)

1. Particle trail system (add to PlayerCar):
   ```js
   constructor() {
     // ... existing
     this.turboParticles = []; // {x, y, life, angle}
   }

   update(input, deltaTime) {
     // ... physics

     // Spawn turbo particles
     if (this.isTurboActive && Math.abs(this.speed) > 2) {
       this.turboParticles.push({
         x: this.x - Math.cos(this.angle) * this.l / 2,
         y: this.y - Math.sin(this.angle) * this.l / 2,
         life: 1.0,
         angle: this.angle + Math.PI
       });
     }

     // Update particles
     this.turboParticles = this.turboParticles.filter(p => {
       p.life -= deltaTime * 2;
       return p.life > 0;
     });
   }

   drawTurboEffects(ctx) {
     this.turboParticles.forEach(p => {
       ctx.save();
       ctx.globalAlpha = p.life * 0.6;
       ctx.fillStyle = '#FFD700'; // Gold color
       ctx.beginPath();
       ctx.arc(p.x, p.y, 5 * p.life, 0, Math.PI * 2);
       ctx.fill();
       ctx.restore();
     });
   }
   ```

2. Screen shake (add to Game class):
   ```js
   draw() {
     ctx.save();

     // Screen shake when turbo active
     if (this.player.isTurboActive) {
       const shakeX = (Math.random() - 0.5) * 3;
       const shakeY = (Math.random() - 0.5) * 3;
       ctx.translate(shakeX, shakeY);
     }

     // ... existing draw code
     ctx.restore();
   }
   ```

3. Speed lines (add to Game.draw):
   ```js
   drawSpeedLines() {
     if (!this.player.isTurboActive) return;

     ctx.save();
     ctx.strokeStyle = 'rgba(255, 255, 255, 0.3)';
     ctx.lineWidth = 2;

     for (let i = 0; i < 20; i++) {
       const x = Math.random() * canvas.width;
       const y = Math.random() * canvas.height;
       const length = 20 + Math.random() * 30;

       ctx.beginPath();
       ctx.moveTo(x, y);
       ctx.lineTo(x + length, y);
       ctx.stroke();
     }
     ctx.restore();
   }
   ```

**D. Audio Effects**
**File:** `index.html` (Audio setup, ~line 180)

1. Add turbo sound effect:
   ```js
   const turboSound = new Audio();
   turboSound.src = 'data:audio/wav;base64,...'; // Or load external file
   turboSound.loop = true;
   turboSound.volume = 0.4;
   ```

2. Control playback in update:
   ```js
   update(deltaTime) {
     // ... existing code

     if (this.player.isTurboActive && turboSound.paused) {
       turboSound.currentTime = 0;
       turboSound.play();
     } else if (!this.player.isTurboActive && !turboSound.paused) {
       turboSound.pause();
     }
   }
   ```

**E. UI Indicator**
**File:** `index.html` (UI panel, ~line 2800)

1. Add turbo indicator to UI:
   ```js
   drawUI() {
     // ... existing UI elements

     // Turbo indicator (top-right corner)
     if (this.player.isTurboActive) {
       ctx.save();
       const time = Date.now() / 200;
       const pulse = 0.8 + Math.sin(time) * 0.2;

       ctx.font = 'bold 24px Arial';
       ctx.fillStyle = `rgba(255, 215, 0, ${pulse})`;
       ctx.strokeStyle = 'rgba(0, 0, 0, 0.8)';
       ctx.lineWidth = 3;

       const text = 'TURBO';
       const x = canvas.width - 120;
       const y = 50;

       ctx.strokeText(text, x, y);
       ctx.fillText(text, x, y);

       // Glow effect
       ctx.shadowColor = '#FFD700';
       ctx.shadowBlur = 20;
       ctx.fillText(text, x, y);

       ctx.restore();
     }
   }
   ```

2. Update vehicle selection stats to include turbo:
   ```js
   drawVehicleStats(vehicle) {
     // ... existing stats (Speed, Acceleration, Handling, Size)

     // Add Turbo stat
     const turboPercent = Math.round((vehicle.turboMultiplier - 1.0) / 0.8 * 100);
     this.drawStatBar(x, y + 100, 'Turbo Boost', turboPercent);
   }
   ```

### Phase 8: Testing

1. Test each vehicle completes Level 1
2. Test each vehicle completes Level 1 WITH turbo
3. Verify localStorage persistence across page refreshes
4. Test "Change Vehicle" button from running state
5. Test bike/scooter rendering (different proportions)
6. Verify parking zone detection works for all sizes
7. Test turbo activation/deactivation (Shift key)
8. Verify turbo visual effects (particles, shake, speed lines)
9. Verify turbo audio feedback
10. Verify turbo UI indicator appears/disappears correctly
11. Test turbo with each vehicle type (different multipliers)
12. Balance physics if any vehicle feels broken

## Critical Files

- **index.html** (Lines 224-271): Add VEHICLE_TYPES config
- **index.html** (Lines 900-1100): PlayerCar class modifications
- **index.html** (Lines 1531-1580): Game constructor, state management
- **index.html** (Lines 2482-2600): Add drawVehicleSelectionScreen()
- **index.html** (Lines 2853-2920): Event handlers

## Technical Notes

- All changes in single file (`index.html`)
- Vanilla JS, no dependencies
- localStorage for persistence (key: `'selectedVehicle'`)
- Backward compatible (sedan as default)
- Vehicle preview reuses existing PlayerCar.draw() method
- Stats normalized to 0-100% scale for display

## Design Decisions

1. **One-time selection** reduces friction, persists choice
2. **Distinct physics** makes vehicle choice meaningful, not just cosmetic
3. **6 vehicles** provides variety without overwhelming choice
4. **Bike/scooter** add extreme challenge (tiny size = precision required)
5. **Sports car** rewards skilled players with speed
6. **Turbo system** adds arcade excitement, per-vehicle tuning creates strategic depth
7. **LocalStorage** simple persistence without backend

## Expected Behavior Changes

### Vehicle Selection Impact
- **Easier parking**: Bike, scooter (smaller footprint)
- **Harder parking**: SUV (larger, less maneuverable)
- **Speed challenge**: Sports car (faster but requires control)
- **Balanced**: Sedan, compact (original experience)

### Turbo Impact
- **Sports car + turbo**: Extreme speed (22.0 × 1.8 = 39.6 max speed!) - high risk/reward
- **Scooter + turbo**: Becomes viable (12.0 × 1.6 = 19.2, matches sedan base speed)
- **SUV + turbo**: Compensates for slow base speed (16.0 × 1.3 = 20.8)
- **Strategic use**: Turbo for straightaways, normal for precise parking maneuvers

## Implementation Summary

**Total Effort:** ~4.5 hours
- Phase 1-2: Core vehicle system (75 min)
- Phase 3-4: Selection screen (90 min)
- Phase 5-6: Game flow + polish (50 min)
- Phase 7: Turbo system (60 min)
- Phase 8: Testing (30 min)

**Key Features:**
1. ✅ 6 distinct vehicles (3 existing + 3 new)
2. ✅ Vehicle selection screen with preview + stats
3. ✅ Per-vehicle physics (dimensions, handling, speed)
4. ✅ Turbo system (Shift key activation)
5. ✅ Per-vehicle turbo multipliers (1.3x - 1.8x)
6. ✅ Visual feedback (particles, shake, speed lines)
7. ✅ Audio feedback (turbo sound loop)
8. ✅ UI indicators (turbo text, vehicle stats)
9. ✅ LocalStorage persistence
10. ✅ "Change Vehicle" button in UI

**File Changes:** Single file (`index.html`) - all modifications inline
