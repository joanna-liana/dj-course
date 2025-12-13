# Parking Garage Spiral Level - Implementation Plan

## Overview
Add a new level (#14) featuring a spiral parking ramp that curves inward from outside edge to center, with medium difficulty (200px ramp width, 1.5 turns/540°).

## User Requirements
- **1.5 spiral turns** (540° total rotation)
- **200px ramp width** (medium difficulty)
- **Start outside → spiral inward** to center
- **Obstacles**: Curbs (walls), pillars (support columns), parked cars

## Implementation

### File to Modify
**`index.html`**
- Add new level object before line 2130 (end of levels array)
- Insert after the last level (around line 2128)

### Level Structure

```javascript
{
    name: "Spirala Parkingowa",
    type: 'lot',
    start: {
        x: canvas.width/2 + 400,
        y: canvas.height/2,
        angle: 90
    },
    obstacles: [
        // 6 pillars distributed along spiral (support columns)
        // Calculated using: r = 350 - 31.83*theta, positioned at theta = 0.5π, 1.0π, 1.5π, 2.0π, 2.5π, 2.8π
    ],
    cars: [
        // 3 obstacle cars at tight turns
        // Positioned at theta = π (180°), 2π (360°), 2.67π (480°)
        // Angled tangentially to spiral curve
    ],
    parkingZones: [
        // Single parking zone at spiral center
        // Position: canvas.width/2, canvas.height/2
    ],
    curbs: [
        // ~108 curbs total (54 outer wall + 54 inner wall)
        // Generated using loop: theta from 0 to 3π in π/18 steps
        // Positioned tangentially along spiral path
    ]
}
```

### Mathematical Foundation

**Archimedean Spiral (Inward):**
```
r(θ) = r_max - b*θ
where:
  r_max = 350px (outer starting radius)
  r_min = 50px (inner ending radius)
  b = (350 - 50) / (3π) ≈ 31.83
  θ = 0 to 3π radians (540°)
```

**Curb Positioning:**
- Outer wall: `r = 350 - 31.83*θ`
- Inner wall: `r = 350 - 31.83*θ - 200` (offset by ramp width)
- Tangent angle: `θ + π/2` (perpendicular to radius)
- Spacing: 10° steps (π/18 radians) = 54 curbs per wall

### Detailed Code Implementation

#### 1. Obstacles (Pillars)
```javascript
obstacles: (() => {
    const centerX = canvas.width / 2;
    const centerY = canvas.height / 2;
    const rMax = 350;
    const b = 31.83;
    const pillars = [];

    [0.5, 1.0, 1.5, 2.0, 2.5, 2.8].forEach((mult, idx) => {
        const theta = mult * Math.PI;
        const rOffset = idx < 5 ? -100 : -80;
        const r = rMax - b * theta + rOffset;
        pillars.push(new Pillar(
            centerX + r * Math.cos(theta),
            centerY + r * Math.sin(theta)
        ));
    });

    return pillars;
})(),
```

#### 2. Cars (Obstacle Cars)
```javascript
cars: (() => {
    const centerX = canvas.width / 2;
    const centerY = canvas.height / 2;
    const rMax = 350;
    const b = 31.83;

    const carConfigs = [
        { theta: Math.PI, rOffset: -100, type: 'sedan', color: '#e74c3c' },
        { theta: 2*Math.PI, rOffset: -90, type: 'compact', color: '#9b59b6' },
        { theta: 2.67*Math.PI, rOffset: -85, type: 'suv', color: '#34495e' }
    ];

    return carConfigs.map(cfg => {
        const r = rMax - b * cfg.theta + cfg.rOffset;
        const x = centerX + r * Math.cos(cfg.theta);
        const y = centerY + r * Math.sin(cfg.theta);
        const angle = (cfg.theta + Math.PI/2) * 180/Math.PI;
        return new ObstacleCar({ x, y, angle, type: cfg.type, color: cfg.color });
    });
})(),
```

#### 3. Parking Zones
```javascript
parkingZones: [
    (() => {
        const centerX = canvas.width / 2;
        const centerY = canvas.height / 2;
        return new ParkingZone({
            x: centerX,
            y: centerY,
            w: 70,
            l: 120,
            angle: 180
        });
    })()
],
```

#### 4. Curbs (Spiral Walls)
```javascript
curbs: (() => {
    const curbs = [];
    const centerX = canvas.width / 2;
    const centerY = canvas.height / 2;
    const rMax = 350;
    const totalRotation = 3 * Math.PI; // 540°
    const b = (rMax - 50) / totalRotation; // 31.83
    const rampWidth = 200;
    const curbWidth = 40;
    const curbLength = 30;
    const angleStep = Math.PI / 18; // 10° steps

    // Outer wall curbs
    for (let theta = 0; theta <= totalRotation; theta += angleStep) {
        const r = rMax - b * theta;
        const x = centerX + r * Math.cos(theta);
        const y = centerY + r * Math.sin(theta);
        const tangentAngle = theta + Math.PI / 2;
        curbs.push(new Curb(x, y, curbWidth, curbLength, tangentAngle));
    }

    // Inner wall curbs
    for (let theta = 0; theta <= totalRotation; theta += angleStep) {
        const r = rMax - b * theta - rampWidth;
        const x = centerX + r * Math.cos(theta);
        const y = centerY + r * Math.sin(theta);
        const tangentAngle = theta + Math.PI / 2;
        curbs.push(new Curb(x, y, curbWidth, curbLength, tangentAngle));
    }

    return curbs;
})()
```

## Testing & Adjustments

After implementation, test and potentially adjust:

1. **Curb gaps**: If visible gaps appear between curbs:
   - Reduce `angleStep` to π/24 (7.5° steps)
   - Or increase `curbLength` to 35-40px

2. **Difficulty tuning**:
   - **Easier**: Increase `rampWidth` to 220px, remove 1 obstacle car
   - **Harder**: Decrease `rampWidth` to 180px, add 4th car

3. **Visual smoothness**: Verify spiral appears as smooth curve, not jagged

4. **Collision detection**: Ensure player can navigate without getting stuck between curbs

## Success Criteria

- [ ] Level appears as #14 in level selection
- [ ] Player starts at outer edge facing inward
- [ ] Spiral curves smoothly from outside to center
- [ ] Curbs form continuous walls (no gaps)
- [ ] Parking zone is accessible at center
- [ ] Collision detection works correctly
- [ ] Level is challenging but completable
