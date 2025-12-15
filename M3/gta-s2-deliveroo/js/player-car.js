class PlayerCar {
    constructor(x, y, angleDeg, vehicleType = 'sedan') {
        this.vehicleType = vehicleType;
        this.reset(x, y, angleDeg);
    }

    reset(x, y, angleDeg) {
        // Apply vehicle-specific properties
        const vType = VEHICLE_TYPES[this.vehicleType];

        this.x = x;
        this.y = y;
        this.angle = angleDeg * (Math.PI / 180);
        this.speed = 0;

        // Wektor prędkości dla zaawansowanej fizyki
        this.velocityX = 0;
        this.velocityY = 0;
        this.angularVelocity = 0; // Prędkość rotacji

        this.steeringAngle = 0;
        this.w = vType.width;
        this.l = vType.length;
        this.wheelBase = vType.wheelBase;
        this.vehicleMaxSpeed = vType.maxSpeed;
        this.vehicleAcceleration = vType.acceleration;
        this.vehicleMaxSteerAngle = vType.maxSteerAngle;
        this.vehicleBrakingForce = vType.brakingForce;
        this.vehicleTireGrip = vType.tireGrip;
        this.vehicleTurboMultiplier = vType.turboMultiplier;
        this.engineOn = true;
        this.enterKeyProcessed = false;
        this.steeringMode = 'DRIVING';

        // Tryb zimowy - domyślnie wyłączony (bezpieczna jazda)
        if (this.winterMode === undefined) {
            this.winterMode = false;
        }

        // Stan poślizgu
        this.isDrifting = false;
        this.driftAngle = 0; // Kąt poślizgu
        this.skidMarks = []; // Ślady opon

        // Hamulec ręczny - startowanie
        this.handbrakeBoost = 0; // Zgromadzona moc (0-1)
        this.previousSpaceKey = false; // Czy w poprzedniej klatce trzymał SPACE

        // Turbo system
        this.isTurboActive = false;
        this.turboParticles = [];
    }

    toggleSteeringMode() {
        if (this.steeringMode === 'DRIVING') {
            this.steeringMode = 'PARKING';
            document.getElementById('toggle-steering-mode').innerText = 'Asystent Kierownicy: WYŁ';
        } else {
            this.steeringMode = 'DRIVING';
            document.getElementById('toggle-steering-mode').innerText = 'Asystent Kierownicy: WŁ';
        }
    }

    toggleWinterMode() {
        this.winterMode = !this.winterMode;
        const btn = document.getElementById('toggle-winter-mode');
        if (this.winterMode) {
            btn.innerText = 'Poślizgi Zimowe: WŁ';
        } else {
            btn.innerText = 'Poślizgi Zimowe: WYŁ';
            // Wyczyść ślady opon przy wyłączeniu trybu zimowego
            this.skidMarks = [];
            this.isDrifting = false;
            // Zatrzymaj dźwięk poślizgu
            if (driftOscillator) {
                stopDriftSound();
            }
        }
    }

    update(input, deltaTime) {
        // Engine toggle
        if (input.keys.Enter) {
            if (!this.enterKeyProcessed) {
                this.engineOn = !this.engineOn;
                this.enterKeyProcessed = true;
            }
        } else {
            this.enterKeyProcessed = false;
        }

        // Wybierz fizykę w zależności od trybu
        if (this.winterMode) {
            this.updateWinterPhysics(input, deltaTime);
        } else {
            this.updateSimplePhysics(input, deltaTime);
        }
    }

    // === PROSTA FIZYKA (bezpieczna, przewidywalna) ===
    updateSimplePhysics(input, deltaTime) {
        // Normalize deltaTime to 60 FPS (deltaTime * 60 gives us a frame multiplier)
        const dt = deltaTime * 60;

        // === TURBO SYSTEM ===
        this.isTurboActive = input.keys.ShiftLeft || input.keys.ShiftRight;
        const turboBoost = this.isTurboActive ? this.vehicleTurboMultiplier : 1.0;
        const effectiveMaxSpeed = this.vehicleMaxSpeed * turboBoost;
        const effectiveAccel = this.vehicleAcceleration * turboBoost;

        // Spawn turbo particles
        if (this.isTurboActive && Math.abs(this.speed) > 2) {
            this.turboParticles.push({
                x: this.x - Math.cos(this.angle) * this.l / 2,
                y: this.y - Math.sin(this.angle) * this.l / 2,
                life: 1.0,
                angle: this.angle + Math.PI
            });
        }

        // Update turbo particles
        this.turboParticles = this.turboParticles.filter(p => {
            p.life -= deltaTime * 2;
            return p.life > 0;
        });

        // Turbo sound control
        if (this.isTurboActive && Math.abs(this.speed) > 1) {
            if (!turboOscillator1) {
                startTurboSound();
            } else {
                updateTurboSound(this.speed);
            }
        } else if (turboOscillator1) {
            stopTurboSound();
        }

        // === HAMULEC RĘCZNY - STARTOWANIE ===
        const isHandbraking = input.keys.Space;
        const isThrottling = input.keys.ArrowUp || input.keys.ArrowDown;

        // Budowanie boost gdy trzyma hamulec + gaz
        if (isHandbraking && isThrottling && this.engineOn) {
            this.handbrakeBoost = Math.min(CONFIG.handbrakeBoostMax, this.handbrakeBoost + CONFIG.handbrakeBoostRate * dt);

            // Hamuj auto podczas budowania boost
            this.speed *= Math.pow(0.8, dt); // Mocne hamowanie
            if (Math.abs(this.speed) < 0.5) this.speed = 0;

            // Dźwięk silnika na wysokich obrotach
            if (!engineRevOscillator) {
                startEngineRevSound(this.handbrakeBoost);
            } else {
                updateEngineRevSound(this.handbrakeBoost);
            }
        }
        // Jeśli puścił hamulec (ale dalej trzyma gaz) - LAUNCH!
        else if (!isHandbraking && this.previousSpaceKey && isThrottling && this.handbrakeBoost > 0.1) {
            // MOCNY START!
            const boostDirection = input.keys.ArrowUp ? 1 : -1;
            this.speed += boostDirection * this.handbrakeBoost * CONFIG.handbrakeBoostMultiplier;
            this.handbrakeBoost = 0; // Zużyte!

            // Zatrzymaj dźwięk silnika
            if (engineRevOscillator) {
                stopEngineRevSound();
            }
        }
        // Normalne zmniejszanie boost gdy nie używany
        else if (this.handbrakeBoost > 0) {
            this.handbrakeBoost = Math.max(0, this.handbrakeBoost - CONFIG.handbrakeBoostDecay * dt);

            // Zatrzymaj dźwięk gdy boost spada
            if (this.handbrakeBoost < 0.1 && engineRevOscillator) {
                stopEngineRevSound();
            }
        }

        this.previousSpaceKey = isHandbraking;

        // === NORMALNA FIZYKA ===
        if (this.engineOn) {
            // 1. Acceleration (tylko jeśli NIE buduje boost)
            if (!(isHandbraking && isThrottling)) {
                if (input.keys.ArrowUp) this.speed += effectiveAccel * dt;
                else if (input.keys.ArrowDown) this.speed -= effectiveAccel * dt;
            }

            // 2. Braking (tylko jeśli NIE trzyma gazu równocześnie)
            if (input.keys.Space && !isThrottling) {
                if (this.speed > 0) this.speed -= this.vehicleBrakingForce * dt;
                else if (this.speed < 0) this.speed += this.vehicleBrakingForce * dt;
                if (Math.abs(this.speed) < 0.5) this.speed = 0;
            }
        }

        // 3. Friction
        if (!input.keys.ArrowUp && !input.keys.ArrowDown && !input.keys.Space) {
            this.speed *= Math.pow(1 - CONFIG.friction, dt);
            if (Math.abs(this.speed) < 0.05) this.speed = 0;
        }
        if (!this.engineOn) {
            this.speed *= Math.pow(1 - CONFIG.friction, dt);
            if (Math.abs(this.speed) < 0.05) this.speed = 0;
        }

        // Limits
        if (this.speed > effectiveMaxSpeed) this.speed = effectiveMaxSpeed;
        if (this.speed < CONFIG.maxReverseSpeed) this.speed = CONFIG.maxReverseSpeed;

        // 4. Steering
        if (this.engineOn) {
            if (input.keys.ArrowLeft) {
                this.steeringAngle -= CONFIG.steerSpeed * dt;
            } else if (input.keys.ArrowRight) {
                this.steeringAngle += CONFIG.steerSpeed * dt;
            } else {
                if (this.steeringMode === 'DRIVING') {
                    // Auto-straighten in Driving Mode
                    if (this.steeringAngle > 0) {
                        this.steeringAngle -= CONFIG.steerRestoringDriving * dt;
                        if (this.steeringAngle < 0) this.steeringAngle = 0;
                    } else if (this.steeringAngle < 0) {
                        this.steeringAngle += CONFIG.steerRestoringDriving * dt;
                        if (this.steeringAngle > 0) this.steeringAngle = 0;
                    }
                }
            }
        }

        // Clamp steer
        if (this.steeringAngle > this.vehicleMaxSteerAngle) this.steeringAngle = this.vehicleMaxSteerAngle;
        if (this.steeringAngle < -this.vehicleMaxSteerAngle) this.steeringAngle = -this.vehicleMaxSteerAngle;

        // 5. Movement - prosty model kinematyczny
        if (Math.abs(this.speed) > 0.05) {
            const L = this.wheelBase;
            const oldAngle = this.angle;

            this.angle += (this.speed / L) * Math.tan(this.steeringAngle) * dt;

            const rearAxleX = this.x - (L / 2) * Math.cos(oldAngle);
            const rearAxleY = this.y - (L / 2) * Math.sin(oldAngle);

            const newRearAxleX = rearAxleX + this.speed * Math.cos(oldAngle) * dt;
            const newRearAxleY = rearAxleY + this.speed * Math.sin(oldAngle) * dt;

            this.x = newRearAxleX + (L / 2) * Math.cos(this.angle);
            this.y = newRearAxleY + (L / 2) * Math.sin(this.angle);
        } else {
            this.x += Math.cos(this.angle) * this.speed * dt;
            this.y += Math.sin(this.angle) * this.speed * dt;
        }

        // Synchronizuj velocityX/Y dla kompatybilności
        this.velocityX = Math.cos(this.angle) * this.speed;
        this.velocityY = Math.sin(this.angle) * this.speed;
        this.angularVelocity = 0;
        this.isDrifting = false;
        this.driftAngle = 0;
    }

    // === ZAAWANSOWANA FIZYKA Z POŚLIZGAMI (tryb zimowy) ===
    updateWinterPhysics(input, deltaTime) {
        // Normalize deltaTime to 60 FPS (deltaTime * 60 gives us a frame multiplier)
        const dt = deltaTime * 60;

        // === TURBO SYSTEM ===
        this.isTurboActive = input.keys.ShiftLeft || input.keys.ShiftRight;
        const turboBoost = this.isTurboActive ? this.vehicleTurboMultiplier : 1.0;
        const effectiveMaxSpeed = this.vehicleMaxSpeed * turboBoost;
        const effectiveAccel = this.vehicleAcceleration * turboBoost;

        // Spawn turbo particles
        const currentSpeed = Math.sqrt(this.velocityX * this.velocityX + this.velocityY * this.velocityY);
        if (this.isTurboActive && currentSpeed > 2) {
            this.turboParticles.push({
                x: this.x - Math.cos(this.angle) * this.l / 2,
                y: this.y - Math.sin(this.angle) * this.l / 2,
                life: 1.0,
                angle: this.angle + Math.PI
            });
        }

        // Update turbo particles
        this.turboParticles = this.turboParticles.filter(p => {
            p.life -= deltaTime * 2;
            return p.life > 0;
        });

        // Turbo sound control
        if (this.isTurboActive && Math.abs(this.speed) > 1) {
            if (!turboOscillator1) {
                startTurboSound();
            } else {
                updateTurboSound(this.speed);
            }
        } else if (turboOscillator1) {
            stopTurboSound();
        }

        // === HAMULEC RĘCZNY - STARTOWANIE ===
        const isHandbraking = input.keys.Space;
        const isThrottling = input.keys.ArrowUp || input.keys.ArrowDown;

        // Budowanie boost gdy trzyma hamulec + gaz
        if (isHandbraking && isThrottling && this.engineOn) {
            this.handbrakeBoost = Math.min(CONFIG.handbrakeBoostMax, this.handbrakeBoost + CONFIG.handbrakeBoostRate * dt);

            // Hamuj auto podczas budowania boost
            this.velocityX *= Math.pow(0.75, dt);
            this.velocityY *= Math.pow(0.75, dt);
            const currentSpeed = Math.sqrt(this.velocityX * this.velocityX + this.velocityY * this.velocityY);
            if (currentSpeed < 0.5) {
                this.velocityX = 0;
                this.velocityY = 0;
            }

            // Dźwięk silnika na wysokich obrotach
            if (!engineRevOscillator) {
                startEngineRevSound(this.handbrakeBoost);
            } else {
                updateEngineRevSound(this.handbrakeBoost);
            }
        }
        // Jeśli puścił hamulec (ale dalej trzyma gaz) - LAUNCH!
        else if (!isHandbraking && this.previousSpaceKey && isThrottling && this.handbrakeBoost > 0.1) {
            // MOCNY START!
            const boostDirection = input.keys.ArrowUp ? 1 : -1;
            const boostPower = boostDirection * this.handbrakeBoost * CONFIG.handbrakeBoostMultiplier;

            // Dodaj boost w kierunku samochodu
            this.velocityX += Math.cos(this.angle) * boostPower;
            this.velocityY += Math.sin(this.angle) * boostPower;

            this.handbrakeBoost = 0; // Zużyte!

            // Zatrzymaj dźwięk silnika
            if (engineRevOscillator) {
                stopEngineRevSound();
            }
        }
        // Normalne zmniejszanie boost gdy nie używany
        else if (this.handbrakeBoost > 0) {
            this.handbrakeBoost = Math.max(0, this.handbrakeBoost - CONFIG.handbrakeBoostDecay * dt);

            // Zatrzymaj dźwięk gdy boost spada
            if (this.handbrakeBoost < 0.1 && engineRevOscillator) {
                stopEngineRevSound();
            }
        }

        this.previousSpaceKey = isHandbraking;

        // 1. Sterowanie - kąt skrętu
        if (this.engineOn) {
            if (input.keys.ArrowLeft) {
                this.steeringAngle -= CONFIG.steerSpeed * dt;
            } else if (input.keys.ArrowRight) {
                this.steeringAngle += CONFIG.steerSpeed * dt;
            } else {
                if (this.steeringMode === 'DRIVING') {
                    // Auto-prostowanie w trybie jazdy
                    if (this.steeringAngle > 0) {
                        this.steeringAngle -= CONFIG.steerRestoringDriving * dt;
                        if (this.steeringAngle < 0) this.steeringAngle = 0;
                    } else if (this.steeringAngle < 0) {
                        this.steeringAngle += CONFIG.steerRestoringDriving * dt;
                        if (this.steeringAngle > 0) this.steeringAngle = 0;
                    }
                }
            }
        }

        // Ogranicz kąt skrętu - zawsze maksymalny, niezależnie od prędkości
        // Fizyka zadba o poślizg przy dużych prędkościach!
        this.steeringAngle = Math.max(-this.vehicleMaxSteerAngle, Math.min(this.vehicleMaxSteerAngle, this.steeringAngle));

        // 2. Akceleracja i hamowanie
        const isBraking = input.keys.Space && !isThrottling; // Hamowanie tylko bez gazu
        let throttle = 0;

        // Akceleracja tylko jeśli NIE buduje boost (hamulec + gaz)
        if (this.engineOn && !(isHandbraking && isThrottling)) {
            if (input.keys.ArrowUp) throttle = effectiveAccel * dt;
            else if (input.keys.ArrowDown) throttle = -effectiveAccel * dt;
        }

        // 3. Oblicz prędkość w lokalnym układzie samochodu (forward/lateral)
        const cos = Math.cos(this.angle);
        const sin = Math.sin(this.angle);

        // Prędkość w kierunku "do przodu" i "na boki" względem auta
        const forwardVelocity = this.velocityX * cos + this.velocityY * sin;
        const lateralVelocity = -this.velocityX * sin + this.velocityY * cos;

        // 4. Zastosuj akcelerację do przodu
        let newForwardVelocity = forwardVelocity + throttle;

        // 5. Oblicz siłę boczną z powodu skrętu kół
        // FIZYKA: Siła odśrodkowa F = m*v²/r, więc rośnie KWADRATOWO z prędkością!
        const baseLateralVelocity = newForwardVelocity * Math.tan(this.steeringAngle);

        // Dodatkowy mnożnik dla dużych prędkości (symuluje v² efekt)
        const speedMagnitude = Math.sqrt(this.velocityX * this.velocityX + this.velocityY * this.velocityY);
        const speedSquaredFactor = 1.0 + (speedMagnitude / effectiveMaxSpeed) * CONFIG.lateralForceMultiplier;

        const desiredLateralVelocity = baseLateralVelocity * speedSquaredFactor;

        // 6. Określ przyczepność opon (grip)
        let currentGrip = isBraking ? CONFIG.tireGripBraking : this.vehicleTireGrip;

        // 7. Sprawdź warunek poślizgu
        const lateralChange = desiredLateralVelocity - lateralVelocity;

        // Jeśli zmiana prędkości bocznej jest zbyt duża = poślizg!
        const lateralAcceleration = Math.abs(lateralChange);

        // Przyczepność rośnie tylko liniowo z prędkością (nie kwadratowo!)
        // To sprawia że przy dużych prędkościach łatwo przekroczyć limit
        const maxGrip = currentGrip * Math.abs(newForwardVelocity);

        if (lateralAcceleration > maxGrip && speedMagnitude > CONFIG.driftThreshold) {
            // POŚLIZG!
            this.isDrifting = true;

            // Ograniczona zmiana prędkości bocznej - opony nie nadążają
            const actualLateralChange = Math.sign(lateralChange) * maxGrip;
            const newLateralVelocity = lateralVelocity + actualLateralChange;

            // Kąt poślizgu
            this.driftAngle = Math.atan2(newLateralVelocity, newForwardVelocity);

            // Podczas poślizgu - wolniejsza rotacja
            this.angularVelocity = (newForwardVelocity / this.wheelBase) * Math.tan(this.steeringAngle) * currentGrip * dt;

            // Zastosuj tarcie podczas poślizgu
            newForwardVelocity *= Math.pow(CONFIG.driftFriction, dt);

            // Konwersja z powrotem do współrzędnych globalnych
            this.velocityX = newForwardVelocity * cos - newLateralVelocity * sin;
            this.velocityY = newForwardVelocity * sin + newLateralVelocity * cos;

            // Dodaj ślad opon podczas poślizgu
            if (Math.abs(this.driftAngle) > 0.15) { // Minimum kąt dla śladów
                this.addSkidMark();
            }

            // Dźwięk piszczących opon - intensywność zależy od kąta poślizgu
            const driftIntensity = Math.min(1.0, Math.abs(this.driftAngle) / 0.5);
            if (!driftOscillator) {
                startDriftSound(driftIntensity);
            } else {
                updateDriftSound(driftIntensity);
            }
        } else {
            // Normalna jazda - pełna przyczepność
            this.isDrifting = false;
            this.driftAngle = 0;

            const newLateralVelocity = desiredLateralVelocity;

            // Normalna rotacja
            this.angularVelocity = (newForwardVelocity / this.wheelBase) * Math.tan(this.steeringAngle) * dt;

            // Konwersja z powrotem do współrzędnych globalnych
            this.velocityX = newForwardVelocity * cos - newLateralVelocity * sin;
            this.velocityY = newForwardVelocity * sin + newLateralVelocity * cos;

            // Zatrzymaj dźwięk poślizgu
            if (driftOscillator) {
                stopDriftSound();
            }
        }

        // 8. Hamowanie
        if (isBraking) {
            const brakingDeceleration = this.vehicleBrakingForce * dt;
            const currentSpeed = Math.sqrt(this.velocityX * this.velocityX + this.velocityY * this.velocityY);

            if (currentSpeed > 0.1) {
                const brakeMultiplier = Math.max(0, (currentSpeed - brakingDeceleration) / currentSpeed);
                this.velocityX *= brakeMultiplier;
                this.velocityY *= brakeMultiplier;
            } else {
                this.velocityX = 0;
                this.velocityY = 0;
            }
        }

        // 9. Tarcie naturalne
        if (!input.keys.ArrowUp && !input.keys.ArrowDown && !isBraking) {
            this.velocityX *= Math.pow(1 - CONFIG.friction, dt);
            this.velocityY *= Math.pow(1 - CONFIG.friction, dt);
        }

        if (!this.engineOn) {
            this.velocityX *= Math.pow(1 - CONFIG.friction, dt);
            this.velocityY *= Math.pow(1 - CONFIG.friction, dt);
        }

        // Zatrzymaj jeśli bardzo wolno
        const finalSpeed = Math.sqrt(this.velocityX * this.velocityX + this.velocityY * this.velocityY);
        if (finalSpeed < 0.05) {
            this.velocityX = 0;
            this.velocityY = 0;
            this.angularVelocity = 0;
        }

        // 10. Ogranicz maksymalną prędkość
        if (finalSpeed > effectiveMaxSpeed) {
            const ratio = effectiveMaxSpeed / finalSpeed;
            this.velocityX *= ratio;
            this.velocityY *= ratio;
        }

        // 11. Aktualizuj rotację
        this.angle += this.angularVelocity;
        this.angularVelocity *= Math.pow(CONFIG.angularDamping, dt);

        // 12. Aktualizuj pozycję
        this.x += this.velocityX * dt;
        this.y += this.velocityY * dt;

        // 13. Aktualizuj zmienną speed dla kompatybilności
        this.speed = Math.sqrt(this.velocityX * this.velocityX + this.velocityY * this.velocityY) *
                     Math.sign(Math.cos(this.angle) * this.velocityX + Math.sin(this.angle) * this.velocityY);

        // 14. Zarządzaj śladami opon (max 200 punktów)
        if (this.skidMarks.length > 200) {
            this.skidMarks.shift();
        }
    }

    addSkidMark() {
        // Dodaj ślad pod tylnymi kołami
        const rearAxleOffset = -CONFIG.wheelBase / 2;
        const wheelOffset = CONFIG.carWidth / 3;

        const cos = Math.cos(this.angle);
        const sin = Math.sin(this.angle);

        // Lewe tylne koło
        const leftX = this.x + (rearAxleOffset * cos - wheelOffset * sin);
        const leftY = this.y + (rearAxleOffset * sin + wheelOffset * cos);

        // Prawe tylne koło
        const rightX = this.x + (rearAxleOffset * cos + wheelOffset * sin);
        const rightY = this.y + (rearAxleOffset * sin - wheelOffset * cos);

        this.skidMarks.push({ x: leftX, y: leftY, angle: this.angle, alpha: 1.0 });
        this.skidMarks.push({ x: rightX, y: rightY, angle: this.angle, alpha: 1.0 });
    }

    drawSkidMarks(ctx) {
        // Rysuj ślady opon
        ctx.save();
        ctx.strokeStyle = 'rgba(30, 30, 30, 0.7)';
        ctx.lineWidth = 3;
        ctx.lineCap = 'round';

        for (let i = 1; i < this.skidMarks.length; i++) {
            const prev = this.skidMarks[i - 1];
            const curr = this.skidMarks[i];

            // Zanikaj starsze ślady
            const fadeIndex = Math.max(0, this.skidMarks.length - 150);
            const alpha = i < fadeIndex ? 0.3 : 0.7;

            ctx.globalAlpha = alpha;
            ctx.beginPath();
            ctx.moveTo(prev.x, prev.y);
            ctx.lineTo(curr.x, curr.y);
            ctx.stroke();
        }

        ctx.restore();
    }

    draw(ctx) {
        ctx.save();
        ctx.translate(this.x, this.y);
        ctx.rotate(this.angle);

        // Scale wheels and lights proportionally to vehicle size
        const vehicleScale = this.w / CONFIG.carWidth;
        const wheelWidth = CONFIG.wheelWidth * vehicleScale;
        const wheelLength = CONFIG.wheelLength * vehicleScale;

        // Symmetrical positions for wheels and lights
        const wheelCenterY = this.w / 3;
        const wheelTopY_L = -wheelCenterY - wheelWidth / 2;
        const wheelTopY_R = wheelCenterY - wheelWidth / 2;

        const headlightCenterY = this.w / 3.5;
        const headlightHeight = 10 * vehicleScale;
        const headlightTopY_L = -headlightCenterY - headlightHeight / 2;
        const headlightTopY_R = headlightCenterY - headlightHeight / 2;


        // Draw Projection (Trajectory)
        // Draw faintly where the car is going
        if (this.engineOn && Math.abs(this.steeringAngle) > 0.05) {
            ctx.save();
            ctx.strokeStyle = 'rgba(255, 255, 0, 0.4)';
            ctx.lineWidth = 2;
            ctx.setLineDash([5, 5]);
            ctx.beginPath();

            const steer = this.steeringAngle;
            const wx = this.wheelBase/2;

            // Left Wheel projection
            let wy_L = -wheelCenterY;
            ctx.moveTo(wx, wy_L);
            ctx.lineTo(wx + Math.cos(steer)*100, wy_L + Math.sin(steer)*100);

            // Right Wheel projection
            let wy_R = wheelCenterY;
            ctx.moveTo(wx, wy_R);
            ctx.lineTo(wx + Math.cos(steer)*100, wy_R + Math.sin(steer)*100);

            ctx.stroke();
            ctx.restore();
        }

        // --- WHEELS ---
        ctx.fillStyle = '#222';
        // Rear
        this.drawWheel(ctx, -this.wheelBase/2, wheelTopY_L, 0, wheelLength, wheelWidth);
        this.drawWheel(ctx, -this.wheelBase/2, wheelTopY_R, 0, wheelLength, wheelWidth);
        // Front
        this.drawWheel(ctx, this.wheelBase/2, wheelTopY_L, this.steeringAngle, wheelLength, wheelWidth);
        this.drawWheel(ctx, this.wheelBase/2, wheelTopY_R, this.steeringAngle, wheelLength, wheelWidth);

        // --- BODY --- (Different colors and shapes per vehicle type)
        let bodyColor, roofColor, strokeColor;

        switch(this.vehicleType) {
            case 'sedan':
                bodyColor = '#3498db'; // Blue
                roofColor = '#85c1e9'; // Light blue
                strokeColor = '#2980b9';
                break;
            case 'suv':
                bodyColor = '#5d4037'; // Brown
                roofColor = '#795548'; // Lighter brown
                strokeColor = '#3e2723';
                break;
            case 'compact':
                bodyColor = '#27ae60'; // Green
                roofColor = '#52c77a'; // Light green
                strokeColor = '#1e8449';
                break;
            case 'sports':
                bodyColor = '#e74c3c'; // Red
                roofColor = '#c0392b'; // Dark red (low profile)
                strokeColor = '#a93226';
                break;
            case 'bike':
                bodyColor = '#34495e'; // Dark gray
                roofColor = '#e67e22'; // Orange accent
                strokeColor = '#2c3e50';
                break;
            case 'scooter':
                bodyColor = '#f39c12'; // Orange/yellow
                roofColor = '#f1c40f'; // Bright yellow
                strokeColor = '#d68910';
                break;
            default:
                bodyColor = '#3498db';
                roofColor = '#85c1e9';
                strokeColor = '#2980b9';
        }

        ctx.fillStyle = bodyColor;
        ctx.beginPath();

        // Different shapes per vehicle
        if (this.vehicleType === 'bike' || this.vehicleType === 'scooter') {
            // Bikes/scooters: slim body, no roof
            ctx.roundRect(-this.l/2, -this.w/4, this.l, this.w/2, 3 * vehicleScale);
        } else if (this.vehicleType === 'sports') {
            // Sports car: lower, sleeker
            ctx.roundRect(-this.l/2, -this.w/2, this.l, this.w, 8 * vehicleScale);
        } else if (this.vehicleType === 'suv') {
            // SUV: taller, boxier
            ctx.roundRect(-this.l/2, -this.w/2, this.l, this.w, 4 * vehicleScale);
        } else {
            // Sedan/compact: normal
            ctx.roundRect(-this.l/2, -this.w/2, this.l, this.w, 6 * vehicleScale);
        }

        ctx.fill();
        ctx.strokeStyle = strokeColor;
        ctx.lineWidth = 2;
        ctx.stroke();

        // Roof (not for bikes/scooters)
        if (this.vehicleType !== 'bike' && this.vehicleType !== 'scooter') {
            ctx.fillStyle = roofColor;
            ctx.beginPath();

            if (this.vehicleType === 'sports') {
                // Sports: smaller, lower roof
                ctx.roundRect(-this.l/5, -this.w/2 + 4 * vehicleScale, this.l/3, this.w - 8 * vehicleScale, 2 * vehicleScale);
            } else if (this.vehicleType === 'suv') {
                // SUV: larger roof
                ctx.roundRect(-this.l/3, -this.w/2 + 4 * vehicleScale, this.l/1.8, this.w - 8 * vehicleScale, 2 * vehicleScale);
            } else {
                // Sedan/compact: normal roof
                ctx.roundRect(-this.l/4, -this.w/2 + 6 * vehicleScale, this.l/2, this.w - 12 * vehicleScale, 3 * vehicleScale);
            }
            ctx.fill();

            // Windshield indication (Front)
            ctx.fillStyle = 'rgba(255,255,255,0.3)';
            ctx.fillRect(this.l/4, -this.w/2 + 7 * vehicleScale, 5 * vehicleScale, this.w - 14 * vehicleScale);
        } else {
            // Bikes/scooters: Add seat/handlebar accent
            ctx.fillStyle = roofColor;
            ctx.beginPath();
            ctx.arc(0, 0, 3 * vehicleScale, 0, Math.PI * 2);
            ctx.fill();
        }

        if (this.engineOn) {
            // Lights
            const isReversing = this.speed < -0.1; // Cofanie
            const isBraking = input.keys.Space || (this.speed > 0 && input.keys.ArrowDown) || (this.speed < 0 && input.keys.ArrowUp) || isReversing;

            // Brake Lights (also light up when reversing)
            const lightWidth = 3 * vehicleScale;
            ctx.fillStyle = isBraking ? '#ff0000' : '#8b0000';
            if(isBraking) { ctx.shadowColor = '#f00'; ctx.shadowBlur = 15; }
            ctx.beginPath();
            ctx.rect(-this.l/2, headlightTopY_L, lightWidth, headlightHeight);
            ctx.rect(-this.l/2, headlightTopY_R, lightWidth, headlightHeight);
            ctx.fill();
            ctx.shadowBlur = 0;

            // Headlights (Beams always on if engine is on)
            ctx.fillStyle = '#f1c40f';
            { // Removed if (this.speed > 0.5)
                 ctx.save();
                 ctx.globalCompositeOperation = 'screen';
                 ctx.fillStyle = 'rgba(255, 255, 200, 0.2)';
                 const beamLength = 150;
                 const beamSpread = 60;
                 ctx.beginPath();
                 ctx.moveTo(this.l/2, headlightTopY_L + headlightHeight/2);
                 ctx.lineTo(this.l/2 + beamLength, (headlightTopY_L + headlightHeight/2) - beamSpread);
                 ctx.lineTo(this.l/2 + beamLength, (headlightTopY_R + headlightHeight/2) + beamSpread);
                 ctx.lineTo(this.l/2, headlightTopY_R + headlightHeight/2);
                 ctx.fill();
                 ctx.restore();
                 ctx.fillStyle = '#fff'; // Bright core
            }
            ctx.beginPath();
            ctx.rect(this.l/2 - 2 * vehicleScale, headlightTopY_L, 2 * vehicleScale, headlightHeight);
            ctx.rect(this.l/2 - 2 * vehicleScale, headlightTopY_R, 2 * vehicleScale, headlightHeight);
            ctx.fill();
        }

        // --- DELIVEROO TEXT ON ROOF ---
        ctx.save();
        ctx.fillStyle = '#000000'; // Black text
        const fontSize = Math.max(8, 12 * vehicleScale);
        ctx.font = `bold ${fontSize}px Arial`;
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        // Draw text along the roof (from back to front)
        ctx.fillText('DELIVEROO', 0, 0);
        ctx.restore();

        ctx.restore();

        // Draw turbo particles (in world space, not vehicle space)
        this.drawTurboEffects(ctx);
    }

    drawTurboEffects(ctx) {
        this.turboParticles.forEach(p => {
            ctx.save();
            ctx.globalAlpha = p.life * 0.6;
            ctx.fillStyle = '#FFD700';
            ctx.beginPath();
            ctx.arc(p.x, p.y, 5 * p.life, 0, Math.PI * 2);
            ctx.fill();
            ctx.restore();
        });
    }

    drawWheel(ctx, x, y, angle, wheelLength = CONFIG.wheelLength, wheelWidth = CONFIG.wheelWidth) {
        ctx.save();
        ctx.translate(x, y);
        ctx.rotate(angle);
        // Tire tread
        ctx.fillStyle = '#1a1a1a';
        ctx.fillRect(-wheelLength/2, 0, wheelLength, wheelWidth);
        // Rim highlight (scale with wheel)
        const rimSize = Math.max(2, wheelLength / 5);
        ctx.fillStyle = '#555';
        ctx.fillRect(-rimSize/2, wheelWidth/4, rimSize, wheelWidth/2);
        ctx.restore();
    }
}
