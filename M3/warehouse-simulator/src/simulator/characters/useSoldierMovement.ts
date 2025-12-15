import { useState, useRef, useCallback, useMemo } from 'react';
import * as THREE from 'three';
import { PatrolRoute } from '../model/patrol-routes';
import { getWorldPosition } from '../model/warehouse-utilities';
import { WAYPOINT_PROXIMITY_THRESHOLD } from '../configuration';

export interface SoldierMovementState {
  position: THREE.Vector3;
  movementDirection: THREE.Vector3;
  updateMovement: (delta: number) => void;
}

export function useSoldierMovement(
  patrolRoute: PatrolRoute,
  speed: number
): SoldierMovementState {
  const [currentWaypointIndex, setCurrentWaypointIndex] = useState(0);

  const currentPositionRef = useRef<THREE.Vector3>(
    useMemo(() => {
      if (patrolRoute.waypoints.length === 0) {
        return new THREE.Vector3(0, 0, 0);
      }
      const firstWaypoint = patrolRoute.waypoints[0];
      const worldPos = getWorldPosition(firstWaypoint.row, firstWaypoint.col);
      return new THREE.Vector3(worldPos.x, 0, worldPos.z);
    }, [patrolRoute])
  );

  const movementDirectionRef = useRef<THREE.Vector3>(new THREE.Vector3(0, 0, 0));

  const updateMovement = useCallback(
    (delta: number) => {
      if (patrolRoute.waypoints.length === 0) return;

      const currentWaypoint = patrolRoute.waypoints[currentWaypointIndex];
      const targetWorldPos = getWorldPosition(currentWaypoint.row, currentWaypoint.col);
      const targetPosition = new THREE.Vector3(targetWorldPos.x, 0, targetWorldPos.z);

      const direction = new THREE.Vector3()
        .subVectors(targetPosition, currentPositionRef.current)
        .normalize();

      movementDirectionRef.current.copy(direction);

      const distance = currentPositionRef.current.distanceTo(targetPosition);

      if (distance < WAYPOINT_PROXIMITY_THRESHOLD) {
        const nextIndex = (currentWaypointIndex + 1) % patrolRoute.waypoints.length;
        setCurrentWaypointIndex(nextIndex);
      } else {
        const movement = direction.multiplyScalar(speed * delta * 60);
        currentPositionRef.current.add(movement);
      }
    },
    [patrolRoute, currentWaypointIndex, speed]
  );

  return {
    position: currentPositionRef.current,
    movementDirection: movementDirectionRef.current,
    updateMovement,
  };
}
