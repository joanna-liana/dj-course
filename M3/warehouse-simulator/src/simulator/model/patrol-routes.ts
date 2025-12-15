import { AisleInfo } from './warehouse.types';

export interface PatrolWaypoint {
  row: number;
  col: number;
}

export interface PatrolRoute {
  id: string;
  waypoints: PatrolWaypoint[];
  isLoop: boolean;
}

interface AisleIntersection {
  row: number;
  col: number;
  horizontalAisleId: string;
  verticalAisleId: string;
}

function findAisleIntersections(aisles: AisleInfo[]): AisleIntersection[] {
  const intersections: AisleIntersection[] = [];

  const horizontalAisles = aisles.filter(a => a.type === 'HORIZONTAL');
  const verticalAisles = aisles.filter(a => a.type === 'VERTICAL');

  for (const hAisle of horizontalAisles) {
    for (const vAisle of verticalAisles) {
      for (const hTile of hAisle.tiles) {
        for (const vTile of vAisle.tiles) {
          if (hTile.row === vTile.row && hTile.col === vTile.col) {
            intersections.push({
              row: hTile.row,
              col: hTile.col,
              horizontalAisleId: hAisle.id,
              verticalAisleId: vAisle.id,
            });
          }
        }
      }
    }
  }

  return intersections;
}

function createRectangularRoute(
  topLeft: AisleIntersection,
  topRight: AisleIntersection,
  bottomRight: AisleIntersection,
  bottomLeft: AisleIntersection,
  routeId: string
): PatrolRoute {
  const waypoints: PatrolWaypoint[] = [];

  waypoints.push({ row: topLeft.row, col: topLeft.col });

  const horizontalSteps1 = Math.abs(topRight.col - topLeft.col);
  for (let i = 1; i < horizontalSteps1; i++) {
    waypoints.push({
      row: topLeft.row,
      col: topLeft.col + i * Math.sign(topRight.col - topLeft.col),
    });
  }
  waypoints.push({ row: topRight.row, col: topRight.col });

  const verticalSteps1 = Math.abs(bottomRight.row - topRight.row);
  for (let i = 1; i < verticalSteps1; i++) {
    waypoints.push({
      row: topRight.row + i * Math.sign(bottomRight.row - topRight.row),
      col: topRight.col,
    });
  }
  waypoints.push({ row: bottomRight.row, col: bottomRight.col });

  const horizontalSteps2 = Math.abs(bottomLeft.col - bottomRight.col);
  for (let i = 1; i < horizontalSteps2; i++) {
    waypoints.push({
      row: bottomRight.row,
      col: bottomRight.col + i * Math.sign(bottomLeft.col - bottomRight.col),
    });
  }
  waypoints.push({ row: bottomLeft.row, col: bottomLeft.col });

  const verticalSteps2 = Math.abs(topLeft.row - bottomLeft.row);
  for (let i = 1; i < verticalSteps2; i++) {
    waypoints.push({
      row: bottomLeft.row + i * Math.sign(topLeft.row - bottomLeft.row),
      col: bottomLeft.col,
    });
  }

  return {
    id: routeId,
    waypoints,
    isLoop: true,
  };
}

export function generatePatrolRoutes(
  aisles: AisleInfo[],
  count: number
): PatrolRoute[] {
  const intersections = findAisleIntersections(aisles);

  if (intersections.length < 4) {
    return [];
  }

  const routes: PatrolRoute[] = [];
  const usedIntersections = new Set<string>();

  const horizontalRows = [...new Set(intersections.map(i => i.row))].sort((a, b) => a - b);
  const verticalCols = [...new Set(intersections.map(i => i.col))].sort((a, b) => a - b);

  let routeCounter = 0;

  for (let hIdx = 0; hIdx < horizontalRows.length - 1 && routes.length < count; hIdx++) {
    for (let vIdx = 0; vIdx < verticalCols.length - 1 && routes.length < count; vIdx++) {
      const topRow = horizontalRows[hIdx];
      const bottomRow = horizontalRows[hIdx + 1];
      const leftCol = verticalCols[vIdx];
      const rightCol = verticalCols[vIdx + 1];

      const topLeft = intersections.find(i => i.row === topRow && i.col === leftCol);
      const topRight = intersections.find(i => i.row === topRow && i.col === rightCol);
      const bottomRight = intersections.find(i => i.row === bottomRow && i.col === rightCol);
      const bottomLeft = intersections.find(i => i.row === bottomRow && i.col === leftCol);

      if (topLeft && topRight && bottomRight && bottomLeft) {
        const intersectionKey = `${topRow}-${bottomRow}-${leftCol}-${rightCol}`;

        if (!usedIntersections.has(intersectionKey)) {
          usedIntersections.add(intersectionKey);

          const route = createRectangularRoute(
            topLeft,
            topRight,
            bottomRight,
            bottomLeft,
            `patrol-${routeCounter++}`
          );
          routes.push(route);
        }
      }
    }
  }

  if (routes.length < count) {
    for (let hIdx = 0; hIdx < horizontalRows.length - 2 && routes.length < count; hIdx++) {
      for (let vIdx = 0; vIdx < verticalCols.length - 2 && routes.length < count; vIdx++) {
        const topRow = horizontalRows[hIdx];
        const bottomRow = horizontalRows[hIdx + 2];
        const leftCol = verticalCols[vIdx];
        const rightCol = verticalCols[vIdx + 2];

        const topLeft = intersections.find(i => i.row === topRow && i.col === leftCol);
        const topRight = intersections.find(i => i.row === topRow && i.col === rightCol);
        const bottomRight = intersections.find(i => i.row === bottomRow && i.col === rightCol);
        const bottomLeft = intersections.find(i => i.row === bottomRow && i.col === leftCol);

        if (topLeft && topRight && bottomRight && bottomLeft) {
          const intersectionKey = `${topRow}-${bottomRow}-${leftCol}-${rightCol}`;

          if (!usedIntersections.has(intersectionKey)) {
            usedIntersections.add(intersectionKey);

            const route = createRectangularRoute(
              topLeft,
              topRight,
              bottomRight,
              bottomLeft,
              `patrol-${routeCounter++}`
            );
            routes.push(route);
          }
        }
      }
    }
  }

  return routes.slice(0, count);
}
