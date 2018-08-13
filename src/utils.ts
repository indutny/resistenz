import * as assert from 'assert';

export interface IPoint {
  readonly x: number;
  readonly y: number;
}

export type Polygon = ReadonlyArray<IPoint>;

export interface IVector {
  readonly x: number;
  readonly y: number;
}

export interface IOrientedRect {
  readonly cx: number;
  readonly cy: number;
  readonly width: number;
  readonly height: number;
  readonly angle: number;
}

export function norm(a: IVector): number {
  return Math.sqrt(a.x ** 2 + a.y ** 2);
}

export function vector(a: IPoint, b: IPoint): IVector {
  return { x: b.x - a.x, y: b.y - a.y };
}

export function dot(a: IVector, b: IVector): number {
  return a.x * b.x + a.y * b.y;
}

export function triangleArea(a: IVector, b: IVector): number {
  return Math.abs(a.x * b.y - a.y * b.x) / 2;
}

export function rotate(p: IPoint, angle: number): IPoint {
  const cos = Math.cos(angle);
  const sin = Math.sin(angle);

  return {
    x: cos * p.x - sin * p.y,
    y: sin * p.x + cos * p.y,
  };
}

export function polygonToRect(polygon: Polygon): IOrientedRect {
  assert.strictEqual(polygon.length, 4);

  // Compute center
  let cx: number = 0;
  let cy: number = 0;
  for (const point of polygon) {
    cx += point.x;
    cy += point.y;
  }
  cx /= polygon.length;
  cy /= polygon.length;

  const v02 = vector(polygon[0], polygon[2]);
  const v13 = vector(polygon[1], polygon[3]);

  // Compute mean diagonal
  const diag = (norm(v02) + norm(v13)) / 2;

  // Calculate area of polygon
  const v01 = vector(polygon[0], polygon[1]);
  const v03 = vector(polygon[0], polygon[3]);
  const v21 = vector(polygon[2], polygon[1]);
  const v23 = vector(polygon[2], polygon[3]);

  const area = triangleArea(v01, v03) + triangleArea(v21, v23);

  // width > height
  const disc = Math.sqrt(diag ** 4 - 4 * area ** 2);
  let width = Math.sqrt((diag ** 2 + disc) / 2);
  let height = Math.sqrt((diag ** 2 - disc) / 2);

  // find longest side
  const longest = [ v01, v03, v21, v23 ].map((vec) => {
    return { vec, norm: norm(vec) };
  }).sort((a, b) => {
    return b.norm - a.norm;
  }).map((entry) => entry.vec)[0];

  // Compute angle without preference for particular orientation
  let angle = Math.atan2(longest.y, longest.x);
  if (angle < 0) {
    angle += Math.PI;
  }

  // Ensure that angle is always in 1st quadrant
  if (angle >= Math.PI / 2) {
    angle = angle - Math.PI / 2;
    const t = width;
    width = height;
    height = t;
  }

  // And is less than 45 degrees
  if (angle >= Math.PI / 4) {
    angle = Math.PI / 2 - angle;
    const t = width;
    width = height;
    height = t;
  }

  return { cx, cy, width, height, angle };
}
