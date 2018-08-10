import * as assert from 'assert';

export interface IPoint {
  readonly x: number;
  readonly y: number;
}

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

export function polygonToRect(polygon: ReadonlyArray<IPoint>): IOrientedRect {
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
  const width = Math.sqrt((diag ** 2 + disc) / 2);
  const height = Math.sqrt((diag ** 2 - disc) / 2);

  // find longest side
  const longest = [ v01, v03, v21, v23 ].map((vec) => {
    return { vec, norm: norm(vec) };
  }).sort((a, b) => {
    return b.norm - a.norm;
  }).map((entry) => entry.vec)[0];

  const horizontal = { x: 1, y: 0 };

  // Compute angle without preference for particular orientation
  const cos = Math.abs(dot(horizontal, longest) / norm(longest));

  const angle = Math.acos(cos);

  return { cx, cy, width, height, angle };
}
