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

export interface IRect {
  readonly x: number;
  readonly y: number;
  readonly width: number;
  readonly height: number;
}

export interface IOrientedRect {
  readonly cx: number;
  readonly cy: number;
  readonly width: number;
  readonly height: number;
  readonly angle: number;
}

export type Color = [ number, number, number ];

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

export function polygonCenter(poly: Polygon): IPoint {
  const center = { x: 0, y: 0 };
  for (const p of poly) {
    center.x += p.x;
    center.y += p.y;
  }
  center.x /= poly.length;
  center.y /= poly.length;
  return center;
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

  assert(width >= height);

  return { cx, cy, width, height, angle };
}

function interpolateColor(a: Color, b: Color, x: number): Color {
  const aX = 1 - x;
  const bX = x;

  return [
    Math.round(a[0] * aX + b[0] * bX),
    Math.round(a[1] * aX + b[1] * bX),
    Math.round(a[2] * aX + b[2] * bX),
  ];
}

const RECT_COLORS: ReadonlyArray<Color> = [
  [ 255, 255, 255 ],
  [ 255, 0, 0 ],
  [ 0, 0, 255 ],
  [ 0, 255, 0 ],
];

export function rectColor(confidence: number): Color {
  const first = (confidence * (RECT_COLORS.length - 1)) | 0;
  const second = first + 1;

  if (first === RECT_COLORS.length - 1) {
    return RECT_COLORS[first];
  }

  return interpolateColor(RECT_COLORS[first], RECT_COLORS[second],
      confidence * (RECT_COLORS.length - 1) - first);
}
