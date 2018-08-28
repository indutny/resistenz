//
//  OrientedRect.swift
//  Resistenz
//
//  Created by Indutnyy, Fedor on 8/26/18.
//  Copyright © 2018 Fedor Indutny. All rights reserved.
//

import Foundation
import UIKit

class OrientedRect {
  public let center: CGPoint
  public let size: CGSize
  public let angle: CGFloat
  public let confidence: CGFloat

  init(cx: Float, cy: Float, width: Float, height: Float, angle: Float,
       confidence: Float) {
    center = CGPoint(x: CGFloat(cx), y: CGFloat(cy))
    size = CGSize(width: CGFloat(width), height: CGFloat(height))
    self.angle = CGFloat(angle)
    self.confidence = CGFloat(confidence)
  }

  func toPath(scale: CGFloat) -> CGPath {
    // NOTE: The points are displayed upside down, thus negative angle
    var transform = CGAffineTransform(rotationAngle: -angle)
    let scaledSize = size.applying(CGAffineTransform(scaleX: scale, y: scale))
    let origin = CGPoint(x: -scaledSize.width / 2, y: -scaledSize.height / 2)
    return CGPath(rect: CGRect(origin: origin, size: scaledSize),
                  transform: &transform)
  }
}
