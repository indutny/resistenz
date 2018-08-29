//
//  ViewController.swift
//  Resistenz
//
//  Created by Indutnyy, Fedor on 8/26/18.
//  Copyright © 2018 Fedor Indutny. All rights reserved.
//

import UIKit
import AVKit
import CoreML
import Vision
import SpriteKit

let kConfidenceThreshold: Float = 0.5
let kGridChannels = 7
let kPriorSizes: Array<(Float, Float)> = [
  ( 0.14377480392797287, 0.059023397839700086 ),
  ( 0.20904473801128326, 0.08287369797830041 ),
  ( 0.2795802996888472, 0.11140121237843759 ),
  ( 0.3760081365223815, 0.1493933380505552 ),
  ( 0.5984967942142249, 0.2427157057261726 ),
]

func sigmoid(_ x: Float) -> Float {
  return 1.0 / (1.0 + exp(-x))
}

class ViewController: UIViewController,
                      AVCaptureVideoDataOutputSampleBufferDelegate {
  @IBOutlet weak var spriteView: SKView!

  private var session: AVCaptureSession!
  private var activeRects: [SKNode] = []

  override func viewDidLoad() {
    super.viewDidLoad()
    session = AVCaptureSession()
    guard let device = AVCaptureDevice.default(for: .video) else {
      print("No device!")
      return
    }
    guard let input = try? AVCaptureDeviceInput(device: device) else {
      print("No input!")
      return
    }
    session.addInput(input)

    let output = AVCaptureVideoDataOutput()
    output.setSampleBufferDelegate(self,
        queue: DispatchQueue(label: "videoProcessing"))
    session.addOutput(output)

    output.connection(with: .video)?.videoOrientation = .portrait

    let preview = AVCaptureVideoPreviewLayer(session: session)
    preview.frame = view.frame
    view.layer.addSublayer(preview)
    preview.zPosition = -1.0

    let scene = SKScene(size: CGSize(width: view.frame.width,
                                     height: view.frame.height))
    scene.backgroundColor = .clear

    self.spriteView.presentScene(scene)
  }

  override func viewWillAppear(_ animated: Bool) {
    session.startRunning()
  }

  override func viewWillDisappear(_ animated: Bool) {
    session.stopRunning()
  }

  override func didReceiveMemoryWarning() {
    super.didReceiveMemoryWarning()
    // Dispose of any resources that can be recreated.
  }

  // MARK: AVKit methods
  func captureOutput(_ output: AVCaptureOutput,
                     didDrop sampleBuffer: CMSampleBuffer,
                     from connection: AVCaptureConnection) {
    // no-op
  }

  func captureOutput(_ output: AVCaptureOutput,
                     didOutput sampleBuffer: CMSampleBuffer,
                     from connection: AVCaptureConnection) {
    guard let imageBuf = CMSampleBufferGetImageBuffer(sampleBuffer) else {
      print("No buffer")
      return
    }
    let handler = VNImageRequestHandler(cvPixelBuffer: imageBuf,
                                        orientation: .up, options: [:])
    do {
      try handler.perform([ self.request ])
    } catch {
      print(error)
      fatalError()
    }

  }

  // MARK: Vision stuff
  private lazy var request: VNCoreMLRequest = {
    let model: VNCoreMLModel = try! VNCoreMLModel(
        for: ResistenzGraphFP16().model)

    let request = VNCoreMLRequest(model: model) {
      [weak self] (request, error) in
      self?.processRects(for: request, error: error)
    }
    request.imageCropAndScaleOption = .centerCrop
    request.usesCPUOnly = true

    return request
  }()

  func processRects(for request: VNRequest, error: Error?) {
    if let error = error {
      print(error.localizedDescription)
      return
    }

    guard let results =
      request.results?[0] as? VNCoreMLFeatureValueObservation else {
        return
      }

    guard let arrayValue = results.featureValue.multiArrayValue else {
      return
    }

    let gridHeight = arrayValue.shape[1].intValue
    let gridWidth = arrayValue.shape[2].intValue
    let gridDepth = arrayValue.shape[0].intValue / kGridChannels

    func getCellValue(_ x: Int, _ y: Int, _ i: Int) -> Float {
      let off = x + y * gridWidth + i * gridWidth * gridHeight
      return arrayValue[off].floatValue
    }

    var rects: [OrientedRect] = []
    for y in 0..<gridHeight {
      for x in 0..<gridWidth {
        for depth in 0..<gridDepth {
          let i = depth * kGridChannels

          let confidence = sigmoid(getCellValue(x, y, i + 6))
          if confidence < kConfidenceThreshold {
            continue
          }

          let cx = sigmoid(getCellValue(x, y, i + 0)) + Float(x)
          let cy = sigmoid(getCellValue(x, y, i + 1)) + Float(y)
          let (priorWidth, priorHeight) = kPriorSizes[depth]
          let width = exp(getCellValue(x, y, i + 2)) * priorWidth
          let height = exp(getCellValue(x, y, i + 3)) * priorHeight

          let cos = getCellValue(x, y, i + 4)
          let sin = getCellValue(x, y, i + 5)
          let angle = atan2(sin, cos)

          let rect = OrientedRect(cx: cx / Float(gridWidth),
                                  cy: cy / Float(gridHeight),
                                  width: width,
                                  height: height,
                                  angle: angle,
                                  confidence: confidence)
          rects.append(rect)
        }
      }
    }

    DispatchQueue.main.async { [weak self] in
      self?.displayRects(rects)
    }
  }

  func displayRects(_ rects: [OrientedRect]) {
    guard let scene = self.spriteView.scene else {
      return
    }
    let scale = min(scene.size.width, scene.size.height)
    for node in activeRects {
      node.removeFromParent()
    }
    activeRects = []
    for rect in rects {
      let node = SKShapeNode(path: rect.toPath(scale: scale))
      let position = CGPoint(
        x: rect.center.x * scale,
        y: (scene.size.height / 2) + scale / 2 - rect.center.y * scale)
      node.position = position
      node.strokeColor = .green
      node.fillColor = UIColor(red: 0.0, green: 1.0, blue: 0.0,
                               alpha: rect.confidence - CGFloat(kConfidenceThreshold))
      node.lineWidth = rect.confidence
      scene.addChild(node)
      activeRects.append(node)
    }
  }
}

