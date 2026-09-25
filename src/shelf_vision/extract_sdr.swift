import Foundation
import AVFoundation
import CoreImage
import ImageIO

let args = CommandLine.arguments
if args.count != 4 { fatalError("Expected source timestamp output") }
let asset = AVURLAsset(url: URL(fileURLWithPath: args[1]))
let generator = AVAssetImageGenerator(asset: asset)
generator.appliesPreferredTrackTransform = true
generator.dynamicRangePolicy = .forceSDR
generator.requestedTimeToleranceBefore = .zero
generator.requestedTimeToleranceAfter = .zero
let requested = CMTime(seconds: Double(args[2])!, preferredTimescale: 60000)
var actual = CMTime.zero
let cg = try generator.copyCGImage(at: requested, actualTime: &actual)
let image = CIImage(cgImage: cg)
let colorSpace = CGColorSpace(name: CGColorSpace.sRGB)!
try CIContext().writePNGRepresentation(of: image, to: URL(fileURLWithPath: args[3]),
                                       format: .RGBA8, colorSpace: colorSpace)
let result: [String: Any] = ["requested_seconds": requested.seconds, "actual_seconds": actual.seconds,
                           "width": cg.width, "height": cg.height]
print(String(data: try JSONSerialization.data(withJSONObject: result), encoding: .utf8)!)
