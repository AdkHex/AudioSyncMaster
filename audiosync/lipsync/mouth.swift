// How open the mouth of the largest face is, frame by frame, in a short
// video: the picture half of the lip check (see audiosync/lipcheck.py).
//
// Reads every frame of the file with AVAssetReader, finds faces and their
// landmarks with Vision, and prints one JSON line per frame:
//   {"t": seconds, "faces": n, "open": inner-lip height / width,
//    "gap": inner-lip height / face height, "size": face height,
//    "x": face centre x, "y": face centre y}
// for the largest face; its fields are null when no face was found.
// macOS only: Vision and AVFoundation.

import AVFoundation
import Foundation
import Vision

func fail(_ message: String) -> Never {
    FileHandle.standardError.write((message + "\n").data(using: .utf8)!)
    exit(1)
}

guard CommandLine.arguments.count >= 2 else { fail("usage: mouth VIDEO.mp4") }
let url = URL(fileURLWithPath: CommandLine.arguments[1])
let asset = AVURLAsset(url: url)
let semaphore = DispatchSemaphore(value: 0)
var videoTrack: AVAssetTrack?
Task {
    videoTrack = try? await asset.loadTracks(withMediaType: .video).first
    semaphore.signal()
}
semaphore.wait()
guard let track = videoTrack else { fail("no video track in \(url.path)") }
guard let reader = try? AVAssetReader(asset: asset) else { fail("cannot read \(url.path)") }
let output = AVAssetReaderTrackOutput(
    track: track,
    outputSettings: [kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA]
)
output.alwaysCopiesSampleData = false
reader.add(output)
guard reader.startReading() else { fail("cannot start reading \(url.path)") }

/// Height over width of a set of landmark points.
func aspect(_ points: [CGPoint]) -> Double? {
    guard points.count >= 4 else { return nil }
    let xs = points.map { Double($0.x) }
    let ys = points.map { Double($0.y) }
    let width = (xs.max() ?? 0) - (xs.min() ?? 0)
    let height = (ys.max() ?? 0) - (ys.min() ?? 0)
    return width > 1e-6 ? height / width : nil
}

/// Height of a set of landmark points, in the face's own normalised units.
func height(_ points: [CGPoint]) -> Double? {
    guard points.count >= 4 else { return nil }
    let ys = points.map { Double($0.y) }
    return (ys.max() ?? 0) - (ys.min() ?? 0)
}

func number(_ value: Double?, _ format: String = "%.5f") -> String {
    value.map { String(format: format, $0) } ?? "null"
}

let out = FileHandle.standardOutput
while let sample = output.copyNextSampleBuffer() {
    guard let pixels = CMSampleBufferGetImageBuffer(sample) else { continue }
    let t = CMTimeGetSeconds(CMSampleBufferGetPresentationTimeStamp(sample))
    let request = VNDetectFaceLandmarksRequest()
    let handler = VNImageRequestHandler(cvPixelBuffer: pixels, options: [:])
    var faces = 0
    var openness: Double? = nil
    var gap: Double? = nil
    var size: Double? = nil
    var cx: Double? = nil
    var cy: Double? = nil
    if (try? handler.perform([request])) != nil, let results = request.results {
        faces = results.count
        // The largest face is the one talking, as a rule: the close-up.
        if let face = results.max(by: { $0.boundingBox.height < $1.boundingBox.height }),
           let lips = face.landmarks?.innerLips {
            openness = aspect(lips.normalizedPoints)
            gap = height(lips.normalizedPoints)
            size = Double(face.boundingBox.height)
            cx = Double(face.boundingBox.midX)
            cy = Double(face.boundingBox.midY)
        }
    }
    var line = "{\"t\": \(String(format: "%.6f", t)), \"faces\": \(faces)"
    line += ", \"open\": " + number(openness) + ", \"gap\": " + number(gap)
    line += ", \"size\": " + number(size, "%.4f") + ", \"x\": " + number(cx, "%.4f") + ", \"y\": " + number(cy, "%.4f") + "}\n"
    out.write(line.data(using: .utf8)!)
}
if reader.status == .failed { fail("reading failed: \(reader.error?.localizedDescription ?? "unknown")") }
