// vision-ocr: Apple Vision text recognition for AudioSyncMaster's Subsync OCR.
//
// The Python engine cannot call Vision itself (it ships with numpy only), so
// it runs this helper once per batch of subtitle line images rather than once
// per image: one process start and one model load for a whole movie.
//
// Usage:
//   vision-ocr [--languages ja-JP,en-US] [--no-correction] [--fast]
//              [--jobs N] [--min-height 0.0] [--custom-words a,b] image.png ...
//   vision-ocr --stdin      < {"images": [...], "languages": [...],
//                               "languageCorrection": true, "fast": false,
//                               "jobs": 4, "customWords": []}
//                            (a bare JSON array of paths works too)
//   vision-ocr --list-languages
//   vision-ocr --version
//
// Output: one JSON object per line and per image, in input order, flushed as
// soon as it is known so the caller can show progress:
//   {"index":0,"path":"...","width":W,"height":H,
//    "observations":[{"text":"...","confidence":0.5,
//                     "box":[x,y,w,h],            // normalised, top-left origin
//                     "candidates":[{"text":"...","confidence":0.3}]}]}
// or {"index":0,"path":"...","error":"..."} for an image that failed.

import CoreGraphics
import Foundation
import ImageIO
import Vision

let helperVersion = "1"

struct Candidate: Encodable {
    let text: String
    let confidence: Float
}

struct Observation: Encodable {
    let text: String
    let confidence: Float
    let box: [Double]
    let candidates: [Candidate]
}

struct ImageResult: Encodable {
    let index: Int
    let path: String
    var width: Int?
    var height: Int?
    var observations: [Observation]?
    var error: String?
}

struct Settings {
    var images: [String] = []
    var languages: [String] = []
    var correction = true
    var fast = false
    var jobs = max(1, min(4, ProcessInfo.processInfo.activeProcessorCount))
    var minHeight: Float = 0
    var customWords: [String] = []
}

func fail(_ message: String) -> Never {
    FileHandle.standardError.write((message + "\n").data(using: .utf8)!)
    exit(2)
}

func emit<T: Encodable>(_ value: T) {
    let encoder = JSONEncoder()
    encoder.outputFormatting = [.withoutEscapingSlashes]
    guard var data = try? encoder.encode(value) else { return }
    data.append(0x0A)
    FileHandle.standardOutput.write(data)
}

func splitList(_ raw: String) -> [String] {
    raw.split(separator: ",").map { $0.trimmingCharacters(in: .whitespaces) }.filter { !$0.isEmpty }
}

func readStdin(into settings: inout Settings) {
    let data = FileHandle.standardInput.readDataToEndOfFile()
    guard !data.isEmpty else { return }
    guard let json = try? JSONSerialization.jsonObject(with: data) else { fail("stdin is not valid JSON") }
    if let paths = json as? [String] {
        settings.images += paths
        return
    }
    guard let object = json as? [String: Any] else { fail("stdin JSON must be an array or an object") }
    if let paths = object["images"] as? [String] { settings.images += paths }
    if let langs = object["languages"] as? [String] { settings.languages = langs }
    if let flag = object["languageCorrection"] as? Bool { settings.correction = flag }
    if let flag = object["fast"] as? Bool { settings.fast = flag }
    if let jobs = object["jobs"] as? Int { settings.jobs = max(1, jobs) }
    if let height = object["minHeight"] as? Double { settings.minHeight = Float(height) }
    if let words = object["customWords"] as? [String] { settings.customWords = words }
}

func listLanguages() {
    struct Languages: Encodable {
        let version: String
        let accurate: [String]
        let fast: [String]
    }
    func supported(_ level: VNRequestTextRecognitionLevel) -> [String] {
        let request = VNRecognizeTextRequest()
        request.recognitionLevel = level
        if #available(macOS 12.0, *) {
            return (try? request.supportedRecognitionLanguages()) ?? []
        }
        // macOS 11 only knows the Latin-script languages.
        return ["en-US", "fr-FR", "it-IT", "de-DE", "es-ES", "pt-BR", "zh-Hans", "zh-Hant"]
    }
    emit(Languages(version: helperVersion, accurate: supported(.accurate), fast: supported(.fast)))
}

func loadImage(_ path: String) -> CGImage? {
    let url = URL(fileURLWithPath: path) as CFURL
    guard let source = CGImageSourceCreateWithURL(url, nil) else { return nil }
    return CGImageSourceCreateImageAtIndex(source, 0, nil)
}

func recognise(index: Int, path: String, settings: Settings) -> ImageResult {
    var result = ImageResult(index: index, path: path)
    guard let image = loadImage(path) else {
        result.error = "Could not read image"
        return result
    }
    result.width = image.width
    result.height = image.height
    // Each image gets its own request: requests are not safe to share
    // between threads.
    let request = VNRecognizeTextRequest()
    request.recognitionLevel = settings.fast ? .fast : .accurate
    request.usesLanguageCorrection = settings.correction
    if !settings.languages.isEmpty {
        request.recognitionLanguages = settings.languages
    }
    if #available(macOS 13.0, *) {
        request.automaticallyDetectsLanguage = settings.languages.isEmpty
    }
    if settings.minHeight > 0 {
        request.minimumTextHeight = settings.minHeight
    }
    if !settings.customWords.isEmpty {
        request.customWords = settings.customWords
    }
    let handler = VNImageRequestHandler(cgImage: image, options: [:])
    do {
        try handler.perform([request])
    } catch {
        result.error = "Vision failed: \(error.localizedDescription)"
        return result
    }
    var observations: [Observation] = []
    for observation in request.results ?? [] {
        let candidates = observation.topCandidates(3)
        guard let best = candidates.first else { continue }
        let b = observation.boundingBox
        observations.append(Observation(
            text: best.string,
            confidence: best.confidence,
            // Vision's origin is bottom-left; callers think top-left.
            box: [Double(b.minX), Double(1.0 - b.maxY), Double(b.width), Double(b.height)],
            candidates: candidates.dropFirst().map { Candidate(text: $0.string, confidence: $0.confidence) }
        ))
    }
    result.observations = observations
    return result
}

/// Prints results in input order as soon as each prefix is complete.
final class OrderedPrinter: @unchecked Sendable {
    private let lock = NSLock()
    private var pending: [Int: ImageResult] = [:]
    private var next = 0

    func deliver(_ result: ImageResult) {
        lock.lock()
        defer { lock.unlock() }
        pending[result.index] = result
        while let ready = pending.removeValue(forKey: next) {
            emit(ready)
            next += 1
        }
    }
}

var settings = Settings()
var useStdin = false
var arguments = Array(CommandLine.arguments.dropFirst())
while !arguments.isEmpty {
    let arg = arguments.removeFirst()
    func value() -> String {
        guard !arguments.isEmpty else { fail("\(arg) needs a value") }
        return arguments.removeFirst()
    }
    switch arg {
    case "--version":
        print("{\"version\":\"\(helperVersion)\"}")
        exit(0)
    case "--list-languages":
        listLanguages()
        exit(0)
    case "--languages", "-l":
        settings.languages = splitList(value())
    case "--no-correction":
        settings.correction = false
    case "--correction":
        settings.correction = true
    case "--fast":
        settings.fast = true
    case "--jobs", "-j":
        settings.jobs = max(1, Int(value()) ?? 1)
    case "--min-height":
        settings.minHeight = Float(value()) ?? 0
    case "--custom-words":
        settings.customWords = splitList(value())
    case "--stdin", "-":
        useStdin = true
    default:
        if arg.hasPrefix("--") { fail("Unknown option \(arg)") }
        settings.images.append(arg)
    }
}
if useStdin || settings.images.isEmpty {
    readStdin(into: &settings)
}

let frozen = settings
let printer = OrderedPrinter()
let queue = OperationQueue()
queue.maxConcurrentOperationCount = frozen.jobs
for (index, path) in frozen.images.enumerated() {
    queue.addOperation {
        let result = autoreleasepool { recognise(index: index, path: path, settings: frozen) }
        printer.deliver(result)
    }
}
queue.waitUntilAllOperationsAreFinished()
