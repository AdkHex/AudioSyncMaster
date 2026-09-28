// Test helper: renders subtitle captions the way Blu-ray authoring tools do
// -- a white (or coloured) fill with a black outline on a transparent
// background -- so the OCR tests have realistic bitmaps without shipping any.
//
// Reads a JSON array of captions on stdin and writes one PNG per caption:
//   [{"out": "/tmp/a.png", "lines": [{"text": "今日は天気", "ruby": [[3, 5, "てんき"]]}],
//     "font": "HiraginoSans-W6", "size": 56, "italic": false, "outline": 4,
//     "fill": [255, 255, 255], "vertical": false}]
// ``ruby`` spans are UTF-16 offsets [start, end) with their reading, drawn at
// half size centred above the base text (furigana). ``italic`` applies a
// 12-degree synthetic oblique, which works for any font and script.
// Prints one JSON line per caption: {"out": ..., "width": W, "height": H}.

import CoreGraphics
import CoreText
import Foundation
import ImageIO

struct LineSpec: Decodable {
    let text: String
    let ruby: [[RubyPart]]?
}

enum RubyPart: Decodable {
    case index(Int)
    case text(String)
    init(from decoder: Decoder) throws {
        let c = try decoder.singleValueContainer()
        if let i = try? c.decode(Int.self) { self = .index(i) } else { self = .text(try c.decode(String.self)) }
    }
}

struct CaptionSpec: Decodable {
    let out: String
    let lines: [LineSpec]
    let font: String?
    let size: Double?
    let italic: Bool?
    let outline: Double?
    let fill: [Double]?
    let vertical: Bool?
    let lineSpacing: Double?
}

func makeFont(_ name: String, _ size: CGFloat, italic: Bool) -> CTFont {
    if italic {
        var shear = CGAffineTransform(a: 1, b: 0, c: 0.21, d: 1, tx: 0, ty: 0)
        return CTFontCreateWithName(name as CFString, size, &shear)
    }
    return CTFontCreateWithName(name as CFString, size, nil)
}

func makeLine(_ text: String, font: CTFont, color: CGColor, stroke: CGFloat? = nil) -> CTLine {
    var attrs: [CFString: Any] = [kCTFontAttributeName: font, kCTForegroundColorAttributeName: color]
    if let stroke = stroke {
        attrs[kCTStrokeWidthAttributeName] = stroke
        attrs[kCTStrokeColorAttributeName] = CGColor(red: 0, green: 0, blue: 0, alpha: 1)
    }
    return CTLineCreateWithAttributedString(NSAttributedString(string: text, attributes: attrs as [NSAttributedString.Key: Any]))
}

struct Metrics {
    let width: CGFloat
    let ascent: CGFloat
    let descent: CGFloat
}

func metrics(_ line: CTLine) -> Metrics {
    var ascent: CGFloat = 0, descent: CGFloat = 0, leading: CGFloat = 0
    let width = CGFloat(CTLineGetTypographicBounds(line, &ascent, &descent, &leading))
    return Metrics(width: width, ascent: ascent, descent: descent)
}

/// Outline first (stroke only, twice the outline width, centred on the
/// glyph edge), then the fill on top: the classic subtitle look.
func drawOutlined(_ ctx: CGContext, _ text: String, font: CTFont, fill: CGColor, outline: CGFloat, x: CGFloat, y: CGFloat) {
    let size = CTFontGetSize(font)
    if outline > 0 {
        let stroke = makeLine(text, font: font, color: fill, stroke: 2 * outline / size * 100)
        ctx.textPosition = CGPoint(x: x, y: y)
        CTLineDraw(stroke, ctx)
    }
    let body = makeLine(text, font: font, color: fill)
    ctx.textPosition = CGPoint(x: x, y: y)
    CTLineDraw(body, ctx)
}

func render(_ spec: CaptionSpec) throws -> (Int, Int) {
    let size = CGFloat(spec.size ?? 56)
    let outline = CGFloat(spec.outline ?? 4)
    let italic = spec.italic ?? false
    let name = spec.font ?? "HiraginoSans-W6"
    let font = makeFont(name, size, italic: italic)
    let rubyFont = makeFont(name, size * 0.5, italic: italic)
    let rgb = spec.fill ?? [255, 255, 255]
    let fill = CGColor(red: rgb[0] / 255, green: rgb[1] / 255, blue: rgb[2] / 255, alpha: 1)
    let pad = outline + 6
    let spacing = CGFloat(spec.lineSpacing ?? 1.2)

    var width: CGFloat = 0
    var height: CGFloat = 0
    var draws: [(CGContext, CGFloat) -> Void] = []  // (context, canvas height)

    if spec.vertical ?? false {
        // Columns right to left, characters top to bottom, each centred;
        // columns are a line pitch apart, as horizontal lines would be.
        let cell = size * 1.08
        let pitch = size * CGFloat(spec.lineSpacing ?? 1.5)
        let columns = spec.lines.map { Array($0.text) }
        let rows = columns.map { $0.count }.max() ?? 0
        width = CGFloat(max(0, columns.count - 1)) * pitch + cell + 2 * pad
        height = CGFloat(rows) * cell + 2 * pad
        let ascent = CTFontGetAscent(font), descent = CTFontGetDescent(font)
        for (c, chars) in columns.enumerated() {
            let left = width - pad - cell - CGFloat(c) * pitch
            for (r, ch) in chars.enumerated() {
                let text = String(ch)
                let w = metrics(makeLine(text, font: font, color: fill)).width
                let top = pad + CGFloat(r) * cell
                draws.append { ctx, h in
                    let baseline = h - top - (cell - (ascent + descent)) / 2 - ascent
                    drawOutlined(ctx, text, font: font, fill: fill, outline: outline, x: left + (cell - w) / 2, y: baseline)
                }
            }
        }
    } else {
        var y: CGFloat = pad
        for spec in spec.lines {
            let line = makeLine(spec.text, font: font, color: fill)
            let m = metrics(line)
            let extra = italic ? 0.21 * m.ascent : 0
            width = max(width, m.width + extra)
            var rubyBand: CGFloat = 0
            var rubies: [(String, CGFloat, CGFloat)] = []  // text, centre x, width
            for part in spec.ruby ?? [] {
                guard part.count == 3, case let .index(s) = part[0], case let .index(e) = part[1], case let .text(t) = part[2] else { continue }
                let x0 = CTLineGetOffsetForStringIndex(line, s, nil)
                let x1 = CTLineGetOffsetForStringIndex(line, e, nil)
                let rw = metrics(makeLine(t, font: rubyFont, color: fill)).width
                rubies.append((t, (x0 + x1) / 2, rw))
            }
            if !rubies.isEmpty {
                rubyBand = CTFontGetAscent(rubyFont) + CTFontGetDescent(rubyFont) + outline * 2 + 2
            }
            let top = y + rubyBand
            let lineWidth = m.width + extra
            let text = spec.text
            draws.append { ctx, h in
                let left = (ctx.width > 0 ? CGFloat(ctx.width) : 0) / 2 - lineWidth / 2
                let baseline = h - top - m.ascent
                drawOutlined(ctx, text, font: font, fill: fill, outline: outline, x: left, y: baseline)
                for (t, centre, rw) in rubies {
                    let rubyBaseline = baseline + m.ascent + outline * 2 + 2 + CTFontGetDescent(rubyFont)
                    drawOutlined(ctx, t, font: rubyFont, fill: fill, outline: max(1, outline / 2), x: left + centre - rw / 2, y: rubyBaseline)
                }
            }
            y = top + (m.ascent + m.descent) * spacing
        }
        width += 2 * pad
        height = y + pad
    }

    let w = Int(ceil(width)), h = Int(ceil(height))
    guard let ctx = CGContext(
        data: nil, width: w, height: h, bitsPerComponent: 8, bytesPerRow: 0,
        space: CGColorSpace(name: CGColorSpace.sRGB)!,
        bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue
    ) else { throw NSError(domain: "render", code: 1) }
    ctx.setLineJoin(.round)
    ctx.setShouldAntialias(true)
    for draw in draws { draw(ctx, CGFloat(h)) }
    guard let image = ctx.makeImage(),
          let dest = CGImageDestinationCreateWithURL(URL(fileURLWithPath: spec.out) as CFURL, "public.png" as CFString, 1, nil)
    else { throw NSError(domain: "render", code: 2) }
    CGImageDestinationAddImage(dest, image, nil)
    guard CGImageDestinationFinalize(dest) else { throw NSError(domain: "render", code: 3) }
    return (w, h)
}

let input = FileHandle.standardInput.readDataToEndOfFile()
let captions = try JSONDecoder().decode([CaptionSpec].self, from: input)
for caption in captions {
    let (w, h) = try render(caption)
    print("{\"out\":\"\(caption.out)\",\"width\":\(w),\"height\":\(h)}")
}
