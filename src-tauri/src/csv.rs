//! CSV rendering with correct escaping and spreadsheet-injection defence.
//!
//! The original built rows with `format!("\"{}\",...")` and no escaping at all,
//! so a filename containing a quote broke the row structure, and a field
//! beginning with `=` became a live formula when opened in a spreadsheet.

use crate::SyncResult;

/// Characters that make a spreadsheet treat a cell as a formula rather than text.
const FORMULA_PREFIXES: [char; 5] = ['=', '+', '-', '@', '\t'];

/// Quote and escape one field per RFC 4180, neutralising formula prefixes.
fn escape(field: &str) -> String {
    // A leading formula character is prefixed with an apostrophe, which
    // spreadsheets treat as "the rest of this cell is literal text".
    let needs_guard = field
        .chars()
        .next()
        .is_some_and(|c| FORMULA_PREFIXES.contains(&c));

    let mut out = String::with_capacity(field.len() + 4);
    out.push('"');
    if needs_guard {
        out.push('\'');
    }
    for character in field.chars() {
        if character == '"' {
            out.push('"'); // RFC 4180 escapes a quote by doubling it.
        }
        out.push(character);
    }
    out.push('"');
    out
}

fn number(value: Option<f64>, decimals: usize) -> String {
    value.map(|v| format!("{v:.decimals$}")).unwrap_or_default()
}

pub fn render(results: &[SyncResult]) -> String {
    let mut out = String::from(
        "Video,Audio,Delay (ms),Confidence,Drift (ms/s),Total Drift (ms),\
         Start Delay (ms),End Delay (ms),Video FPS,Audio Track,Video Codec,Audio Codec,\
         Codec Delay Removed (ms),Windows Used,Windows Total,Elapsed (ms),Status\n",
    );

    for result in results {
        let status = match (&result.error, result.delay_ms) {
            (Some(error), _) => error.clone(),
            (None, Some(_)) => {
                if result.is_likely_cut.unwrap_or(false) {
                    "DIFFERENT CUT".to_string()
                } else if result.is_rate_mismatch.unwrap_or(false) {
                    "FRAME RATE MISMATCH".to_string()
                } else if result.has_significant_drift.unwrap_or(false) {
                    "OK (drift detected)".to_string()
                } else {
                    "OK".to_string()
                }
            }
            (None, None) => "No match".to_string(),
        };

        let row = [
            escape(&result.video_file),
            escape(&result.audio_file),
            number(result.delay_ms, 1),
            number(result.confidence, 3),
            number(result.drift_ms_per_s, 4),
            number(result.total_drift_ms, 1),
            number(result.start_delay_ms, 1),
            number(result.end_delay_ms, 1),
            number(result.primary_fps, 3),
            result
                .secondary_track
                .map(|v| (v + 1).to_string())
                .unwrap_or_default(),
            escape(result.primary_codec.as_deref().unwrap_or("")),
            escape(result.secondary_codec.as_deref().unwrap_or("")),
            number(result.codec_delay_ms, 3),
            result
                .windows_used
                .map(|v| v.to_string())
                .unwrap_or_default(),
            result
                .windows_total
                .map(|v| v.to_string())
                .unwrap_or_default(),
            result.elapsed_ms.map(|v| v.to_string()).unwrap_or_default(),
            escape(&status),
        ]
        .join(",");

        out.push_str(&row);
        out.push('\n');
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn result_with(video: &str, audio: &str) -> SyncResult {
        SyncResult {
            video_file: video.to_string(),
            audio_file: audio.to_string(),
            delay_ms: Some(120.5),
            confidence: Some(0.94),
            ..Default::default()
        }
    }

    #[test]
    fn quotes_in_filenames_are_doubled_not_dropped() {
        let rendered = render(&[result_with("She said \"hi\".mkv", "dub.ac3")]);
        assert!(
            rendered.contains(r#""She said ""hi"".mkv""#),
            "quote not escaped per RFC 4180: {rendered}"
        );
        // The row must still have exactly one line of data.
        assert_eq!(
            rendered.lines().count(),
            2,
            "escaping broke the row structure"
        );
    }

    #[test]
    fn commas_do_not_split_fields() {
        let rendered = render(&[result_with("Movie, The (2019).mkv", "dub.ac3")]);
        let data_line = rendered.lines().nth(1).unwrap();
        assert!(data_line.starts_with(r#""Movie, The (2019).mkv""#));
    }

    #[test]
    fn formula_prefixes_are_neutralised() {
        for dangerous in ["=cmd|'/c calc'!A1", "+1+1", "-1+1", "@SUM(A1)"] {
            let rendered = render(&[result_with(dangerous, "a.ac3")]);
            let data_line = rendered.lines().nth(1).unwrap();
            assert!(
                data_line.starts_with("\"'"),
                "formula prefix not guarded for {dangerous}: {data_line}"
            );
        }
    }

    #[test]
    fn newlines_in_errors_stay_inside_the_quoted_field() {
        let mut result = result_with("a.mkv", "b.ac3");
        result.delay_ms = None;
        result.error = Some("line one\nline two".into());
        let rendered = render(&[result]);
        assert!(rendered.contains("line one\nline two"));
        assert!(rendered.trim_end().ends_with('"'));
    }

    #[test]
    fn missing_values_render_as_empty_not_zero() {
        let result = SyncResult {
            video_file: "a.mkv".into(),
            audio_file: "b.ac3".into(),
            delay_ms: None,
            error: Some("No match".into()),
            ..Default::default()
        };
        let rendered = render(&[result]);
        let data_line = rendered.lines().nth(1).unwrap();
        assert!(
            data_line.contains(",,"),
            "absent numbers should be blank, not 0: {data_line}"
        );
        assert!(!data_line.contains("0.0"), "absent value rendered as zero");
    }

    #[test]
    fn header_column_count_matches_row_column_count() {
        let rendered = render(&[result_with("a.mkv", "b.ac3")]);
        let mut lines = rendered.lines();
        let header_columns = lines.next().unwrap().split(',').count();
        // Count top-level commas outside quotes for the data row.
        let data = lines.next().unwrap();
        let mut in_quotes = false;
        let mut columns = 1;
        let mut chars = data.chars().peekable();
        while let Some(c) = chars.next() {
            match c {
                '"' if in_quotes && chars.peek() == Some(&'"') => {
                    chars.next();
                }
                '"' => in_quotes = !in_quotes,
                ',' if !in_quotes => columns += 1,
                _ => {}
            }
        }
        assert_eq!(header_columns, columns, "header/row column mismatch");
    }
}
