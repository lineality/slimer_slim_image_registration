//! slimer_slim_image_registration_module.rs
//!
//! # Slimer: Slim Image Registration Module (v2.0 MVP)
//!
//! Zero-heap, vanilla-Rust 2D image registration and fusion engine with native PNG I/O,
//! plus streamed 3D visualisation outputs (OBJ wireframe, textured OBJ mesh, SVG).
//!
//! ## Project purpose
//! Slimer aligns ("registers") two raster images of the same scene and produces ONE
//! output from the aligned pair, without ever loading a whole image into memory:
//!
//! | CLI mode | Output                                      | Entry function                     |
//! |----------|---------------------------------------------|------------------------------------|
//! | `2d`     | fused 8-bit grayscale PNG                   | [`enhance_png_streamed`]           |
//! | `3d`     | Wavefront OBJ wireframe (`v` + `l`)         | [`generate_3d_wireframe_obj`]      |
//! | `mesh`   | Wavefront OBJ (`v`,`vt`,`f`) + MTL texture  | [`generate_3d_textured_mesh_obj`], [`write_mtl_material_file`] |
//! | `svg`    | 2:1 dimetric wireframe projection (SVG)     | [`generate_3d_wireframe_svg`]      |
//!
//! ## What "registration" means in this MVP
//! - Model: one global integer translation (dx, dy). No rotation, scale, or perspective.
//! - Kernel: "1D Projections (Row / Column Collapsing)" from the design doc. Each image is
//!   collapsed into row-sum and column-sum luma signatures, then a normalised
//!   SAD (Sum of Absolute Differences) search runs over +/- `MAX_SEARCH_RADIUS`, further
//!   limited to half the image extent so at least 50% of each signature overlaps.
//! - Sign convention (project-wide, used by every consumer):
//!   content at reference (x, y) appears at target (x + dx, y + dy).
//! - Confidence (ppm, 0..=1_000_000): (worst SAD - best SAD) / worst SAD, averaged over both
//!   axes. Below `REGISTRATION_MIN_CONFIDENCE_PPM` registration returns
//!   `RegInsufficientContrast`, which is `is_degradable()`: the caller may fall back to the
//!   identity transform (design-doc kernel 5, "Identity Transform with Degraded Confidence").
//!
//! ## Output honesty notes (features vs. bugs)
//! - `2d` is a two-frame registered *denoise*: a pixel is averaged with its registered
//!   partner only when they agree within `FUSION_AGREEMENT_THRESHOLD`; otherwise the
//!   reference pixel is kept. Same pixel dimensions as input. NOT super-resolution.
//! - 3D Z values are a heuristic "pseudo-depth" = luma / 4 + local stereo disparity * 3.
//!   This is a visualisation aid: there is no camera model or baseline, so Z is not metric.
//!
//! ## Architectural invariants (from MVP v1, still current)
//! 1. Zero dynamic allocation in the engine (no Vec, Box, or String in production paths).
//! 2. Vanilla Rust: no third-party crates, no external tools.
//! 3. Power-of-10 style safety: every loop bounded by a static constant or a validated
//!    extent, no recursion, `#![forbid(unsafe_code)]`.
//! 4. Full mode & case handling: verbose diagnostics only in debug builds.
//!
//! ## Memory budget (Grace Hopper check)
//! Peak engine stack use is about 110 KiB, while streaming two images in lock-step:
//! - 2 x Deflate decoders (32 KiB window each, mandated by RFC 1951),
//! - 2 x pairs of 4 KiB scanline buffers,
//! - 2 x 4 KiB file read buffers,
//! - one 4 KiB write buffer.
//! This fits default stacks (Linux main 8 MiB, Windows main 1 MiB, Rust test threads 2 MiB).
//! The 4 KiB I/O buffers match the common filesystem page size and replace the v1
//! one-syscall-per-byte reads.
//!
//! ## Supported input
//! PNG, 8 bits per channel, non-interlaced. Colour types:
//! 0 (gray), 2 (RGB), 4 (gray + alpha), 6 (RGBA). Alpha is ignored; colour is converted
//! to luma with BT.601 integer weights. Width and height must be in
//! `MIN_IMAGE_DIMENSION..=MAX_IMAGE_*`.
//! Rejected with explicit codes: palette (type 3), 16-bit, interlaced, unknown critical
//! chunks, non-PNG files (including JPEG).
//!
//! ## Integrity checks
//! - The IHDR CRC is verified.
//! - Each IDAT CRC is verified when the decoder crosses into the next chunk.
//! - The final IDAT chunk's CRC and the zlib Adler-32 trailer are NOT verified: decoding
//!   stops after the last scanline, and reading on only to verify a trailer was judged not
//!   worth the extra I/O for this MVP. This is a documented limitation.
//!
//! ## Mode & case-handling policy
//! - Production-release: never panics, no heap, no printing. Every failure returns a
//!   2-byte `SlimerError`. Logging is done by the caller (`main`), never by the failing
//!   function.
//! - Debug: `#[cfg(debug_assertions)] eprintln!` diagnostics at error sites;
//!   `#[cfg(all(debug_assertions, not(test)))] debug_assert!` for internal
//!   required-conditions (e.g. a corrupted stored value).
//! - Test: `assert!` only inside `#[cfg(test)]` test functions.
//!
//! ## Function return policy
//! Everything that touches I/O, indexing, or arithmetic on external data returns
//! `Result<_, SlimerError>`. A few total mappings return plain values by design, because
//! they have no failure path:
//! - checksum state updates,
//! - enum-to-constant mappings,
//! - fixed-table builders.
//!
//! ## Indexing policy
//! Runtime indexing uses `.get()` / `.get_mut()` and returns an error on a miss.
//! Direct `array[CONST]` indexing is used only with constant indices into fixed-size
//! arrays; the compiler rejects those at build time if they are out of bounds.

#![forbid(unsafe_code)]

use std::fs::File;
use std::io::{ErrorKind, Read, Seek, SeekFrom, Write};
use std::time::{SystemTime, UNIX_EPOCH};

// ============================================================================
// CONSTANTS & STATIC BOUNDS
// ============================================================================

/// Maximum supported image width in pixels (sizes all row buffers on the stack).
pub const MAX_IMAGE_WIDTH: usize = 1024;
/// Maximum supported image height in pixels (sizes the row-signature array).
pub const MAX_IMAGE_HEIGHT: usize = 1024;
/// Minimum supported width/height; smaller images carry too little signal to register.
pub const MIN_IMAGE_DIMENSION: usize = 8;
/// Maximum translation searched per axis, in pixels.
pub const MAX_SEARCH_RADIUS: i32 = 32;
/// Upper bound of the confidence scale (parts per million).
pub const CONFIDENCE_PPM_MAX: u32 = 1_000_000;
/// Below this combined confidence, registration reports `RegInsufficientContrast`.
pub const REGISTRATION_MIN_CONFIDENCE_PPM: u32 = 40_000;
/// `2d` fusion: average two registered pixels only if their luma differs by less than this.
pub const FUSION_AGREEMENT_THRESHOLD: u8 = 32;
/// Spacing (pixels) between 3D grid vertices in x and y.
pub const WIREFRAME_GRID_STEP: usize = 16;
/// Max grid vertices per row: ceil(MAX_IMAGE_WIDTH / step), plus a safety margin.
pub const MAX_GRID_POINTS_PER_ROW: usize = (MAX_IMAGE_WIDTH / WIREFRAME_GRID_STEP) + 2;
/// Local (per-vertex) stereo disparity search range, in pixels, on top of the global dx.
pub const MAX_LOCAL_DISPARITY_SEARCH: i32 = 6;
/// Half-width of the 1D SAD window used for local disparity (window = 2*h + 1 pixels).
pub const DISPARITY_WINDOW_HALF_WIDTH: i32 = 3;
/// Pseudo-depth: luma contribution divisor (0..=255 luma -> 0..=63).
pub const DEPTH_LUMA_DIVISOR: i32 = 4;
/// Pseudo-depth: multiplier applied to the local disparity.
pub const DEPTH_DISPARITY_GAIN: i32 = 3;

// BT.601 luma approximation; weights sum to 256 so the result is `>> 8`.
const LUMA_WEIGHT_RED: u32 = 77;
const LUMA_WEIGHT_GREEN: u32 = 150;
const LUMA_WEIGHT_BLUE: u32 = 29;

const MAX_PNG_BYTES_PER_PIXEL: usize = 4;
const MAX_SCANLINE_BYTES: usize = MAX_IMAGE_WIDTH * MAX_PNG_BYTES_PER_PIXEL;
const FILE_READ_BUFFER_SIZE: usize = 4096;
const STACK_WRITE_BUFFER_SIZE: usize = 4096;
/// Bound on `ErrorKind::Interrupted` re-reads inside one buffer refill.
const MAX_IO_INTERRUPT_RETRIES: usize = 16;

const PNG_SIGNATURE: [u8; 8] = [0x89, b'P', b'N', b'G', 0x0D, 0x0A, 0x1A, 0x0A];
/// 8 signature + 4 length + 4 type + 13 data + 4 CRC.
const PNG_SIGNATURE_AND_IHDR_LEN: usize = 33;
/// PNG spec: chunk lengths must not exceed 2^31 - 1.
const PNG_MAX_CHUNK_LENGTH: u32 = 0x7FFF_FFFF;
/// Bound on ancillary chunks walked before the first IDAT.
const MAX_PNG_CHUNKS_BEFORE_IDAT: usize = 4096;
/// Bound on back-to-back zero-length IDAT chunks (legal but pathological).
const MAX_CONSECUTIVE_EMPTY_IDAT_CHUNKS: usize = 1024;
/// Complete IEND chunk: length 0, type "IEND", CRC 0xAE426082.
const PNG_IEND_CHUNK: [u8; 12] = [0, 0, 0, 0, b'I', b'E', b'N', b'D', 0xAE, 0x42, 0x60, 0x82];
/// zlib header: CM=8 (deflate), CINFO=7 (32K window), FLEVEL=0, FCHECK valid.
const ZLIB_STORED_HEADER: [u8; 2] = [0x78, 0x01];
/// Per written row: 5-byte stored-block header + 1 PNG filter byte.
const PNG_STORED_ROW_OVERHEAD: usize = 6;
/// zlib header (2) + Adler-32 trailer (4).
const ZLIB_HEADER_AND_TRAILER_BYTES: usize = 6;

const DEFLATE_WINDOW_SIZE: usize = 32_768;
/// Code lengths 0..=15 (RFC 1951 maximum code length is 15).
const HUFFMAN_COUNT_SLOTS: usize = 16;
const MAX_LITLEN_SYMBOLS: usize = 288;
const MAX_DIST_SYMBOLS: usize = 32;
const CODE_LENGTH_SYMBOLS: usize = 19;
/// Largest single `read_bits` request (Deflate needs at most 16).
const MAX_BITS_PER_READ: u32 = 24;
/// Bound on internal decoder steps per output byte (covers runs of empty blocks).
const MAX_DEFLATE_STEPS_PER_BYTE: usize = 65_536;

const CODE_LENGTH_ORDER: [usize; CODE_LENGTH_SYMBOLS] = [
    16, 17, 18, 0, 8, 7, 9, 6, 10, 5, 11, 4, 12, 3, 13, 2, 14, 1, 15,
];
const DEFLATE_LENGTH_BASES: [u16; 29] = [
    3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 15, 17, 19, 23, 27, 31, 35, 43, 51, 59, 67, 83, 99, 115, 131,
    163, 195, 227, 258,
];
const DEFLATE_LENGTH_EXTRA_BITS: [u8; 29] = [
    0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 0,
];
const DEFLATE_DISTANCE_BASES: [u16; 30] = [
    1, 2, 3, 4, 5, 7, 9, 13, 17, 25, 33, 49, 65, 97, 129, 193, 257, 385, 513, 769, 1025, 1537,
    2049, 3073, 4097, 6145, 8193, 12289, 16385, 24577,
];
const DEFLATE_DISTANCE_EXTRA_BITS: [u8; 30] = [
    0, 0, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10, 10, 11, 11, 12, 12, 13,
    13,
];

const SVG_VIEWBOX_MARGIN: i32 = 16;
const MAX_MTL_REFERENCE_NAME_BYTES: usize = 255;
const ADLER32_MODULUS: u32 = 65_521;

// ============================================================================
// ERROR ENUM (FIELDLESS, 2-BYTE, REPR U16, APPEND-ONLY)
// ============================================================================

/// Project-wide error code.
///
/// The payload is always 2 bytes: no heap and no PII.
///
/// Rules:
/// - Codes are permanent. Never renumber or reuse a code.
/// - Gaps (e.g. 401, 403) are retired or reserved numbers and must stay unused.
///
/// Reserved blocks:
/// - 120-199 PNG / zlib / Deflate decoding
/// - 200-299 file-system I/O
/// - 300-399 value integrity (custom-type checks, internal indexing)
/// - 400-499 registration
/// - 500-599 streaming pipeline (writers)
/// - 600-699 command-line runner
/// - 700-799 3D mesh output
#[repr(u16)]
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum SlimerError {
    PngMagicInvalid = 120,
    PngUnsupportedColorType = 121,
    PngUnsupportedBitDepth = 122,
    PngUnsupportedCompression = 123,
    PngUnsupportedInterlace = 124,
    PngCorruptChunk = 125,
    PngZlibHeaderInvalid = 126,
    PngDeflateCorrupt = 127,
    /// IHDR filter *method* field is not 0.
    PngFilterTypeInvalid = 128,
    PngIhdrCrcMismatch = 129,
    PngIdatCrcMismatch = 130,
    /// Huffman code lengths are over-subscribed or out of range.
    PngHuffmanTableInvalid = 131,
    /// LZ77 back-reference points before the start of the decoded data.
    PngDeflateDistanceTooFar = 132,
    /// Non-IDAT chunk inside the IDAT sequence, duplicate IHDR, or too many empty IDATs.
    PngChunkSequenceInvalid = 133,
    /// Per-scanline filter *type* byte is greater than 4.
    PngScanlineFilterInvalid = 134,
    /// Compressed stream ended before all scanlines were decoded.
    PngImageDataTruncated = 135,
    PngUnsupportedCriticalChunk = 136,
    PngZlibPresetDictionary = 137,

    IoReadFailed = 200,
    IoWriteFailed = 201,
    IoSeekFailed = 202,
    IoUnexpectedEof = 203,
    IoFlushFailed = 204,
    IoCreateFailed = 205,
    IoOpenFailed = 206,
    IoInputNotFound = 207,
    IoPermissionDenied = 208,
    IoOutputAlreadyExists = 209,
    IoPathResolveFailed = 210,

    ValIntegrityDimUnderflow = 300,
    ValIntegrityDimOverflow = 301,
    ValIntegrityShiftOutOfBounds = 302,
    ValIntegrityScoreCorrupt = 303,
    ValCorruptedBitFlipDetected = 304,
    /// Internal index outside a buffer (should-not-happen; caught, not panicked).
    ValIndexOutOfBounds = 305,
    /// Caller passed an argument outside the documented domain.
    ValArgumentInvalid = 306,

    RegDimMismatch = 400,
    RegInsufficientContrast = 402,
    RegArithmeticOverflow = 404,

    PipeBufferExhaustion = 503,
    PipeRowCountExceeded = 504,
    PipeRowCountIncomplete = 505,

    CliArgumentsMissing = 600,
    CliModeUnknown = 601,
    CliTooManyArguments = 602,
    CliOutputPathConflict = 603,

    MeshFilenameInvalid = 700,
    MeshIndexOverflow = 701,
}

impl SlimerError {
    /// Stable numeric code, for terse production output and the error log.
    #[inline]
    pub fn code(self) -> u16 {
        self as u16
    }

    /// Recovery Tier 1: may a bounded retry of the same command succeed?
    ///
    /// Exhaustive by design: adding a variant forces a decision here.
    pub fn is_retryable(self) -> bool {
        match self {
            Self::IoReadFailed
            | Self::IoWriteFailed
            | Self::IoSeekFailed
            | Self::IoFlushFailed
            | Self::IoOpenFailed
            | Self::IoCreateFailed => true,
            Self::PngMagicInvalid
            | Self::PngUnsupportedColorType
            | Self::PngUnsupportedBitDepth
            | Self::PngUnsupportedCompression
            | Self::PngUnsupportedInterlace
            | Self::PngCorruptChunk
            | Self::PngZlibHeaderInvalid
            | Self::PngDeflateCorrupt
            | Self::PngFilterTypeInvalid
            | Self::PngIhdrCrcMismatch
            | Self::PngIdatCrcMismatch
            | Self::PngHuffmanTableInvalid
            | Self::PngDeflateDistanceTooFar
            | Self::PngChunkSequenceInvalid
            | Self::PngScanlineFilterInvalid
            | Self::PngImageDataTruncated
            | Self::PngUnsupportedCriticalChunk
            | Self::PngZlibPresetDictionary
            | Self::IoUnexpectedEof
            | Self::IoInputNotFound
            | Self::IoPermissionDenied
            | Self::IoOutputAlreadyExists
            | Self::IoPathResolveFailed
            | Self::ValIntegrityDimUnderflow
            | Self::ValIntegrityDimOverflow
            | Self::ValIntegrityShiftOutOfBounds
            | Self::ValIntegrityScoreCorrupt
            | Self::ValCorruptedBitFlipDetected
            | Self::ValIndexOutOfBounds
            | Self::ValArgumentInvalid
            | Self::RegDimMismatch
            | Self::RegInsufficientContrast
            | Self::RegArithmeticOverflow
            | Self::PipeBufferExhaustion
            | Self::PipeRowCountExceeded
            | Self::PipeRowCountIncomplete
            | Self::CliArgumentsMissing
            | Self::CliModeUnknown
            | Self::CliTooManyArguments
            | Self::CliOutputPathConflict
            | Self::MeshFilenameInvalid
            | Self::MeshIndexOverflow => false,
        }
    }

    /// Recovery Tier 2: may the caller continue with a safe degraded result instead?
    #[inline]
    pub fn is_degradable(self) -> bool {
        matches!(self, Self::RegInsufficientContrast)
    }
}

/// Human-readable form. Compiled ONLY in debug/test builds (never in production).
#[cfg(debug_assertions)]
impl std::fmt::Display for SlimerError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "SLIMER-{} {:?}", self.code(), self)
    }
}

/// Maps a read failure to a project code.
///
/// EOF is permanent (truncated file). Every other read failure may be transient.
fn classify_read_error(kind: ErrorKind) -> SlimerError {
    match kind {
        ErrorKind::UnexpectedEof => SlimerError::IoUnexpectedEof,
        _ => SlimerError::IoReadFailed,
    }
}

/// Maps an input-file open failure to a project code (used by the CLI runner).
pub fn classify_open_error(kind: ErrorKind) -> SlimerError {
    match kind {
        ErrorKind::NotFound => SlimerError::IoInputNotFound,
        ErrorKind::PermissionDenied => SlimerError::IoPermissionDenied,
        _ => SlimerError::IoOpenFailed,
    }
}

/// Maps an output-file create failure to a project code (used by the CLI runner).
pub fn classify_create_error(kind: ErrorKind) -> SlimerError {
    match kind {
        ErrorKind::AlreadyExists => SlimerError::IoOutputAlreadyExists,
        ErrorKind::PermissionDenied => SlimerError::IoPermissionDenied,
        _ => SlimerError::IoCreateFailed,
    }
}

// ============================================================================
// SMALL PANIC-FREE BYTE HELPERS
// ============================================================================

/// Reads one byte at `index`, returning an error instead of panicking.
#[inline]
fn byte_at(bytes: &[u8], index: usize) -> Result<u8, SlimerError> {
    match bytes.get(index) {
        Some(&value) => Ok(value),
        None => Err(SlimerError::ValIndexOutOfBounds),
    }
}

/// Reads a big-endian u32 at `offset` (PNG chunk lengths, dimensions, CRCs).
fn read_be_u32_at(bytes: &[u8], offset: usize) -> Result<u32, SlimerError> {
    let end = offset
        .checked_add(4)
        .ok_or(SlimerError::ValIndexOutOfBounds)?;
    let window = bytes
        .get(offset..end)
        .ok_or(SlimerError::ValIndexOutOfBounds)?;
    match <[u8; 4]>::try_from(window) {
        Ok(array) => Ok(u32::from_be_bytes(array)),
        Err(_) => Err(SlimerError::ValIndexOutOfBounds),
    }
}

/// True if `expected` occurs at `offset`. Out of range is `false`, never a panic.
fn slice_matches_at(bytes: &[u8], offset: usize, expected: &[u8]) -> bool {
    match offset.checked_add(expected.len()) {
        Some(end) => bytes.get(offset..end) == Some(expected),
        None => false,
    }
}

// ============================================================================
// ENFORCED CUSTOM TYPES (VALUE INTEGRITY)
// ============================================================================

/// Image width in pixels, guaranteed to be in `MIN_IMAGE_DIMENSION..=MAX_IMAGE_WIDTH`.
///
/// `get()` re-checks the bounds on every access. A stored value corrupted after
/// construction (bit-flip) becomes an error, not silent misbehaviour.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct BoundedWidth {
    pixels: usize,
}

impl BoundedWidth {
    /// Validates and wraps a width (expected-case input validation).
    pub fn new(pixels: usize) -> Result<Self, SlimerError> {
        if pixels < MIN_IMAGE_DIMENSION {
            #[cfg(debug_assertions)]
            eprintln!(
                "BWID-300: width {} below minimum {}",
                pixels, MIN_IMAGE_DIMENSION
            );
            return Err(SlimerError::ValIntegrityDimUnderflow);
        }
        if pixels > MAX_IMAGE_WIDTH {
            #[cfg(debug_assertions)]
            eprintln!(
                "BWID-301: width {} above maximum {}",
                pixels, MAX_IMAGE_WIDTH
            );
            return Err(SlimerError::ValIntegrityDimOverflow);
        }
        Ok(Self { pixels })
    }

    /// Returns the width after a validity re-check.
    #[inline]
    pub fn get(&self) -> Result<usize, SlimerError> {
        self.validity_recheck()?;
        Ok(self.pixels)
    }

    /// Re-checks the stored value (catches post-construction corruption).
    pub fn validity_recheck(&self) -> Result<(), SlimerError> {
        let is_valid = self.pixels >= MIN_IMAGE_DIMENSION && self.pixels <= MAX_IMAGE_WIDTH;
        #[cfg(all(debug_assertions, not(test)))]
        debug_assert!(is_valid, "BWID-304: stored width {} corrupted", self.pixels);
        if !is_valid {
            return Err(SlimerError::ValCorruptedBitFlipDetected);
        }
        Ok(())
    }
}

/// Image height in pixels, guaranteed to be in `MIN_IMAGE_DIMENSION..=MAX_IMAGE_HEIGHT`.
///
/// This is a separate type from width so the two can never be swapped silently.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct BoundedHeight {
    pixels: usize,
}

impl BoundedHeight {
    /// Validates and wraps a height (expected-case input validation).
    pub fn new(pixels: usize) -> Result<Self, SlimerError> {
        if pixels < MIN_IMAGE_DIMENSION {
            #[cfg(debug_assertions)]
            eprintln!(
                "BHGT-300: height {} below minimum {}",
                pixels, MIN_IMAGE_DIMENSION
            );
            return Err(SlimerError::ValIntegrityDimUnderflow);
        }
        if pixels > MAX_IMAGE_HEIGHT {
            #[cfg(debug_assertions)]
            eprintln!(
                "BHGT-301: height {} above maximum {}",
                pixels, MAX_IMAGE_HEIGHT
            );
            return Err(SlimerError::ValIntegrityDimOverflow);
        }
        Ok(Self { pixels })
    }

    /// Returns the height after a validity re-check.
    #[inline]
    pub fn get(&self) -> Result<usize, SlimerError> {
        self.validity_recheck()?;
        Ok(self.pixels)
    }

    /// Re-checks the stored value (catches post-construction corruption).
    pub fn validity_recheck(&self) -> Result<(), SlimerError> {
        let is_valid = self.pixels >= MIN_IMAGE_DIMENSION && self.pixels <= MAX_IMAGE_HEIGHT;
        #[cfg(all(debug_assertions, not(test)))]
        debug_assert!(
            is_valid,
            "BHGT-304: stored height {} corrupted",
            self.pixels
        );
        if !is_valid {
            return Err(SlimerError::ValCorruptedBitFlipDetected);
        }
        Ok(())
    }
}

/// Registration result: global translation plus confidence.
///
/// Invariants:
/// - |dx|, |dy| <= `MAX_SEARCH_RADIUS`;
/// - confidence <= `CONFIDENCE_PPM_MAX`.
///
/// Sign convention: reference (x, y) corresponds to target (x + dx, y + dy).
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct RegistrationShift {
    dx: i32,
    dy: i32,
    confidence_ppm: u32,
}

impl RegistrationShift {
    /// Validates and constructs a shift.
    pub fn new(dx: i32, dy: i32, confidence_ppm: u32) -> Result<Self, SlimerError> {
        if !(-MAX_SEARCH_RADIUS..=MAX_SEARCH_RADIUS).contains(&dx) {
            #[cfg(debug_assertions)]
            eprintln!("RSH-302: dx {} outside +/-{}", dx, MAX_SEARCH_RADIUS);
            return Err(SlimerError::ValIntegrityShiftOutOfBounds);
        }
        if !(-MAX_SEARCH_RADIUS..=MAX_SEARCH_RADIUS).contains(&dy) {
            #[cfg(debug_assertions)]
            eprintln!("RSH-302: dy {} outside +/-{}", dy, MAX_SEARCH_RADIUS);
            return Err(SlimerError::ValIntegrityShiftOutOfBounds);
        }
        if confidence_ppm > CONFIDENCE_PPM_MAX {
            #[cfg(debug_assertions)]
            eprintln!(
                "RSH-303: confidence {} above {}",
                confidence_ppm, CONFIDENCE_PPM_MAX
            );
            return Err(SlimerError::ValIntegrityScoreCorrupt);
        }
        Ok(Self {
            dx,
            dy,
            confidence_ppm,
        })
    }

    /// Tier-2 degraded result: identity transform (0, 0) with zero confidence.
    ///
    /// Infallible constant constructor: it always produces a valid value.
    #[inline]
    pub fn identity_fallback() -> Self {
        Self {
            dx: 0,
            dy: 0,
            confidence_ppm: 0,
        }
    }

    /// Horizontal shift after a validity re-check.
    #[inline]
    pub fn get_dx(&self) -> Result<i32, SlimerError> {
        self.validity_recheck()?;
        Ok(self.dx)
    }

    /// Vertical shift after a validity re-check.
    #[inline]
    pub fn get_dy(&self) -> Result<i32, SlimerError> {
        self.validity_recheck()?;
        Ok(self.dy)
    }

    /// Confidence in ppm after a validity re-check.
    #[inline]
    pub fn get_confidence(&self) -> Result<u32, SlimerError> {
        self.validity_recheck()?;
        Ok(self.confidence_ppm)
    }

    /// Re-checks all stored fields (catches post-construction corruption).
    pub fn validity_recheck(&self) -> Result<(), SlimerError> {
        let range = -MAX_SEARCH_RADIUS..=MAX_SEARCH_RADIUS;
        let is_valid = range.contains(&self.dx)
            && range.contains(&self.dy)
            && self.confidence_ppm <= CONFIDENCE_PPM_MAX;
        #[cfg(all(debug_assertions, not(test)))]
        debug_assert!(is_valid, "RSH-304: stored shift corrupted: {:?}", self);
        if !is_valid {
            return Err(SlimerError::ValCorruptedBitFlipDetected);
        }
        Ok(())
    }
}

/// Supported PNG colour layouts (8 bits per channel).
///
/// Using an enum, not a raw byte count, means bytes-per-pixel can never disagree with
/// the colour type.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum PngColorKind {
    Gray,
    GrayAlpha,
    Rgb,
    Rgba,
}

impl PngColorKind {
    /// Maps the IHDR colour-type byte. Palette (3) and unknown values are rejected.
    fn from_ihdr_code(code: u8) -> Result<Self, SlimerError> {
        match code {
            0 => Ok(Self::Gray),
            2 => Ok(Self::Rgb),
            4 => Ok(Self::GrayAlpha),
            6 => Ok(Self::Rgba),
            _ => {
                #[cfg(debug_assertions)]
                eprintln!(
                    "PCK-121: unsupported PNG colour type {} (supported: 0, 2, 4, 6)",
                    code
                );
                Err(SlimerError::PngUnsupportedColorType)
            }
        }
    }

    /// Bytes per pixel at 8 bits per channel (total mapping, never 0).
    #[inline]
    pub fn bytes_per_pixel(self) -> usize {
        match self {
            Self::Gray => 1,
            Self::GrayAlpha => 2,
            Self::Rgb => 3,
            Self::Rgba => 4,
        }
    }
}

// ============================================================================
// STACK-BASED CHECKSUMS (CRC-32 & ADLER-32)
// ============================================================================

/// Builds the IEEE CRC-32 lookup table at compile time (1 KiB of static data, no heap).
const fn build_crc32_table() -> [u32; 256] {
    let mut table = [0u32; 256];
    let mut entry_index: usize = 0;
    while entry_index < 256 {
        let mut value = entry_index as u32;
        let mut bit: usize = 0;
        while bit < 8 {
            value = if value & 1 != 0 {
                0xEDB8_8320 ^ (value >> 1)
            } else {
                value >> 1
            };
            bit += 1;
        }
        table[entry_index] = value;
        entry_index += 1;
    }
    table
}

static CRC32_TABLE: [u32; 256] = build_crc32_table();

/// Streaming CRC-32 (PNG chunk checksums).
#[derive(Copy, Clone, Debug)]
pub struct Crc32 {
    state: u32,
}

impl Default for Crc32 {
    fn default() -> Self {
        Self::new()
    }
}

impl Crc32 {
    pub fn new() -> Self {
        Self { state: 0xFFFF_FFFF }
    }

    /// Feeds bytes into the checksum.
    ///
    /// The table index is masked to 0..=255. `.get()` keeps this panic-free even under
    /// corruption: a miss contributes 0, which produces a detectable CRC mismatch
    /// downstream.
    pub fn update(&mut self, data: &[u8]) {
        for &byte in data {
            let index = ((self.state ^ u32::from(byte)) & 0xFF) as usize;
            let entry = CRC32_TABLE.get(index).copied().unwrap_or(0);
            self.state = entry ^ (self.state >> 8);
        }
    }

    pub fn finish(&self) -> u32 {
        self.state ^ 0xFFFF_FFFF
    }
}

/// Streaming Adler-32 (zlib trailer of PNGs we write).
#[derive(Copy, Clone, Debug)]
pub struct Adler32 {
    sum_a: u32,
    sum_b: u32,
}

impl Default for Adler32 {
    fn default() -> Self {
        Self::new()
    }
}

impl Adler32 {
    pub fn new() -> Self {
        Self { sum_a: 1, sum_b: 0 }
    }

    /// Feeds bytes. Reducing modulo 65521 per byte keeps both sums far below u32::MAX.
    pub fn update(&mut self, data: &[u8]) {
        for &byte in data {
            self.sum_a = (self.sum_a + u32::from(byte)) % ADLER32_MODULUS;
            self.sum_b = (self.sum_b + self.sum_a) % ADLER32_MODULUS;
        }
    }

    pub fn finish(&self) -> u32 {
        (self.sum_b << 16) | self.sum_a
    }
}

// ============================================================================
// PNG HEADER PARSER
// ============================================================================

/// Validated facts about one PNG file: everything the streaming decoder needs.
///
/// Fields are private so they cannot be modified to bypass validation.
#[derive(Copy, Clone, Debug)]
pub struct PngHeader {
    width: BoundedWidth,
    height: BoundedHeight,
    color_kind: PngColorKind,
    first_idat_offset: u64,
}

impl PngHeader {
    pub fn width(&self) -> BoundedWidth {
        self.width
    }
    pub fn height(&self) -> BoundedHeight {
        self.height
    }
    pub fn color_kind(&self) -> PngColorKind {
        self.color_kind
    }
    /// File offset of the first IDAT chunk's length field.
    pub fn first_idat_offset(&self) -> u64 {
        self.first_idat_offset
    }
}

/// Parses and validates the PNG signature and IHDR (including its CRC), then walks the
/// chunks to locate the first IDAT.
///
/// Policy:
/// - Ancillary chunks (tEXt, gAMA, ...) are skipped.
/// - PLTE is tolerated (it is only a suggested palette for colour types 2/6).
/// - Any other critical chunk is rejected, as the PNG spec requires.
/// - The file is read from offset 0 every call (stateless; safe to call on every retry).
pub fn parse_png_header(file: &mut File) -> Result<PngHeader, SlimerError> {
    if let Err(_e) = file.seek(SeekFrom::Start(0)) {
        #[cfg(debug_assertions)]
        eprintln!("PPH-202: seek to file start failed: {}", _e);
        return Err(SlimerError::IoSeekFailed);
    }

    let mut head = [0u8; PNG_SIGNATURE_AND_IHDR_LEN];
    if let Err(e) = file.read_exact(&mut head) {
        #[cfg(debug_assertions)]
        eprintln!("PPH-200: signature/IHDR read failed: {}", e);
        return Err(classify_read_error(e.kind()));
    }

    if !slice_matches_at(&head, 0, &PNG_SIGNATURE) {
        #[cfg(debug_assertions)]
        eprintln!("PPH-120: PNG signature invalid (file is not a PNG)");
        return Err(SlimerError::PngMagicInvalid);
    }

    let ihdr_length = read_be_u32_at(&head, 8)?;
    if ihdr_length != 13 || !slice_matches_at(&head, 12, b"IHDR") {
        #[cfg(debug_assertions)]
        eprintln!("PPH-125: first chunk is not a 13-byte IHDR");
        return Err(SlimerError::PngCorruptChunk);
    }

    // The CRC covers chunk type + data = bytes 12..29. The stored CRC is at 29..33.
    let crc_region = head.get(12..29).ok_or(SlimerError::ValIndexOutOfBounds)?;
    let mut ihdr_crc = Crc32::new();
    ihdr_crc.update(crc_region);
    let stored_crc = read_be_u32_at(&head, 29)?;
    if ihdr_crc.finish() != stored_crc {
        #[cfg(debug_assertions)]
        eprintln!("PPH-129: IHDR CRC mismatch (file corrupted)");
        return Err(SlimerError::PngIhdrCrcMismatch);
    }

    let raw_width = read_be_u32_at(&head, 16)?;
    let raw_height = read_be_u32_at(&head, 20)?;
    // Constant indices into a fixed [u8; 33]: checked by the compiler at build time.
    let bit_depth = head[24];
    let color_type_code = head[25];
    let compression_method = head[26];
    let filter_method = head[27];
    let interlace_method = head[28];

    if bit_depth != 8 {
        #[cfg(debug_assertions)]
        eprintln!("PPH-122: bit depth {} unsupported (8 only)", bit_depth);
        return Err(SlimerError::PngUnsupportedBitDepth);
    }
    let color_kind = PngColorKind::from_ihdr_code(color_type_code)?;
    if compression_method != 0 {
        #[cfg(debug_assertions)]
        eprintln!(
            "PPH-123: compression method {} unsupported",
            compression_method
        );
        return Err(SlimerError::PngUnsupportedCompression);
    }
    if filter_method != 0 {
        #[cfg(debug_assertions)]
        eprintln!("PPH-128: filter method {} unsupported", filter_method);
        return Err(SlimerError::PngFilterTypeInvalid);
    }
    if interlace_method != 0 {
        #[cfg(debug_assertions)]
        eprintln!("PPH-124: interlaced (Adam7) PNG unsupported");
        return Err(SlimerError::PngUnsupportedInterlace);
    }

    let width_pixels =
        usize::try_from(raw_width).map_err(|_| SlimerError::ValIntegrityDimOverflow)?;
    let height_pixels =
        usize::try_from(raw_height).map_err(|_| SlimerError::ValIntegrityDimOverflow)?;
    let width = BoundedWidth::new(width_pixels)?;
    let height = BoundedHeight::new(height_pixels)?;

    // Walk chunk headers until the first IDAT (bounded).
    let mut cursor = PNG_SIGNATURE_AND_IHDR_LEN as u64;
    for _chunk_index in 0..MAX_PNG_CHUNKS_BEFORE_IDAT {
        if let Err(_e) = file.seek(SeekFrom::Start(cursor)) {
            #[cfg(debug_assertions)]
            eprintln!("PPH-202: seek to chunk failed: {}", _e);
            return Err(SlimerError::IoSeekFailed);
        }
        let mut chunk_head = [0u8; 8];
        if let Err(e) = file.read_exact(&mut chunk_head) {
            #[cfg(debug_assertions)]
            eprintln!("PPH-200: chunk header read failed: {}", e);
            return Err(classify_read_error(e.kind()));
        }
        let chunk_length = read_be_u32_at(&chunk_head, 0)?;
        if chunk_length > PNG_MAX_CHUNK_LENGTH {
            #[cfg(debug_assertions)]
            eprintln!("PPH-125: chunk length {} exceeds PNG limit", chunk_length);
            return Err(SlimerError::PngCorruptChunk);
        }
        let chunk_type = [chunk_head[4], chunk_head[5], chunk_head[6], chunk_head[7]];

        if &chunk_type == b"IDAT" {
            return Ok(PngHeader {
                width,
                height,
                color_kind,
                first_idat_offset: cursor,
            });
        }
        if &chunk_type == b"IEND" {
            #[cfg(debug_assertions)]
            eprintln!("PPH-125: IEND reached before any IDAT (no image data)");
            return Err(SlimerError::PngCorruptChunk);
        }
        if &chunk_type == b"IHDR" {
            #[cfg(debug_assertions)]
            eprintln!("PPH-133: duplicate IHDR chunk");
            return Err(SlimerError::PngChunkSequenceInvalid);
        }
        // Bit 5 of the first type byte clear (uppercase) = critical chunk.
        let is_critical = (chunk_type[0] & 0x20) == 0;
        if is_critical && &chunk_type != b"PLTE" {
            #[cfg(debug_assertions)]
            eprintln!("PPH-136: unsupported critical chunk {:?}", chunk_type);
            return Err(SlimerError::PngUnsupportedCriticalChunk);
        }
        // Skip: 4 length + 4 type + data + 4 CRC.
        let skip = 12u64
            .checked_add(u64::from(chunk_length))
            .ok_or(SlimerError::RegArithmeticOverflow)?;
        cursor = cursor
            .checked_add(skip)
            .ok_or(SlimerError::RegArithmeticOverflow)?;
    }

    #[cfg(debug_assertions)]
    eprintln!(
        "PPH-125: no IDAT within {} chunks",
        MAX_PNG_CHUNKS_BEFORE_IDAT
    );
    Err(SlimerError::PngCorruptChunk)
}

// ============================================================================
// BUFFERED SEQUENTIAL FILE READER (4 KiB STACK BUFFER)
// ============================================================================

/// Sequential byte reader over a file region.
///
/// Replaces v1's one-syscall-per-byte reads. It only ever reads forward from the
/// starting offset, so at most 4 KiB of the file is held in memory at once.
struct BufferedSequentialReader<'a> {
    file: &'a mut File,
    buffer: [u8; FILE_READ_BUFFER_SIZE],
    buffer_position: usize,
    buffer_filled: usize,
}

impl<'a> BufferedSequentialReader<'a> {
    fn new_at_offset(file: &'a mut File, offset: u64) -> Result<Self, SlimerError> {
        if let Err(_e) = file.seek(SeekFrom::Start(offset)) {
            #[cfg(debug_assertions)]
            eprintln!("BSR-202: seek to offset {} failed: {}", offset, _e);
            return Err(SlimerError::IoSeekFailed);
        }
        Ok(Self {
            file,
            buffer: [0u8; FILE_READ_BUFFER_SIZE],
            buffer_position: 0,
            buffer_filled: 0,
        })
    }

    /// Refills the buffer.
    ///
    /// `Interrupted` is re-tried here (bounded). A read of 0 bytes means EOF, which is
    /// a truncated file.
    fn refill(&mut self) -> Result<(), SlimerError> {
        for _attempt in 0..MAX_IO_INTERRUPT_RETRIES {
            match self.file.read(&mut self.buffer) {
                Ok(0) => {
                    #[cfg(debug_assertions)]
                    eprintln!("BSR-203: unexpected end of file");
                    return Err(SlimerError::IoUnexpectedEof);
                }
                Ok(count) => {
                    if count > FILE_READ_BUFFER_SIZE {
                        return Err(SlimerError::ValIndexOutOfBounds);
                    }
                    self.buffer_position = 0;
                    self.buffer_filled = count;
                    return Ok(());
                }
                Err(e) if e.kind() == ErrorKind::Interrupted => continue,
                Err(e) => {
                    #[cfg(debug_assertions)]
                    eprintln!("BSR-200: read failed: {}", e);
                    return Err(classify_read_error(e.kind()));
                }
            }
        }
        #[cfg(debug_assertions)]
        eprintln!(
            "BSR-200: read interrupted {} times",
            MAX_IO_INTERRUPT_RETRIES
        );
        Err(SlimerError::IoReadFailed)
    }

    fn read_byte(&mut self) -> Result<u8, SlimerError> {
        if self.buffer_position >= self.buffer_filled {
            self.refill()?;
        }
        let value = byte_at(&self.buffer, self.buffer_position)?;
        self.buffer_position += 1; // < buffer_filled <= 4096: cannot overflow
        Ok(value)
    }

    fn read_into(&mut self, destination: &mut [u8]) -> Result<(), SlimerError> {
        for slot in destination.iter_mut() {
            *slot = self.read_byte()?;
        }
        Ok(())
    }
}

// ============================================================================
// MULTI-CHUNK IDAT BIT STREAM READER (WITH PER-CHUNK CRC)
// ============================================================================

/// Presents the concatenated payload of all IDAT chunks as one LSB-first bit stream
/// (the Deflate bit order).
///
/// The CRC of each chunk is verified when the reader crosses into the next chunk.
/// Zero-length IDAT chunks are legal and are skipped; this was a v1 underflow bug.
struct IdatBitReader<'a> {
    source: BufferedSequentialReader<'a>,
    chunk_bytes_remaining: u32,
    chunk_crc: Crc32,
    bit_buffer: u32,
    bits_in_buffer: u32,
}

impl<'a> IdatBitReader<'a> {
    fn new(file: &'a mut File, first_idat_offset: u64) -> Result<Self, SlimerError> {
        let source = BufferedSequentialReader::new_at_offset(file, first_idat_offset)?;
        let mut reader = Self {
            source,
            chunk_bytes_remaining: 0,
            chunk_crc: Crc32::new(),
            bit_buffer: 0,
            bits_in_buffer: 0,
        };
        reader.open_next_idat_chunk()?;
        Ok(reader)
    }

    /// Reads the next chunk header, which must be IDAT (the spec requires consecutive IDATs).
    fn open_next_idat_chunk(&mut self) -> Result<(), SlimerError> {
        let mut chunk_head = [0u8; 8];
        self.source.read_into(&mut chunk_head)?;
        let length = read_be_u32_at(&chunk_head, 0)?;
        if !slice_matches_at(&chunk_head, 4, b"IDAT") {
            #[cfg(debug_assertions)]
            eprintln!("IBR-133: expected IDAT, found {:?}", chunk_head.get(4..8));
            return Err(SlimerError::PngChunkSequenceInvalid);
        }
        if length > PNG_MAX_CHUNK_LENGTH {
            #[cfg(debug_assertions)]
            eprintln!("IBR-125: IDAT length {} exceeds PNG limit", length);
            return Err(SlimerError::PngCorruptChunk);
        }
        self.chunk_bytes_remaining = length;
        self.chunk_crc = Crc32::new();
        self.chunk_crc.update(b"IDAT");
        Ok(())
    }

    /// Reads the 4-byte CRC that follows a fully consumed IDAT payload and compares it.
    fn verify_current_chunk_crc(&mut self) -> Result<(), SlimerError> {
        let mut stored = [0u8; 4];
        self.source.read_into(&mut stored)?;
        if u32::from_be_bytes(stored) != self.chunk_crc.finish() {
            #[cfg(debug_assertions)]
            eprintln!("IBR-130: IDAT CRC mismatch (file corrupted)");
            return Err(SlimerError::PngIdatCrcMismatch);
        }
        Ok(())
    }

    /// Next compressed byte, crossing chunk boundaries (and empty chunks) as needed.
    fn next_compressed_byte(&mut self) -> Result<u8, SlimerError> {
        let mut chunks_crossed: usize = 0;
        while self.chunk_bytes_remaining == 0 {
            if chunks_crossed >= MAX_CONSECUTIVE_EMPTY_IDAT_CHUNKS {
                #[cfg(debug_assertions)]
                eprintln!("IBR-133: too many consecutive empty IDAT chunks");
                return Err(SlimerError::PngChunkSequenceInvalid);
            }
            chunks_crossed += 1;
            self.verify_current_chunk_crc()?;
            self.open_next_idat_chunk()?;
        }
        let value = self.source.read_byte()?;
        self.chunk_crc.update(&[value]);
        self.chunk_bytes_remaining = self
            .chunk_bytes_remaining
            .checked_sub(1)
            .ok_or(SlimerError::ValCorruptedBitFlipDetected)?;
        Ok(value)
    }

    /// Reads `count` bits (0..=24), LSB first.
    ///
    /// Invariant: `bits_in_buffer < 8` between calls. Shift-amount checks guard
    /// against a corrupted value of that field.
    fn read_bits(&mut self, count: u32) -> Result<u32, SlimerError> {
        if count > MAX_BITS_PER_READ {
            return Err(SlimerError::ValArgumentInvalid);
        }
        let mut refills: usize = 0;
        while self.bits_in_buffer < count {
            if refills >= 4 {
                return Err(SlimerError::ValCorruptedBitFlipDetected);
            }
            refills += 1;
            let next = self.next_compressed_byte()?;
            let shifted = u32::from(next)
                .checked_shl(self.bits_in_buffer)
                .ok_or(SlimerError::ValCorruptedBitFlipDetected)?;
            self.bit_buffer |= shifted;
            self.bits_in_buffer += 8; // < 32: bounded by the count <= 24 precondition
        }
        let mask = if count == 0 { 0 } else { (1u32 << count) - 1 };
        let value = self.bit_buffer & mask;
        self.bit_buffer = self.bit_buffer.checked_shr(count).unwrap_or(0);
        self.bits_in_buffer -= count; // >= count guaranteed by the loop above
        Ok(value)
    }

    /// Discards bits up to the next byte boundary (needed before stored-block headers).
    fn align_to_byte(&mut self) -> Result<(), SlimerError> {
        let discard = self.bits_in_buffer % 8;
        self.bit_buffer = self.bit_buffer.checked_shr(discard).unwrap_or(0);
        self.bits_in_buffer -= discard;
        Ok(())
    }
}

// ============================================================================
// CANONICAL HUFFMAN TABLE (ZERO HEAP)
// ============================================================================

/// Canonical Huffman decoding table (RFC 1951 3.2.2), "count + sorted symbols" form.
#[derive(Copy, Clone)]
struct HuffmanTable {
    counts: [u16; HUFFMAN_COUNT_SLOTS],
    symbols: [u16; MAX_LITLEN_SYMBOLS],
}

impl HuffmanTable {
    fn new() -> Self {
        Self {
            counts: [0; HUFFMAN_COUNT_SLOTS],
            symbols: [0; MAX_LITLEN_SYMBOLS],
        }
    }

    /// Builds the table from per-symbol code lengths.
    ///
    /// Rejects lengths > 15 and over-subscribed codes; the v1 code accepted both.
    /// Incomplete codes are allowed, as zlib allows them (e.g. a single distance code).
    fn build(&mut self, code_lengths: &[u8]) -> Result<(), SlimerError> {
        if code_lengths.len() > MAX_LITLEN_SYMBOLS {
            #[cfg(debug_assertions)]
            eprintln!(
                "HUF-306: {} code lengths exceed capacity",
                code_lengths.len()
            );
            return Err(SlimerError::ValArgumentInvalid);
        }
        self.counts = [0; HUFFMAN_COUNT_SLOTS];
        self.symbols = [0; MAX_LITLEN_SYMBOLS];

        for &code_length in code_lengths {
            let count_slot = match self.counts.get_mut(usize::from(code_length)) {
                Some(slot) => slot,
                None => {
                    #[cfg(debug_assertions)]
                    eprintln!("HUF-131: code length {} exceeds 15", code_length);
                    return Err(SlimerError::PngHuffmanTableInvalid);
                }
            };
            *count_slot = count_slot
                .checked_add(1)
                .ok_or(SlimerError::PngHuffmanTableInvalid)?;
        }
        // Length 0 means "symbol unused": it occupies no code space.
        self.counts[0] = 0;

        let mut remaining_code_space: i32 = 1;
        for code_length in 1..HUFFMAN_COUNT_SLOTS {
            remaining_code_space <<= 1; // at most 2^15: no overflow
            let count = *self
                .counts
                .get(code_length)
                .ok_or(SlimerError::ValIndexOutOfBounds)?;
            remaining_code_space -= i32::from(count);
            if remaining_code_space < 0 {
                #[cfg(debug_assertions)]
                eprintln!("HUF-131: over-subscribed Huffman code");
                return Err(SlimerError::PngHuffmanTableInvalid);
            }
        }

        let mut next_offset = [0u16; HUFFMAN_COUNT_SLOTS];
        for code_length in 1..(HUFFMAN_COUNT_SLOTS - 1) {
            let base = *next_offset
                .get(code_length)
                .ok_or(SlimerError::ValIndexOutOfBounds)?;
            let count = *self
                .counts
                .get(code_length)
                .ok_or(SlimerError::ValIndexOutOfBounds)?;
            let slot = next_offset
                .get_mut(code_length + 1)
                .ok_or(SlimerError::ValIndexOutOfBounds)?;
            *slot = base
                .checked_add(count)
                .ok_or(SlimerError::PngHuffmanTableInvalid)?;
        }

        for (symbol, &code_length) in code_lengths.iter().enumerate() {
            if code_length == 0 {
                continue;
            }
            let offset_slot = next_offset
                .get_mut(usize::from(code_length))
                .ok_or(SlimerError::ValIndexOutOfBounds)?;
            let symbol_slot = self
                .symbols
                .get_mut(usize::from(*offset_slot))
                .ok_or(SlimerError::PngHuffmanTableInvalid)?;
            *symbol_slot =
                u16::try_from(symbol).map_err(|_| SlimerError::PngHuffmanTableInvalid)?;
            *offset_slot = offset_slot
                .checked_add(1)
                .ok_or(SlimerError::PngHuffmanTableInvalid)?;
        }
        Ok(())
    }

    /// Decodes one symbol, reading the code bit by bit (MSB of the code first).
    fn decode(&self, reader: &mut IdatBitReader<'_>) -> Result<u16, SlimerError> {
        let mut code: i32 = 0;
        let mut first: i32 = 0;
        let mut index: i32 = 0;
        for code_length in 1..HUFFMAN_COUNT_SLOTS {
            code |= reader.read_bits(1)? as i32;
            let count = i32::from(
                *self
                    .counts
                    .get(code_length)
                    .ok_or(SlimerError::ValIndexOutOfBounds)?,
            );
            if code - count < first {
                let symbol_index = usize::try_from(index + (code - first))
                    .map_err(|_| SlimerError::PngDeflateCorrupt)?;
                return self
                    .symbols
                    .get(symbol_index)
                    .copied()
                    .ok_or(SlimerError::PngDeflateCorrupt);
            }
            index += count;
            first = (first + count) << 1;
            code <<= 1;
        }
        #[cfg(debug_assertions)]
        eprintln!("HUF-127: no valid Huffman code within 15 bits");
        Err(SlimerError::PngDeflateCorrupt)
    }
}

/// Fixed literal/length code lengths (RFC 1951 3.2.6). Total table builder.
fn fixed_litlen_code_lengths() -> [u8; MAX_LITLEN_SYMBOLS] {
    let mut lengths = [0u8; MAX_LITLEN_SYMBOLS];
    for (symbol, slot) in lengths.iter_mut().enumerate() {
        *slot = match symbol {
            0..=143 => 8,
            144..=255 => 9,
            256..=279 => 7,
            _ => 8,
        };
    }
    lengths
}

// ============================================================================
// STREAMING DEFLATE DECOMPRESSOR (ZERO HEAP)
// ============================================================================

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
enum DeflateBlockState {
    /// The next step must read a block header.
    AwaitingHeader,
    /// Inside a stored (type 0) block; `stored_remaining` raw bytes are left.
    Stored,
    /// Inside a fixed or dynamic Huffman block.
    Huffman,
}

/// Pull-based zlib/Deflate decoder: produces one decompressed byte per call.
///
/// The 32 KiB history window is the only large buffer.
///
/// v2 fixes:
/// - stored blocks now close correctly;
/// - back-reference distances are checked against the history actually produced;
/// - dynamic-header repeat codes are bounds-checked.
pub struct DeflateByteStream<'a> {
    reader: IdatBitReader<'a>,
    window: [u8; DEFLATE_WINDOW_SIZE],
    window_position: usize,
    bytes_produced: u64,
    litlen_table: HuffmanTable,
    distance_table: HuffmanTable,
    match_remaining: u16,
    match_distance: u16,
    stored_remaining: u16,
    block_state: DeflateBlockState,
    final_block_seen: bool,
}

impl<'a> DeflateByteStream<'a> {
    /// Opens the zlib stream at the first IDAT chunk and validates the 2-byte zlib header.
    pub fn new(file: &'a mut File, first_idat_offset: u64) -> Result<Self, SlimerError> {
        let mut reader = IdatBitReader::new(file, first_idat_offset)?;
        let cmf = reader.read_bits(8)?;
        let flg = reader.read_bits(8)?;
        let header_check = (cmf * 256 + flg) % 31;
        let compression_method = cmf & 0x0F;
        let window_log = cmf >> 4;
        if header_check != 0 || compression_method != 8 || window_log > 7 {
            #[cfg(debug_assertions)]
            eprintln!("DBS-126: invalid zlib header CMF={:#x} FLG={:#x}", cmf, flg);
            return Err(SlimerError::PngZlibHeaderInvalid);
        }
        if flg & 0x20 != 0 {
            #[cfg(debug_assertions)]
            eprintln!("DBS-137: zlib preset dictionary not allowed in PNG");
            return Err(SlimerError::PngZlibPresetDictionary);
        }
        Ok(Self {
            reader,
            window: [0u8; DEFLATE_WINDOW_SIZE],
            window_position: 0,
            bytes_produced: 0,
            litlen_table: HuffmanTable::new(),
            distance_table: HuffmanTable::new(),
            match_remaining: 0,
            match_distance: 0,
            stored_remaining: 0,
            block_state: DeflateBlockState::AwaitingHeader,
            final_block_seen: false,
        })
    }

    /// Records an output byte in the history window.
    fn push_output_byte(&mut self, value: u8) -> Result<(), SlimerError> {
        let position = self.window_position % DEFLATE_WINDOW_SIZE;
        let slot = self
            .window
            .get_mut(position)
            .ok_or(SlimerError::ValIndexOutOfBounds)?;
        *slot = value;
        self.window_position = (position + 1) % DEFLATE_WINDOW_SIZE;
        self.bytes_produced = self.bytes_produced.saturating_add(1);
        Ok(())
    }

    /// Pulls exactly one decompressed byte from the Deflate stream.
    pub fn next_byte(&mut self) -> Result<u8, SlimerError> {
        for _step in 0..MAX_DEFLATE_STEPS_PER_BYTE {
            // 1. Drain an active LZ77 back-reference.
            if self.match_remaining > 0 {
                let distance = usize::from(self.match_distance);
                if distance == 0 || distance > DEFLATE_WINDOW_SIZE {
                    return Err(SlimerError::ValCorruptedBitFlipDetected);
                }
                let position = self.window_position % DEFLATE_WINDOW_SIZE;
                let source_index =
                    (position + DEFLATE_WINDOW_SIZE - distance) % DEFLATE_WINDOW_SIZE;
                let value = byte_at(&self.window, source_index)?;
                self.push_output_byte(value)?;
                self.match_remaining -= 1;
                return Ok(value);
            }

            match self.block_state {
                // 2. Drain a stored (uncompressed) block; close it when empty.
                DeflateBlockState::Stored => {
                    if self.stored_remaining > 0 {
                        let value = self.reader.read_bits(8)? as u8;
                        self.push_output_byte(value)?;
                        self.stored_remaining -= 1;
                        return Ok(value);
                    }
                    self.block_state = DeflateBlockState::AwaitingHeader;
                }
                // 3. Open the next block, unless the final block has already ended.
                DeflateBlockState::AwaitingHeader => {
                    if self.final_block_seen {
                        #[cfg(debug_assertions)]
                        eprintln!("DBS-135: read past final Deflate block (image data truncated)");
                        return Err(SlimerError::PngImageDataTruncated);
                    }
                    self.open_next_block()?;
                }
                // 4. Decode a literal, an end-of-block, or a length/distance pair.
                DeflateBlockState::Huffman => {
                    let symbol = self.litlen_table.decode(&mut self.reader)?;
                    if symbol < 256 {
                        let value = symbol as u8;
                        self.push_output_byte(value)?;
                        return Ok(value);
                    }
                    if symbol == 256 {
                        self.block_state = DeflateBlockState::AwaitingHeader;
                    } else {
                        self.begin_back_reference(symbol)?;
                    }
                }
            }
        }
        #[cfg(debug_assertions)]
        eprintln!("DBS-127: step limit exceeded without producing a byte");
        Err(SlimerError::PngDeflateCorrupt)
    }

    /// Reads BFINAL/BTYPE and prepares the block.
    fn open_next_block(&mut self) -> Result<(), SlimerError> {
        self.final_block_seen = self.reader.read_bits(1)? == 1;
        let block_type = self.reader.read_bits(2)?;
        match block_type {
            0 => {
                self.reader.align_to_byte()?;
                let length = self.reader.read_bits(16)?;
                let length_complement = self.reader.read_bits(16)?;
                if (length ^ 0xFFFF) != length_complement {
                    #[cfg(debug_assertions)]
                    eprintln!(
                        "DBS-127: stored LEN {} / NLEN {} mismatch",
                        length, length_complement
                    );
                    return Err(SlimerError::PngDeflateCorrupt);
                }
                self.stored_remaining =
                    u16::try_from(length).map_err(|_| SlimerError::PngDeflateCorrupt)?;
                self.block_state = DeflateBlockState::Stored;
            }
            1 => {
                let litlen_lengths = fixed_litlen_code_lengths();
                self.litlen_table.build(&litlen_lengths)?;
                let distance_lengths = [5u8; MAX_DIST_SYMBOLS];
                self.distance_table.build(&distance_lengths)?;
                self.block_state = DeflateBlockState::Huffman;
            }
            2 => {
                self.read_dynamic_tables()?;
                self.block_state = DeflateBlockState::Huffman;
            }
            _ => {
                #[cfg(debug_assertions)]
                eprintln!("DBS-127: reserved block type 3");
                return Err(SlimerError::PngDeflateCorrupt);
            }
        }
        Ok(())
    }

    /// Reads a dynamic-Huffman block header (RFC 1951 3.2.7) with strict validation.
    fn read_dynamic_tables(&mut self) -> Result<(), SlimerError> {
        let litlen_count = self.reader.read_bits(5)? as usize + 257;
        let distance_count = self.reader.read_bits(5)? as usize + 1;
        let code_length_count = self.reader.read_bits(4)? as usize + 4;
        if litlen_count > 286 || distance_count > 30 {
            #[cfg(debug_assertions)]
            eprintln!(
                "DBS-127: HLIT {} / HDIST {} out of range",
                litlen_count, distance_count
            );
            return Err(SlimerError::PngDeflateCorrupt);
        }

        let mut code_length_lengths = [0u8; CODE_LENGTH_SYMBOLS];
        for &symbol_position in CODE_LENGTH_ORDER.iter().take(code_length_count) {
            let value = self.reader.read_bits(3)? as u8;
            let slot = code_length_lengths
                .get_mut(symbol_position)
                .ok_or(SlimerError::ValIndexOutOfBounds)?;
            *slot = value;
        }
        let mut code_length_table = HuffmanTable::new();
        code_length_table.build(&code_length_lengths)?;

        let mut combined_lengths = [0u8; MAX_LITLEN_SYMBOLS + MAX_DIST_SYMBOLS];
        let total = litlen_count + distance_count;
        let mut filled: usize = 0;
        // Every iteration fills >= 1 entry, so this bound always suffices.
        for _step in 0..(MAX_LITLEN_SYMBOLS + MAX_DIST_SYMBOLS) {
            if filled >= total {
                break;
            }
            let symbol = code_length_table.decode(&mut self.reader)?;
            let (value, repeat) = match symbol {
                0..=15 => (symbol as u8, 1usize),
                16 => {
                    if filled == 0 {
                        #[cfg(debug_assertions)]
                        eprintln!("DBS-127: repeat-previous code with no previous length");
                        return Err(SlimerError::PngDeflateCorrupt);
                    }
                    let previous = byte_at(&combined_lengths, filled - 1)?;
                    (previous, 3 + self.reader.read_bits(2)? as usize)
                }
                17 => (0u8, 3 + self.reader.read_bits(3)? as usize),
                18 => (0u8, 11 + self.reader.read_bits(7)? as usize),
                _ => {
                    #[cfg(debug_assertions)]
                    eprintln!("DBS-127: invalid code-length symbol {}", symbol);
                    return Err(SlimerError::PngDeflateCorrupt);
                }
            };
            let end = filled
                .checked_add(repeat)
                .ok_or(SlimerError::PngDeflateCorrupt)?;
            if end > total {
                #[cfg(debug_assertions)]
                eprintln!(
                    "DBS-127: code-length repeat overruns table ({} > {})",
                    end, total
                );
                return Err(SlimerError::PngDeflateCorrupt);
            }
            let destination = combined_lengths
                .get_mut(filled..end)
                .ok_or(SlimerError::ValIndexOutOfBounds)?;
            for slot in destination.iter_mut() {
                *slot = value;
            }
            filled = end;
        }
        if filled != total {
            return Err(SlimerError::PngDeflateCorrupt);
        }

        let litlen_lengths = combined_lengths
            .get(..litlen_count)
            .ok_or(SlimerError::ValIndexOutOfBounds)?;
        if byte_at(litlen_lengths, 256)? == 0 {
            #[cfg(debug_assertions)]
            eprintln!("DBS-127: end-of-block symbol has no code");
            return Err(SlimerError::PngDeflateCorrupt);
        }
        self.litlen_table.build(litlen_lengths)?;
        let distance_lengths = combined_lengths
            .get(litlen_count..total)
            .ok_or(SlimerError::ValIndexOutOfBounds)?;
        self.distance_table.build(distance_lengths)?;
        Ok(())
    }

    /// Decodes length + distance for a back-reference and validates the distance
    /// against the history produced so far.
    fn begin_back_reference(&mut self, length_symbol: u16) -> Result<(), SlimerError> {
        let length_index = usize::from(length_symbol)
            .checked_sub(257)
            .ok_or(SlimerError::PngDeflateCorrupt)?;
        let length_base = DEFLATE_LENGTH_BASES
            .get(length_index)
            .copied()
            .ok_or(SlimerError::PngDeflateCorrupt)?;
        let length_extra_bits = DEFLATE_LENGTH_EXTRA_BITS
            .get(length_index)
            .copied()
            .ok_or(SlimerError::PngDeflateCorrupt)?;
        let match_length =
            u32::from(length_base) + self.reader.read_bits(u32::from(length_extra_bits))?;

        let distance_symbol = usize::from(self.distance_table.decode(&mut self.reader)?);
        let distance_base = DEFLATE_DISTANCE_BASES
            .get(distance_symbol)
            .copied()
            .ok_or(SlimerError::PngDeflateCorrupt)?;
        let distance_extra_bits = DEFLATE_DISTANCE_EXTRA_BITS
            .get(distance_symbol)
            .copied()
            .ok_or(SlimerError::PngDeflateCorrupt)?;
        let distance =
            u32::from(distance_base) + self.reader.read_bits(u32::from(distance_extra_bits))?;

        let available_history = self.bytes_produced.min(DEFLATE_WINDOW_SIZE as u64);
        if distance == 0 || u64::from(distance) > available_history {
            #[cfg(debug_assertions)]
            eprintln!(
                "DBS-132: distance {} exceeds history {}",
                distance, available_history
            );
            return Err(SlimerError::PngDeflateDistanceTooFar);
        }
        self.match_remaining =
            u16::try_from(match_length).map_err(|_| SlimerError::PngDeflateCorrupt)?;
        self.match_distance =
            u16::try_from(distance).map_err(|_| SlimerError::PngDeflateCorrupt)?;
        Ok(())
    }
}

// ============================================================================
// PNG SCANLINE RECONSTRUCTION & LUMA CONVERSION
// ============================================================================

/// PNG Paeth predictor (PNG spec section 9.4).
pub fn png_paeth_predictor(left: u8, up: u8, up_left: u8) -> Result<u8, SlimerError> {
    let a = i16::from(left);
    let b = i16::from(up);
    let c = i16::from(up_left);
    let estimate = a + b - c; // range -255..=510: fits i16
    let distance_a = (estimate - a).abs();
    let distance_b = (estimate - b).abs();
    let distance_c = (estimate - c).abs();
    if distance_a <= distance_b && distance_a <= distance_c {
        Ok(left)
    } else if distance_b <= distance_c {
        Ok(up)
    } else {
        Ok(up_left)
    }
}

/// Reverses one PNG scanline filter for a single byte.
///
/// Neighbours:
/// - `left` is the reconstructed byte one pixel to the left (0 at the row start);
/// - `up` is the byte above;
/// - `up_left` is the byte above-left.
///
/// Unknown filter types are an error (v1 passed them through silently).
pub fn png_unfilter_byte(
    filtered: u8,
    filter_type: u8,
    left: u8,
    up: u8,
    up_left: u8,
) -> Result<u8, SlimerError> {
    match filter_type {
        0 => Ok(filtered),
        1 => Ok(filtered.wrapping_add(left)),
        2 => Ok(filtered.wrapping_add(up)),
        3 => {
            let average = (u16::from(left) + u16::from(up)) / 2;
            Ok(filtered.wrapping_add(average as u8)) // average <= 255
        }
        4 => Ok(filtered.wrapping_add(png_paeth_predictor(left, up, up_left)?)),
        _ => {
            #[cfg(debug_assertions)]
            eprintln!("PUB-134: invalid scanline filter type {}", filter_type);
            Err(SlimerError::PngScanlineFilterInvalid)
        }
    }
}

/// Converts one pixel to 8-bit luma.
///
/// - Alpha is ignored.
/// - RGB uses BT.601 integer weights (77, 150, 29) / 256.
/// - The pixel length must match the colour kind.
fn luma_from_pixel(color_kind: PngColorKind, pixel: &[u8]) -> Result<u8, SlimerError> {
    match (color_kind, pixel) {
        (PngColorKind::Gray, [gray]) | (PngColorKind::GrayAlpha, [gray, _]) => Ok(*gray),
        (PngColorKind::Rgb, [red, green, blue]) | (PngColorKind::Rgba, [red, green, blue, _]) => {
            let weighted = u32::from(*red) * LUMA_WEIGHT_RED
                + u32::from(*green) * LUMA_WEIGHT_GREEN
                + u32::from(*blue) * LUMA_WEIGHT_BLUE;
            u8::try_from(weighted >> 8).map_err(|_| SlimerError::RegArithmeticOverflow)
        }
        _ => Err(SlimerError::ValIndexOutOfBounds),
    }
}

/// Streams a PNG one decoded luma row at a time.
///
/// Holds only two raw scanlines (current and previous, needed by the PNG filters).
pub struct PngScanlineReader<'a> {
    deflate: DeflateByteStream<'a>,
    previous_row: [u8; MAX_SCANLINE_BYTES],
    current_row: [u8; MAX_SCANLINE_BYTES],
    width: BoundedWidth,
    height: BoundedHeight,
    color_kind: PngColorKind,
    rows_decoded: usize,
}

impl<'a> PngScanlineReader<'a> {
    pub fn new(file: &'a mut File, header: &PngHeader) -> Result<Self, SlimerError> {
        header.width().validity_recheck()?;
        header.height().validity_recheck()?;
        let deflate = DeflateByteStream::new(file, header.first_idat_offset())?;
        Ok(Self {
            deflate,
            previous_row: [0u8; MAX_SCANLINE_BYTES],
            current_row: [0u8; MAX_SCANLINE_BYTES],
            width: header.width(),
            height: header.height(),
            color_kind: header.color_kind(),
            rows_decoded: 0,
        })
    }

    /// Decodes the next row into `out_luma[..width]`.
    ///
    /// Returns:
    /// - `Ok(true)` when a row was produced;
    /// - `Ok(false)` once all `height` rows have been delivered;
    /// - `Err` on corruption or I/O failure.
    pub fn next_luma_row(
        &mut self,
        out_luma: &mut [u8; MAX_IMAGE_WIDTH],
    ) -> Result<bool, SlimerError> {
        let width = self.width.get()?;
        let height = self.height.get()?;
        if self.rows_decoded >= height {
            return Ok(false);
        }
        let bytes_per_pixel = self.color_kind.bytes_per_pixel();
        let row_bytes = match width.checked_mul(bytes_per_pixel) {
            Some(count) if count <= MAX_SCANLINE_BYTES => count,
            _ => return Err(SlimerError::ValIndexOutOfBounds),
        };

        let filter_type = self.deflate.next_byte()?;
        if filter_type > 4 {
            #[cfg(debug_assertions)]
            eprintln!(
                "PSR-134: row {} has invalid filter type {}",
                self.rows_decoded, filter_type
            );
            return Err(SlimerError::PngScanlineFilterInvalid);
        }

        for x in 0..row_bytes {
            let filtered = self.deflate.next_byte()?;
            let (left, up_left) = if x >= bytes_per_pixel {
                let neighbour = x - bytes_per_pixel;
                (
                    byte_at(&self.current_row, neighbour)?,
                    byte_at(&self.previous_row, neighbour)?,
                )
            } else {
                (0, 0)
            };
            let up = byte_at(&self.previous_row, x)?;
            let reconstructed = png_unfilter_byte(filtered, filter_type, left, up, up_left)?;
            let slot = self
                .current_row
                .get_mut(x)
                .ok_or(SlimerError::ValIndexOutOfBounds)?;
            *slot = reconstructed;
        }

        let color_kind = self.color_kind;
        let pixel_bytes = self
            .current_row
            .get(..row_bytes)
            .ok_or(SlimerError::ValIndexOutOfBounds)?;
        // bytes_per_pixel >= 1 always (enum mapping), so chunks_exact cannot panic.
        for (luma_slot, pixel) in out_luma
            .iter_mut()
            .zip(pixel_bytes.chunks_exact(bytes_per_pixel))
        {
            *luma_slot = luma_from_pixel(color_kind, pixel)?;
        }

        let source = self
            .current_row
            .get(..row_bytes)
            .ok_or(SlimerError::ValIndexOutOfBounds)?;
        let destination = self
            .previous_row
            .get_mut(..row_bytes)
            .ok_or(SlimerError::ValIndexOutOfBounds)?;
        destination.copy_from_slice(source); // equal lengths by construction

        self.rows_decoded += 1; // <= height <= 1024
        Ok(true)
    }
}

// ============================================================================
// 1D PROJECTION SIGNATURES & SHIFT SOLVER (REGISTRATION KERNEL)
// ============================================================================

/// Row-sum and column-sum luma signatures of one image (16 KiB on the stack).
///
/// Max value per entry: 1024 * 255, far below u64::MAX.
pub struct ProjectionSignatures {
    row_sums: [u64; MAX_IMAGE_HEIGHT],
    col_sums: [u64; MAX_IMAGE_WIDTH],
}

impl Default for ProjectionSignatures {
    fn default() -> Self {
        Self::new()
    }
}

impl ProjectionSignatures {
    pub fn new() -> Self {
        Self {
            row_sums: [0u64; MAX_IMAGE_HEIGHT],
            col_sums: [0u64; MAX_IMAGE_WIDTH],
        }
    }

    pub fn row_sums(&self) -> &[u64] {
        &self.row_sums
    }

    pub fn col_sums(&self) -> &[u64] {
        &self.col_sums
    }

    /// Streams the whole PNG once and accumulates both signatures.
    ///
    /// A missing row is an error (the image is truncated). Any rows already
    /// accumulated are discarded on failure.
    pub fn compute_from_png_stream(
        &mut self,
        file: &mut File,
        header: &PngHeader,
    ) -> Result<(), SlimerError> {
        let width = header.width().get()?;
        let height = header.height().get()?;
        self.row_sums = [0u64; MAX_IMAGE_HEIGHT];
        self.col_sums = [0u64; MAX_IMAGE_WIDTH];

        let mut reader = PngScanlineReader::new(file, header)?;
        let mut luma_row = [0u8; MAX_IMAGE_WIDTH];
        for y in 0..height {
            if !reader.next_luma_row(&mut luma_row)? {
                #[cfg(debug_assertions)]
                eprintln!("PJS-135: image ended at row {} of {}", y, height);
                return Err(SlimerError::PngImageDataTruncated);
            }
            let pixels = luma_row
                .get(..width)
                .ok_or(SlimerError::ValIndexOutOfBounds)?;
            let mut row_total: u64 = 0;
            for (col_sum, &luma) in self.col_sums.iter_mut().zip(pixels.iter()) {
                row_total = row_total
                    .checked_add(u64::from(luma))
                    .ok_or(SlimerError::RegArithmeticOverflow)?;
                *col_sum = col_sum
                    .checked_add(u64::from(luma))
                    .ok_or(SlimerError::RegArithmeticOverflow)?;
            }
            let row_slot = self
                .row_sums
                .get_mut(y)
                .ok_or(SlimerError::ValIndexOutOfBounds)?;
            *row_slot = row_total;
        }
        Ok(())
    }
}

/// Finds the shift `d` that best aligns `signature_a[i]` with `signature_b[i + d]`.
///
/// Policy:
/// - SAD is normalised by the overlap count.
/// - The effective radius is `min(radius, MAX_SEARCH_RADIUS, length / 2)`, so at least
///   half of each signature overlaps (tiny overlaps gave spurious v1 matches).
/// - Ties prefer the smallest |d|. A featureless signal therefore yields shift 0.
///
/// Returns `(shift, confidence_ppm)`.
pub fn solve_1d_shift(
    signature_a: &[u64],
    signature_b: &[u64],
    length: usize,
    radius: i32,
) -> Result<(i32, u32), SlimerError> {
    if length < MIN_IMAGE_DIMENSION {
        #[cfg(debug_assertions)]
        eprintln!(
            "S1D-300: length {} below minimum {}",
            length, MIN_IMAGE_DIMENSION
        );
        return Err(SlimerError::ValIntegrityDimUnderflow);
    }
    if radius < 0 {
        #[cfg(debug_assertions)]
        eprintln!("S1D-306: negative search radius {}", radius);
        return Err(SlimerError::ValArgumentInvalid);
    }
    let a = signature_a
        .get(..length)
        .ok_or(SlimerError::ValArgumentInvalid)?;
    let b = signature_b
        .get(..length)
        .ok_or(SlimerError::ValArgumentInvalid)?;
    let half_length = i32::try_from(length / 2).map_err(|_| SlimerError::ValArgumentInvalid)?;
    let effective_radius = radius.min(MAX_SEARCH_RADIUS).min(half_length);

    let mut best_shift: i32 = 0;
    let mut best_sad: u64 = u64::MAX;
    let mut worst_sad: u64 = 0;
    let mut any_candidate = false;

    for shift in -effective_radius..=effective_radius {
        let mut sad_total: u64 = 0;
        let mut overlap: u64 = 0;
        for (index_a, &value_a) in a.iter().enumerate() {
            let index_b = index_a as i64 + i64::from(shift); // index_a <= 1024
            if index_b < 0 {
                continue;
            }
            if let Some(&value_b) = b.get(index_b as usize) {
                sad_total = sad_total.saturating_add(value_a.abs_diff(value_b));
                overlap += 1;
            }
        }
        if overlap == 0 {
            continue;
        }
        any_candidate = true;
        let normalized = sad_total / overlap;
        if normalized < best_sad || (normalized == best_sad && shift.abs() < best_shift.abs()) {
            best_sad = normalized;
            best_shift = shift;
        }
        if normalized > worst_sad {
            worst_sad = normalized;
        }
    }

    if !any_candidate {
        return Ok((0, 0));
    }
    let confidence = if worst_sad > best_sad && worst_sad > 0 {
        let numerator = u128::from(worst_sad - best_sad) * u128::from(CONFIDENCE_PPM_MAX);
        let ppm = (numerator / u128::from(worst_sad)).min(u128::from(CONFIDENCE_PPM_MAX));
        u32::try_from(ppm).map_err(|_| SlimerError::ValIntegrityScoreCorrupt)?
    } else {
        0
    };
    Ok((best_shift, confidence))
}

/// Both images must share dimensions (MVP scope: same sensor / same framing).
fn require_matching_dimensions(
    header_a: &PngHeader,
    header_b: &PngHeader,
) -> Result<(usize, usize), SlimerError> {
    let width_a = header_a.width().get()?;
    let height_a = header_a.height().get()?;
    let width_b = header_b.width().get()?;
    let height_b = header_b.height().get()?;
    if width_a != width_b || height_a != height_b {
        #[cfg(debug_assertions)]
        eprintln!(
            "RMD-400: {}x{} vs {}x{}",
            width_a, height_a, width_b, height_b
        );
        return Err(SlimerError::RegDimMismatch);
    }
    Ok((width_a, height_a))
}

/// Registers the target image (file_b) against the reference image (file_a).
///
/// The combined confidence is the average of both axes. A featureless axis contributes
/// a shift of 0 (tie rule) and 0 confidence, lowering the combined score.
///
/// `RegInsufficientContrast` is degradable: the caller decides whether to use
/// `RegistrationShift::identity_fallback()`.
pub fn register_png_images(
    file_a: &mut File,
    file_b: &mut File,
) -> Result<RegistrationShift, SlimerError> {
    let header_a = parse_png_header(file_a)?;
    let header_b = parse_png_header(file_b)?;
    let (width, height) = require_matching_dimensions(&header_a, &header_b)?;

    let mut signatures_a = ProjectionSignatures::new();
    let mut signatures_b = ProjectionSignatures::new();
    signatures_a.compute_from_png_stream(file_a, &header_a)?;
    signatures_b.compute_from_png_stream(file_b, &header_b)?;

    let (dy, confidence_y) = solve_1d_shift(
        signatures_a.row_sums(),
        signatures_b.row_sums(),
        height,
        MAX_SEARCH_RADIUS,
    )?;
    let (dx, confidence_x) = solve_1d_shift(
        signatures_a.col_sums(),
        signatures_b.col_sums(),
        width,
        MAX_SEARCH_RADIUS,
    )?;

    let combined_confidence = (confidence_y + confidence_x) / 2; // each <= 1e6: no overflow
    if combined_confidence < REGISTRATION_MIN_CONFIDENCE_PPM {
        #[cfg(debug_assertions)]
        eprintln!(
            "RPI-402: confidence {} ppm below {} ppm (dx={}, dy={})",
            combined_confidence, REGISTRATION_MIN_CONFIDENCE_PPM, dx, dy
        );
        return Err(SlimerError::RegInsufficientContrast);
    }
    RegistrationShift::new(dx, dy, combined_confidence)
}

// ============================================================================
// ALIGNED SCANLINE PAIR READER (SINGLE dy-SYNCHRONISATION POLICY)
// ============================================================================

/// Per-row report from [`AlignedScanlinePairReader`].
#[derive(Copy, Clone, Debug)]
struct AlignedRowInfo {
    /// Reference row index (0..height).
    y: usize,
    /// True if the target row `y + dy` exists and was loaded into `line_b`.
    target_row_available: bool,
}

/// Streams reference and target rows in lock-step under a registration shift.
///
/// This is the ONE place the dy-synchronisation policy lives. v1 duplicated it in four
/// generators, with inconsistent results.
///
/// Policy:
/// - Every reference row 0..height is delivered, in order.
/// - The target row `y + dy` is supplied when it exists. Rows without a partner are
///   reported with `target_row_available = false`; consumers use the reference row alone.
struct AlignedScanlinePairReader<'a> {
    reader_a: PngScanlineReader<'a>,
    reader_b: PngScanlineReader<'a>,
    width: BoundedWidth,
    height: BoundedHeight,
    shift: RegistrationShift,
    next_y: usize,
}

impl<'a> AlignedScanlinePairReader<'a> {
    fn open(
        file_a: &'a mut File,
        file_b: &'a mut File,
        shift: &RegistrationShift,
    ) -> Result<Self, SlimerError> {
        let header_a = parse_png_header(file_a)?;
        let header_b = parse_png_header(file_b)?;
        require_matching_dimensions(&header_a, &header_b)?;
        let dy = shift.get_dy()?;

        let reader_a = PngScanlineReader::new(file_a, &header_a)?;
        let mut reader_b = PngScanlineReader::new(file_b, &header_b)?;

        // dy > 0: target rows 0..dy have no reference partner; skip them.
        if dy > 0 {
            let mut discard_row = [0u8; MAX_IMAGE_WIDTH];
            for _ in 0..dy.unsigned_abs() {
                if !reader_b.next_luma_row(&mut discard_row)? {
                    break; // dy >= height: no row will ever pair; handled per row below
                }
            }
        }
        Ok(Self {
            reader_a,
            reader_b,
            width: header_a.width(),
            height: header_a.height(),
            shift: *shift,
            next_y: 0,
        })
    }

    /// Loads the next reference row (and its target partner, if any).
    ///
    /// Returns `Ok(None)` after the last row.
    fn next_aligned_row(
        &mut self,
        line_a: &mut [u8; MAX_IMAGE_WIDTH],
        line_b: &mut [u8; MAX_IMAGE_WIDTH],
    ) -> Result<Option<AlignedRowInfo>, SlimerError> {
        let height = self.height.get()?;
        if self.next_y >= height {
            return Ok(None);
        }
        let y = self.next_y;
        if !self.reader_a.next_luma_row(line_a)? {
            return Err(SlimerError::PngImageDataTruncated);
        }
        let dy = self.shift.get_dy()?; // re-checked every row
        let target_y = y as i64 + i64::from(dy);
        let target_row_available = target_y >= 0 && (target_y as u64) < height as u64;
        if target_row_available && !self.reader_b.next_luma_row(line_b)? {
            return Err(SlimerError::PngImageDataTruncated);
        }
        self.next_y = y + 1;
        Ok(Some(AlignedRowInfo {
            y,
            target_row_available,
        }))
    }
}

/// Returns `line[x + offset]` if that index is inside `0..width`, otherwise `None`.
fn sample_shifted_pixel(
    line: &[u8],
    x: usize,
    offset: i32,
    width: usize,
) -> Result<Option<u8>, SlimerError> {
    let target = i64::try_from(x).map_err(|_| SlimerError::ValArgumentInvalid)? + i64::from(offset);
    if target < 0 {
        return Ok(None);
    }
    let target_index = usize::try_from(target).map_err(|_| SlimerError::ValArgumentInvalid)?;
    if target_index >= width {
        return Ok(None);
    }
    Ok(Some(byte_at(line, target_index)?))
}

// ============================================================================
// STACK-BUFFERED WRITER (ZERO HEAP)
// ============================================================================

/// Buffered writer with a 4 KiB stack buffer and allocation-free number formatting.
///
/// Callers MUST call `finish()`. Nothing is flushed on drop, because a destructor cannot
/// report errors.
pub struct StackBufWriter<'a> {
    file: &'a mut File,
    buffer: [u8; STACK_WRITE_BUFFER_SIZE],
    filled: usize,
}

impl<'a> StackBufWriter<'a> {
    pub fn new(file: &'a mut File) -> Self {
        Self {
            file,
            buffer: [0u8; STACK_WRITE_BUFFER_SIZE],
            filled: 0,
        }
    }

    /// Writes buffered bytes to the file.
    pub fn flush_buffer(&mut self) -> Result<(), SlimerError> {
        if self.filled == 0 {
            return Ok(());
        }
        let pending = self
            .buffer
            .get(..self.filled)
            .ok_or(SlimerError::PipeBufferExhaustion)?;
        if let Err(_e) = self.file.write_all(pending) {
            #[cfg(debug_assertions)]
            eprintln!("SBW-201: buffered write failed: {}", _e);
            return Err(SlimerError::IoWriteFailed);
        }
        self.filled = 0;
        Ok(())
    }

    /// Appends bytes. Inputs larger than the buffer are written through directly.
    pub fn write_bytes(&mut self, bytes: &[u8]) -> Result<(), SlimerError> {
        let needed = self
            .filled
            .checked_add(bytes.len())
            .ok_or(SlimerError::PipeBufferExhaustion)?;
        if needed > STACK_WRITE_BUFFER_SIZE {
            self.flush_buffer()?;
        }
        if bytes.len() > STACK_WRITE_BUFFER_SIZE {
            if let Err(_e) = self.file.write_all(bytes) {
                #[cfg(debug_assertions)]
                eprintln!("SBW-201: direct write failed: {}", _e);
                return Err(SlimerError::IoWriteFailed);
            }
            return Ok(());
        }
        let end = self
            .filled
            .checked_add(bytes.len())
            .ok_or(SlimerError::PipeBufferExhaustion)?;
        let destination = self
            .buffer
            .get_mut(self.filled..end)
            .ok_or(SlimerError::PipeBufferExhaustion)?;
        destination.copy_from_slice(bytes);
        self.filled = end;
        Ok(())
    }

    /// Writes an unsigned decimal, left-padded with zeros to `minimum_digits` (max 20).
    pub fn write_u64_zero_padded(
        &mut self,
        value: u64,
        minimum_digits: usize,
    ) -> Result<(), SlimerError> {
        const MAX_DECIMAL_DIGITS: usize = 20; // u64::MAX has 20 digits
        if minimum_digits > MAX_DECIMAL_DIGITS {
            return Err(SlimerError::ValArgumentInvalid);
        }
        let mut digits = [b'0'; MAX_DECIMAL_DIGITS];
        let mut start = MAX_DECIMAL_DIGITS;
        let mut remaining = value;
        for _digit in 0..MAX_DECIMAL_DIGITS {
            start -= 1; // at most 20 iterations from 20: never underflows
            let slot = digits
                .get_mut(start)
                .ok_or(SlimerError::PipeBufferExhaustion)?;
            *slot = b'0' + (remaining % 10) as u8;
            remaining /= 10;
            if remaining == 0 {
                break;
            }
        }
        let padded_start = start.min(MAX_DECIMAL_DIGITS - minimum_digits);
        let text = digits
            .get(padded_start..)
            .ok_or(SlimerError::PipeBufferExhaustion)?;
        self.write_bytes(text)
    }

    pub fn write_u64(&mut self, value: u64) -> Result<(), SlimerError> {
        self.write_u64_zero_padded(value, 1)
    }

    /// Signed decimal; `i32::MIN` is handled via `unsigned_abs`.
    pub fn write_i32(&mut self, value: i32) -> Result<(), SlimerError> {
        if value < 0 {
            self.write_bytes(b"-")?;
        }
        self.write_u64(u64::from(value.unsigned_abs()))
    }

    /// Flushes the buffer and the OS file handle.
    pub fn finish(&mut self) -> Result<(), SlimerError> {
        self.flush_buffer()?;
        if let Err(_e) = self.file.flush() {
            #[cfg(debug_assertions)]
            eprintln!("SBW-204: file flush failed: {}", _e);
            return Err(SlimerError::IoFlushFailed);
        }
        Ok(())
    }
}

// ============================================================================
// STREAMING GRAYSCALE PNG WRITER (STORED DEFLATE BLOCKS)
// ============================================================================

/// Writes an 8-bit grayscale PNG row by row.
///
/// Encoding:
/// - one stored (uncompressed) Deflate block per row;
/// - a single IDAT chunk whose length is computed up front.
///
/// Policy:
/// - Exactly `height` rows must be written. Extra or missing rows are errors, because
///   either would produce a corrupt file (a v1 bug).
/// - Output is uncompressed, so file size is roughly width * height bytes. This trades
///   size for zero heap and simplicity.
pub struct PngGrayStreamWriter<'a> {
    output: StackBufWriter<'a>,
    width: BoundedWidth,
    height: BoundedHeight,
    rows_written: usize,
    idat_crc: Crc32,
    image_adler: Adler32,
}

impl<'a> PngGrayStreamWriter<'a> {
    /// Writes the signature, IHDR, and IDAT header + zlib header.
    pub fn begin(
        file: &'a mut File,
        width: BoundedWidth,
        height: BoundedHeight,
    ) -> Result<Self, SlimerError> {
        let width_pixels = width.get()?;
        let height_pixels = height.get()?;
        let width_bytes = u32::try_from(width_pixels)
            .map_err(|_| SlimerError::RegArithmeticOverflow)?
            .to_be_bytes();
        let height_bytes = u32::try_from(height_pixels)
            .map_err(|_| SlimerError::RegArithmeticOverflow)?
            .to_be_bytes();
        // Bit depth 8, colour type 0 (gray), compression 0, filter 0, interlace 0.
        let ihdr_data: [u8; 13] = [
            width_bytes[0],
            width_bytes[1],
            width_bytes[2],
            width_bytes[3],
            height_bytes[0],
            height_bytes[1],
            height_bytes[2],
            height_bytes[3],
            8,
            0,
            0,
            0,
            0,
        ];
        let mut ihdr_crc = Crc32::new();
        ihdr_crc.update(b"IHDR");
        ihdr_crc.update(&ihdr_data);

        let mut output = StackBufWriter::new(file);
        output.write_bytes(&PNG_SIGNATURE)?;
        output.write_bytes(&13u32.to_be_bytes())?;
        output.write_bytes(b"IHDR")?;
        output.write_bytes(&ihdr_data)?;
        output.write_bytes(&ihdr_crc.finish().to_be_bytes())?;

        let bytes_per_row = width_pixels
            .checked_add(PNG_STORED_ROW_OVERHEAD)
            .ok_or(SlimerError::RegArithmeticOverflow)?;
        let idat_length = bytes_per_row
            .checked_mul(height_pixels)
            .and_then(|total| total.checked_add(ZLIB_HEADER_AND_TRAILER_BYTES))
            .ok_or(SlimerError::RegArithmeticOverflow)?;
        let idat_length_u32 =
            u32::try_from(idat_length).map_err(|_| SlimerError::RegArithmeticOverflow)?;
        output.write_bytes(&idat_length_u32.to_be_bytes())?;
        output.write_bytes(b"IDAT")?;

        let mut writer = Self {
            output,
            width,
            height,
            rows_written: 0,
            idat_crc: Crc32::new(),
            image_adler: Adler32::new(),
        };
        writer.idat_crc.update(b"IDAT");
        writer.emit_idat_payload(&ZLIB_STORED_HEADER, false)?;
        Ok(writer)
    }

    /// Writes IDAT payload bytes, updating the chunk CRC and (optionally) the Adler-32.
    fn emit_idat_payload(
        &mut self,
        bytes: &[u8],
        include_in_adler: bool,
    ) -> Result<(), SlimerError> {
        self.idat_crc.update(bytes);
        if include_in_adler {
            self.image_adler.update(bytes);
        }
        self.output.write_bytes(bytes)
    }

    /// Writes one row of exactly `width` luma bytes (`row` may be longer; extra is ignored).
    pub fn write_row(&mut self, row: &[u8]) -> Result<(), SlimerError> {
        let width_pixels = self.width.get()?;
        let height_pixels = self.height.get()?;
        if self.rows_written >= height_pixels {
            #[cfg(debug_assertions)]
            eprintln!(
                "PGW-504: row {} exceeds height {}",
                self.rows_written, height_pixels
            );
            return Err(SlimerError::PipeRowCountExceeded);
        }
        let pixels = row
            .get(..width_pixels)
            .ok_or(SlimerError::ValArgumentInvalid)?;
        let is_final_row = self.rows_written + 1 == height_pixels;
        let block_length =
            u16::try_from(width_pixels + 1).map_err(|_| SlimerError::RegArithmeticOverflow)?;
        let length_bytes = block_length.to_le_bytes();
        let complement_bytes = (!block_length).to_le_bytes();
        let block_header = [
            u8::from(is_final_row), // BFINAL bit; BTYPE = 00 (stored)
            length_bytes[0],
            length_bytes[1],
            complement_bytes[0],
            complement_bytes[1],
        ];
        self.emit_idat_payload(&block_header, false)?;
        self.emit_idat_payload(&[0u8], true)?; // PNG filter type 0 (None)
        self.emit_idat_payload(pixels, true)?;
        self.rows_written += 1;
        Ok(())
    }

    /// Writes the Adler-32, IDAT CRC, and IEND, then flushes.
    ///
    /// Errors if the row count is incomplete.
    pub fn finish(mut self) -> Result<(), SlimerError> {
        let height_pixels = self.height.get()?;
        if self.rows_written != height_pixels {
            #[cfg(debug_assertions)]
            eprintln!(
                "PGW-505: {} of {} rows written",
                self.rows_written, height_pixels
            );
            return Err(SlimerError::PipeRowCountIncomplete);
        }
        let adler_bytes = self.image_adler.finish().to_be_bytes();
        self.emit_idat_payload(&adler_bytes, false)?;
        let crc_bytes = self.idat_crc.finish().to_be_bytes();
        self.output.write_bytes(&crc_bytes)?;
        self.output.write_bytes(&PNG_IEND_CHUNK)?;
        self.output.finish()
    }
}

// ============================================================================
// 2D FUSION (MODE `2d`)
// ============================================================================

/// Fusion rule: average when the pixels agree, otherwise keep the reference pixel.
///
/// This avoids ghosting where the scene changed or registration is locally wrong.
fn fuse_registered_pixels(reference: u8, target: u8) -> Result<u8, SlimerError> {
    if reference.abs_diff(target) < FUSION_AGREEMENT_THRESHOLD {
        let average = (u16::from(reference) + u16::from(target)) / 2;
        u8::try_from(average).map_err(|_| SlimerError::RegArithmeticOverflow)
    } else {
        Ok(reference)
    }
}

/// Mode `2d`: streams a fused 8-bit grayscale PNG with the reference image's geometry.
pub fn enhance_png_streamed(
    file_a: &mut File,
    file_b: &mut File,
    shift: &RegistrationShift,
    out_file: &mut File,
) -> Result<(), SlimerError> {
    let mut pair = AlignedScanlinePairReader::open(file_a, file_b, shift)?;
    let width = pair.width.get()?;
    let global_dx = shift.get_dx()?;
    let mut writer = PngGrayStreamWriter::begin(out_file, pair.width, pair.height)?;

    let mut line_a = [0u8; MAX_IMAGE_WIDTH];
    let mut line_b = [0u8; MAX_IMAGE_WIDTH];
    let mut fused_row = [0u8; MAX_IMAGE_WIDTH];
    let mut rows_processed: usize = 0;

    while let Some(row) = pair.next_aligned_row(&mut line_a, &mut line_b)? {
        rows_processed += 1;
        if rows_processed > MAX_IMAGE_HEIGHT {
            return Err(SlimerError::PipeRowCountExceeded);
        }
        for x in 0..width {
            let reference_pixel = byte_at(&line_a, x)?;
            let output_pixel = if row.target_row_available {
                match sample_shifted_pixel(&line_b, x, global_dx, width)? {
                    Some(target_pixel) => fuse_registered_pixels(reference_pixel, target_pixel)?,
                    None => reference_pixel,
                }
            } else {
                reference_pixel
            };
            let slot = fused_row
                .get_mut(x)
                .ok_or(SlimerError::ValIndexOutOfBounds)?;
            *slot = output_pixel;
        }
        writer.write_row(
            fused_row
                .get(..width)
                .ok_or(SlimerError::ValIndexOutOfBounds)?,
        )?;
    }
    writer.finish()
}

// ============================================================================
// PSEUDO-DEPTH (SHARED BY ALL 3D OUTPUTS)
// ============================================================================

/// Local stereo disparity at `x`.
///
/// Method: 1D SAD over a (2*h+1)-pixel window, searched over
/// +/- `MAX_LOCAL_DISPARITY_SEARCH` around the global `dx`.
///
/// Ties prefer 0, so textureless regions read as flat.
fn estimate_local_disparity(
    line_a: &[u8],
    line_b: &[u8],
    x: usize,
    global_dx: i32,
    width: usize,
) -> Result<i32, SlimerError> {
    let x_signed = i64::try_from(x).map_err(|_| SlimerError::ValArgumentInvalid)?;
    let width_signed = i64::try_from(width).map_err(|_| SlimerError::ValArgumentInvalid)?;
    let mut best_disparity: i32 = 0;
    let mut best_cost: u64 = u64::MAX;

    for disparity in -MAX_LOCAL_DISPARITY_SEARCH..=MAX_LOCAL_DISPARITY_SEARCH {
        let mut cost: u64 = 0;
        let mut samples: u64 = 0;
        for offset in -DISPARITY_WINDOW_HALF_WIDTH..=DISPARITY_WINDOW_HALF_WIDTH {
            let ax = x_signed + i64::from(offset);
            let bx = ax + i64::from(global_dx) + i64::from(disparity);
            if ax < 0 || bx < 0 || ax >= width_signed || bx >= width_signed {
                continue;
            }
            let value_a = byte_at(line_a, ax as usize)?;
            let value_b = byte_at(line_b, bx as usize)?;
            cost += u64::from(value_a.abs_diff(value_b));
            samples += 1;
        }
        if samples == 0 {
            continue;
        }
        let normalized = cost * 256 / samples; // scaled to keep integer precision
        if normalized < best_cost
            || (normalized == best_cost && disparity.abs() < best_disparity.abs())
        {
            best_cost = normalized;
            best_disparity = disparity;
        }
    }
    Ok(best_disparity)
}

/// Pseudo-depth = luma / DEPTH_LUMA_DIVISOR + disparity * DEPTH_DISPARITY_GAIN.
///
/// Visualisation heuristic, not metric depth. Range is approximately -18..=81.
fn compute_pseudo_depth(luma: u8, disparity: i32) -> Result<i32, SlimerError> {
    let luma_part = i32::from(luma) / DEPTH_LUMA_DIVISOR;
    let disparity_part = disparity
        .checked_mul(DEPTH_DISPARITY_GAIN)
        .ok_or(SlimerError::RegArithmeticOverflow)?;
    luma_part
        .checked_add(disparity_part)
        .ok_or(SlimerError::RegArithmeticOverflow)
}

/// Pseudo-depth for the grid vertex at `x`. Disparity is 0 when no target row exists.
fn sample_pseudo_depth(
    line_a: &[u8],
    line_b: &[u8],
    x: usize,
    global_dx: i32,
    width: usize,
    target_row_available: bool,
) -> Result<i32, SlimerError> {
    let luma = byte_at(line_a, x)?;
    let disparity = if target_row_available {
        estimate_local_disparity(line_a, line_b, x, global_dx, width)?
    } else {
        0
    };
    compute_pseudo_depth(luma, disparity)
}

/// Grid vertices along an extent: one every `WIREFRAME_GRID_STEP`, starting at 0.
fn count_grid_samples(extent: usize) -> Result<u32, SlimerError> {
    let samples = extent
        .checked_add(WIREFRAME_GRID_STEP - 1)
        .ok_or(SlimerError::MeshIndexOverflow)?
        / WIREFRAME_GRID_STEP;
    u32::try_from(samples).map_err(|_| SlimerError::MeshIndexOverflow)
}

/// 1-based OBJ vertex index of grid cell (row, column), in row-major order.
fn grid_vertex_index(row: u32, column: u32, columns: u32) -> Result<u32, SlimerError> {
    row.checked_mul(columns)
        .and_then(|base| base.checked_add(column))
        .and_then(|index| index.checked_add(1))
        .ok_or(SlimerError::MeshIndexOverflow)
}

/// Converts image row `y` to an "up is +Y" coordinate: (height - 1 - y).
fn up_axis_coordinate(height: usize, y: usize) -> Result<usize, SlimerError> {
    height
        .checked_sub(1)
        .and_then(|top| top.checked_sub(y))
        .ok_or(SlimerError::ValIndexOutOfBounds)
}

fn write_obj_vertex(
    writer: &mut StackBufWriter,
    x: usize,
    y_up: usize,
    z: i32,
) -> Result<(), SlimerError> {
    writer.write_bytes(b"v ")?;
    writer.write_u64(u64::try_from(x).map_err(|_| SlimerError::ValArgumentInvalid)?)?;
    writer.write_bytes(b" ")?;
    writer.write_u64(u64::try_from(y_up).map_err(|_| SlimerError::ValArgumentInvalid)?)?;
    writer.write_bytes(b" ")?;
    writer.write_i32(z)?;
    writer.write_bytes(b"\n")
}

fn write_obj_line_element(
    writer: &mut StackBufWriter,
    from: u32,
    to: u32,
) -> Result<(), SlimerError> {
    writer.write_bytes(b"l ")?;
    writer.write_u64(u64::from(from))?;
    writer.write_bytes(b" ")?;
    writer.write_u64(u64::from(to))?;
    writer.write_bytes(b"\n")
}

/// Writes `f a/a b/b c/c`. Vertex and texture indices coincide by construction.
fn write_obj_textured_triangle(
    writer: &mut StackBufWriter,
    a: u32,
    b: u32,
    c: u32,
) -> Result<(), SlimerError> {
    writer.write_bytes(b"f ")?;
    for (position, index) in [a, b, c].into_iter().enumerate() {
        if position > 0 {
            writer.write_bytes(b" ")?;
        }
        writer.write_u64(u64::from(index))?;
        writer.write_bytes(b"/")?;
        writer.write_u64(u64::from(index))?;
    }
    writer.write_bytes(b"\n")
}

// ============================================================================
// MODE `3d`: WAVEFRONT OBJ WIREFRAME (v & l)
// ============================================================================

/// Streams a Wavefront OBJ file of vertices (`v`) and line edges (`l`) forming a
/// topological grid over the registered pair.
///
/// - Phase 1 streams vertices.
/// - Phase 2 streams edges (index arithmetic only; no stored vertex data).
/// - One vertex every `WIREFRAME_GRID_STEP` pixels in x and y.
/// - Z is pseudo-depth.
pub fn generate_3d_wireframe_obj(
    file_a: &mut File,
    file_b: &mut File,
    shift: &RegistrationShift,
    out_obj: &mut File,
) -> Result<(), SlimerError> {
    let mut pair = AlignedScanlinePairReader::open(file_a, file_b, shift)?;
    let width = pair.width.get()?;
    let height = pair.height.get()?;
    let global_dx = shift.get_dx()?;
    let grid_columns = count_grid_samples(width)?;

    let mut line_a = [0u8; MAX_IMAGE_WIDTH];
    let mut line_b = [0u8; MAX_IMAGE_WIDTH];
    let mut writer = StackBufWriter::new(out_obj);
    writer.write_bytes(b"# Slimer pure-Rust zero-heap 3D wireframe OBJ\n")?;
    writer.write_bytes(b"# Z is pseudo-depth (luma/4 + local disparity*3), not metric depth\n")?;
    writer.write_bytes(b"o terrain_wireframe\n")?;

    // --- PHASE 1: STREAM VERTICES (v X Y Z) ---
    let mut grid_rows: u32 = 0;
    let mut rows_processed: usize = 0;
    while let Some(row) = pair.next_aligned_row(&mut line_a, &mut line_b)? {
        rows_processed += 1;
        if rows_processed > MAX_IMAGE_HEIGHT {
            return Err(SlimerError::PipeRowCountExceeded);
        }
        if row.y % WIREFRAME_GRID_STEP != 0 {
            continue;
        }
        let y_up = up_axis_coordinate(height, row.y)?;
        for x in (0..width).step_by(WIREFRAME_GRID_STEP) {
            let depth = sample_pseudo_depth(
                &line_a,
                &line_b,
                x,
                global_dx,
                width,
                row.target_row_available,
            )?;
            write_obj_vertex(&mut writer, x, y_up, depth)?;
        }
        grid_rows = grid_rows
            .checked_add(1)
            .ok_or(SlimerError::MeshIndexOverflow)?;
    }

    // --- PHASE 2: STREAM EDGES (l V1 V2) ---
    for grid_row in 0..grid_rows {
        for grid_column in 0..grid_columns {
            let current = grid_vertex_index(grid_row, grid_column, grid_columns)?;
            if grid_column + 1 < grid_columns {
                let right = grid_vertex_index(grid_row, grid_column + 1, grid_columns)?;
                write_obj_line_element(&mut writer, current, right)?;
            }
            if grid_row + 1 < grid_rows {
                let below = grid_vertex_index(grid_row + 1, grid_column, grid_columns)?;
                write_obj_line_element(&mut writer, current, below)?;
            }
        }
    }
    writer.finish()
}

// ============================================================================
// MODE `svg`: ISOMETRIC (2:1 DIMETRIC) SVG WIREFRAME
// ============================================================================

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
struct ProjectedPoint2D {
    u: i32,
    v: i32,
}

fn write_svg_edge(
    writer: &mut StackBufWriter,
    from: ProjectedPoint2D,
    to: ProjectedPoint2D,
) -> Result<(), SlimerError> {
    writer.write_bytes(b"<line x1=\"")?;
    writer.write_i32(from.u)?;
    writer.write_bytes(b"\" y1=\"")?;
    writer.write_i32(from.v)?;
    writer.write_bytes(b"\" x2=\"")?;
    writer.write_i32(to.u)?;
    writer.write_bytes(b"\" y2=\"")?;
    writer.write_i32(to.v)?;
    writer.write_bytes(b"\"/>\n")
}

/// Streams the wireframe as an SVG vector graphic in a single pass.
///
/// - Projection: integer 2:1 dimetric, u = x - y, v = (x + y) / 2 - z.
/// - No floats and no heap. Only the previous grid row of projected points is kept.
pub fn generate_3d_wireframe_svg(
    file_a: &mut File,
    file_b: &mut File,
    shift: &RegistrationShift,
    out_svg: &mut File,
) -> Result<(), SlimerError> {
    let mut pair = AlignedScanlinePairReader::open(file_a, file_b, shift)?;
    let width = pair.width.get()?;
    let height = pair.height.get()?;
    let global_dx = shift.get_dx()?;
    let width_i32 = i32::try_from(width).map_err(|_| SlimerError::RegArithmeticOverflow)?;
    let height_i32 = i32::try_from(height).map_err(|_| SlimerError::RegArithmeticOverflow)?;

    // viewBox derived from the projection and pseudo-depth range. All operands are
    // bounded (<= 1024, depth -18..=81), so plain i32 arithmetic cannot overflow.
    let depth_max = 255 / DEPTH_LUMA_DIVISOR + MAX_LOCAL_DISPARITY_SEARCH * DEPTH_DISPARITY_GAIN;
    let depth_min = -(MAX_LOCAL_DISPARITY_SEARCH * DEPTH_DISPARITY_GAIN);
    let min_u = -height_i32 - SVG_VIEWBOX_MARGIN;
    let min_v = -depth_max - SVG_VIEWBOX_MARGIN;
    let view_width = width_i32 + height_i32 + 2 * SVG_VIEWBOX_MARGIN;
    let view_height =
        (width_i32 + height_i32) / 2 + (depth_max - depth_min) + 2 * SVG_VIEWBOX_MARGIN;

    let mut writer = StackBufWriter::new(out_svg);
    writer.write_bytes(b"<svg xmlns=\"http://www.w3.org/2000/svg\" viewBox=\"")?;
    writer.write_i32(min_u)?;
    writer.write_bytes(b" ")?;
    writer.write_i32(min_v)?;
    writer.write_bytes(b" ")?;
    writer.write_i32(view_width)?;
    writer.write_bytes(b" ")?;
    writer.write_i32(view_height)?;
    writer.write_bytes(b"\" style=\"background-color:#111418;\">\n")?;
    writer.write_bytes(b"<!-- Slimer pseudo-depth wireframe; Z is not metric depth -->\n")?;
    writer.write_bytes(b"<g stroke=\"#00e5ff\" stroke-width=\"1\" stroke-opacity=\"0.85\">\n")?;

    let mut line_a = [0u8; MAX_IMAGE_WIDTH];
    let mut line_b = [0u8; MAX_IMAGE_WIDTH];
    let mut previous_row_points = [ProjectedPoint2D { u: 0, v: 0 }; MAX_GRID_POINTS_PER_ROW];
    let mut current_row_points = [ProjectedPoint2D { u: 0, v: 0 }; MAX_GRID_POINTS_PER_ROW];
    let mut previous_row_count: usize = 0;
    let mut has_previous_row = false;
    let mut rows_processed: usize = 0;

    while let Some(row) = pair.next_aligned_row(&mut line_a, &mut line_b)? {
        rows_processed += 1;
        if rows_processed > MAX_IMAGE_HEIGHT {
            return Err(SlimerError::PipeRowCountExceeded);
        }
        if row.y % WIREFRAME_GRID_STEP != 0 {
            continue;
        }
        let y_i32 = i32::try_from(row.y).map_err(|_| SlimerError::RegArithmeticOverflow)?;
        let mut column_index: usize = 0;
        for x in (0..width).step_by(WIREFRAME_GRID_STEP) {
            if column_index >= MAX_GRID_POINTS_PER_ROW {
                return Err(SlimerError::PipeBufferExhaustion);
            }
            let depth = sample_pseudo_depth(
                &line_a,
                &line_b,
                x,
                global_dx,
                width,
                row.target_row_available,
            )?;
            let x_i32 = i32::try_from(x).map_err(|_| SlimerError::RegArithmeticOverflow)?;
            let point = ProjectedPoint2D {
                u: x_i32 - y_i32,
                v: (x_i32 + y_i32) / 2 - depth,
            };
            let slot = current_row_points
                .get_mut(column_index)
                .ok_or(SlimerError::ValIndexOutOfBounds)?;
            *slot = point;

            // 1. Horizontal edge to the left neighbour.
            if column_index > 0 {
                let left = *current_row_points
                    .get(column_index - 1)
                    .ok_or(SlimerError::ValIndexOutOfBounds)?;
                write_svg_edge(&mut writer, left, point)?;
            }
            // 2. Vertical edge to the neighbour in the previous grid row.
            if has_previous_row && column_index < previous_row_count {
                let above = *previous_row_points
                    .get(column_index)
                    .ok_or(SlimerError::ValIndexOutOfBounds)?;
                write_svg_edge(&mut writer, above, point)?;
            }
            column_index += 1;
        }
        previous_row_points = current_row_points; // fixed-size array copy (stack)
        previous_row_count = column_index;
        has_previous_row = true;
    }

    writer.write_bytes(b"</g>\n</svg>\n")?;
    writer.finish()
}

// ============================================================================
// MODE `mesh`: TEXTURED OBJ + MTL
// ============================================================================

/// Writes a texture coordinate at the pixel CENTRE, (2i + 1) / (2n), with 4 decimals.
///
/// No floats. Requires `index < count`.
fn write_normalized_uv(
    writer: &mut StackBufWriter,
    index: usize,
    count: usize,
) -> Result<(), SlimerError> {
    if count == 0 || index >= count {
        return Err(SlimerError::ValArgumentInvalid);
    }
    let numerator = u64::try_from(index)
        .ok()
        .and_then(|i| i.checked_mul(2))
        .and_then(|i| i.checked_add(1))
        .and_then(|i| i.checked_mul(10_000))
        .ok_or(SlimerError::RegArithmeticOverflow)?;
    let denominator = u64::try_from(count)
        .ok()
        .and_then(|n| n.checked_mul(2))
        .ok_or(SlimerError::RegArithmeticOverflow)?;
    let ten_thousandths = numerator / denominator; // always < 10_000 because index < count
    writer.write_bytes(b"0.")?;
    writer.write_u64_zero_padded(ten_thousandths, 4)
}

/// Returns the final path component (after the last '/' or '\\').
///
/// Errors if that component is empty.
pub fn extract_filename_bytes(path: &[u8]) -> Result<&[u8], SlimerError> {
    let start = match path.iter().rposition(|&byte| byte == b'/' || byte == b'\\') {
        Some(separator) => separator + 1,
        None => 0,
    };
    let name = path.get(start..).ok_or(SlimerError::MeshFilenameInvalid)?;
    if name.is_empty() {
        return Err(SlimerError::MeshFilenameInvalid);
    }
    Ok(name)
}

/// Validates a file name embedded in OBJ/MTL text.
///
/// Rejected: control characters (a newline would inject directives), empty names, and
/// names longer than `MAX_MTL_REFERENCE_NAME_BYTES`.
fn validate_mtl_reference_name(name: &[u8]) -> Result<(), SlimerError> {
    if name.is_empty() || name.len() > MAX_MTL_REFERENCE_NAME_BYTES {
        return Err(SlimerError::MeshFilenameInvalid);
    }
    if name.iter().any(|&byte| byte < 0x20 || byte == 0x7F) {
        #[cfg(debug_assertions)]
        eprintln!("VMN-700: control character in OBJ/MTL file name");
        return Err(SlimerError::MeshFilenameInvalid);
    }
    Ok(())
}

/// Writes the companion .mtl that maps the reference photo onto the mesh.
///
/// The texture is referenced by FILE NAME ONLY. This keeps the user's directory
/// structure out of the output. MeshLab therefore expects the image next to the OBJ.
pub fn write_mtl_material_file(
    out_mtl: &mut File,
    texture_image_filename: &[u8],
) -> Result<(), SlimerError> {
    let texture_name = extract_filename_bytes(texture_image_filename)?;
    validate_mtl_reference_name(texture_name)?;
    let mut writer = StackBufWriter::new(out_mtl);
    writer.write_bytes(b"# Material definition for MeshLab / Blender (Slimer)\n")?;
    writer.write_bytes(b"newmtl image_texture\n")?;
    writer.write_bytes(b"Ka 1.000 1.000 1.000\n")?;
    writer.write_bytes(b"Kd 1.000 1.000 1.000\n")?;
    writer.write_bytes(b"Ks 0.000 0.000 0.000\n")?;
    writer.write_bytes(b"d 1.0\n")?;
    writer.write_bytes(b"illum 1\n")?;
    writer.write_bytes(b"map_Kd ")?;
    writer.write_bytes(texture_name)?;
    writer.write_bytes(b"\n")?;
    writer.finish()
}

/// Streams a textured 3D surface mesh (OBJ).
///
/// Contents:
/// - vertices `v` and pixel-centre UVs `vt`;
/// - two triangles `f` per grid cell, linked to `mtl_filename` (file name only).
pub fn generate_3d_textured_mesh_obj(
    file_a: &mut File,
    file_b: &mut File,
    shift: &RegistrationShift,
    out_obj: &mut File,
    mtl_filename: &[u8],
) -> Result<(), SlimerError> {
    let material_name = extract_filename_bytes(mtl_filename)?;
    validate_mtl_reference_name(material_name)?;

    let mut pair = AlignedScanlinePairReader::open(file_a, file_b, shift)?;
    let width = pair.width.get()?;
    let height = pair.height.get()?;
    let global_dx = shift.get_dx()?;
    let grid_columns = count_grid_samples(width)?;

    let mut line_a = [0u8; MAX_IMAGE_WIDTH];
    let mut line_b = [0u8; MAX_IMAGE_WIDTH];
    let mut writer = StackBufWriter::new(out_obj);
    writer.write_bytes(b"# Slimer textured 3D mesh (Z is pseudo-depth, not metric)\nmtllib ")?;
    writer.write_bytes(material_name)?;
    writer.write_bytes(b"\no terrain_mesh\nusemtl image_texture\n")?;

    // --- PHASE 1: STREAM VERTICES (v) AND TEXTURE COORDINATES (vt) ---
    let mut grid_rows: u32 = 0;
    let mut rows_processed: usize = 0;
    while let Some(row) = pair.next_aligned_row(&mut line_a, &mut line_b)? {
        rows_processed += 1;
        if rows_processed > MAX_IMAGE_HEIGHT {
            return Err(SlimerError::PipeRowCountExceeded);
        }
        if row.y % WIREFRAME_GRID_STEP != 0 {
            continue;
        }
        let y_up = up_axis_coordinate(height, row.y)?;
        for x in (0..width).step_by(WIREFRAME_GRID_STEP) {
            let depth = sample_pseudo_depth(
                &line_a,
                &line_b,
                x,
                global_dx,
                width,
                row.target_row_available,
            )?;
            write_obj_vertex(&mut writer, x, y_up, depth)?;
            writer.write_bytes(b"vt ")?;
            write_normalized_uv(&mut writer, x, width)?;
            writer.write_bytes(b" ")?;
            write_normalized_uv(&mut writer, y_up, height)?; // V = 1 at the image top
            writer.write_bytes(b"\n")?;
        }
        grid_rows = grid_rows
            .checked_add(1)
            .ok_or(SlimerError::MeshIndexOverflow)?;
    }

    // --- PHASE 2: STREAM TRIANGULAR FACES (f v/vt ...) ---
    for grid_row in 0..grid_rows.saturating_sub(1) {
        for grid_column in 0..grid_columns.saturating_sub(1) {
            let top_left = grid_vertex_index(grid_row, grid_column, grid_columns)?;
            let top_right = grid_vertex_index(grid_row, grid_column + 1, grid_columns)?;
            let bottom_left = grid_vertex_index(grid_row + 1, grid_column, grid_columns)?;
            let bottom_right = grid_vertex_index(grid_row + 1, grid_column + 1, grid_columns)?;
            write_obj_textured_triangle(&mut writer, top_left, bottom_left, top_right)?;
            write_obj_textured_triangle(&mut writer, top_right, bottom_left, bottom_right)?;
        }
    }
    writer.finish()
}

// ============================================================================
// ERROR LOG (CALLED BY THE RUNNER, NEVER BY FAILING FUNCTIONS)
// ============================================================================

/// True if a stage tag is short lowercase ASCII / digits / '_' (safe to log).
fn is_safe_log_tag(tag: &[u8]) -> bool {
    !tag.is_empty()
        && tag.len() <= 32
        && tag
            .iter()
            .all(|&byte| byte.is_ascii_lowercase() || byte.is_ascii_digit() || byte == b'_')
}

/// Appends one terse, PII-free line to an already-open log file.
///
/// Format: `slimer_error unix_time=<secs|unknown> code=<u16> stage=<tag>`.
/// The line is assembled in the stack buffer and written in one call.
pub fn append_error_log_entry(
    log_file: &mut File,
    error: SlimerError,
    stage_tag: &[u8],
) -> Result<(), SlimerError> {
    let unix_seconds = match SystemTime::now().duration_since(UNIX_EPOCH) {
        Ok(elapsed) => Some(elapsed.as_secs()),
        Err(_) => None, // clock before 1970: log "unknown" rather than fail
    };
    let mut writer = StackBufWriter::new(log_file);
    writer.write_bytes(b"slimer_error unix_time=")?;
    match unix_seconds {
        Some(seconds) => writer.write_u64(seconds)?,
        None => writer.write_bytes(b"unknown")?,
    }
    writer.write_bytes(b" code=")?;
    writer.write_u64(u64::from(error.code()))?;
    writer.write_bytes(b" stage=")?;
    if is_safe_log_tag(stage_tag) {
        writer.write_bytes(stage_tag)?;
    } else {
        writer.write_bytes(b"invalid_tag")?;
    }
    writer.write_bytes(b"\n")?;
    writer.finish()
}

// ============================================================================
// CARGO TESTS (GATED WITH #[cfg(test)])
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::{Path, PathBuf};
    use std::sync::atomic::{AtomicUsize, Ordering};

    // ---------------- test fixtures (heap allowed in tests) ----------------

    static TEST_FILE_COUNTER: AtomicUsize = AtomicUsize::new(0);

    fn unique_test_path(tag: &str) -> PathBuf {
        let serial = TEST_FILE_COUNTER.fetch_add(1, Ordering::SeqCst);
        std::env::temp_dir().join(format!(
            "slimer_test_{}_{}_{}",
            std::process::id(),
            serial,
            tag
        ))
    }

    fn remove_test_file(path: &Path) {
        if let Err(e) = std::fs::remove_file(path) {
            eprintln!("test cleanup: could not remove temp file: {}", e);
        }
    }

    fn write_test_file(tag: &str, bytes: &[u8]) -> PathBuf {
        let path = unique_test_path(tag);
        std::fs::write(&path, bytes).unwrap();
        path
    }

    fn write_gray_png_fixture(
        tag: &str,
        width: usize,
        height: usize,
        pixel: impl Fn(usize, usize) -> u8,
    ) -> PathBuf {
        let path = unique_test_path(tag);
        let mut file = File::create(&path).unwrap();
        let mut writer = PngGrayStreamWriter::begin(
            &mut file,
            BoundedWidth::new(width).unwrap(),
            BoundedHeight::new(height).unwrap(),
        )
        .unwrap();
        let mut row = [0u8; MAX_IMAGE_WIDTH];
        for y in 0..height {
            for x in 0..width {
                row[x] = pixel(x, y);
            }
            writer.write_row(&row[..width]).unwrap();
        }
        writer.finish().unwrap();
        path
    }

    fn read_luma_rows(path: &Path) -> Vec<Vec<u8>> {
        let mut file = File::open(path).unwrap();
        let header = parse_png_header(&mut file).unwrap();
        let width = header.width().get().unwrap();
        let mut reader = PngScanlineReader::new(&mut file, &header).unwrap();
        let mut row = [0u8; MAX_IMAGE_WIDTH];
        let mut rows = Vec::new();
        while reader.next_luma_row(&mut row).unwrap() {
            rows.push(row[..width].to_vec());
        }
        rows
    }

    fn zlib_wrap_stored(raw: &[u8]) -> Vec<u8> {
        let mut out = vec![0x78, 0x01];
        let pieces: Vec<&[u8]> = if raw.is_empty() {
            vec![&raw[..]]
        } else {
            raw.chunks(65_535).collect()
        };
        let piece_count = pieces.len();
        for (index, piece) in pieces.iter().enumerate() {
            out.push(u8::from(index + 1 == piece_count));
            let length = piece.len() as u16;
            out.extend_from_slice(&length.to_le_bytes());
            out.extend_from_slice(&(!length).to_le_bytes());
            out.extend_from_slice(piece);
        }
        let mut adler = Adler32::new();
        adler.update(raw);
        out.extend_from_slice(&adler.finish().to_be_bytes());
        out
    }

    fn png_chunk(chunk_type: &[u8; 4], data: &[u8]) -> Vec<u8> {
        let mut out = Vec::new();
        out.extend_from_slice(&(data.len() as u32).to_be_bytes());
        out.extend_from_slice(chunk_type);
        out.extend_from_slice(data);
        let mut crc = Crc32::new();
        crc.update(chunk_type);
        crc.update(data);
        out.extend_from_slice(&crc.finish().to_be_bytes());
        out
    }

    fn build_png_bytes(
        width: u32,
        height: u32,
        bit_depth: u8,
        color_type: u8,
        interlace: u8,
        extra_chunks_before_idat: &[Vec<u8>],
        idat_payload: &[u8],
    ) -> Vec<u8> {
        let mut out = PNG_SIGNATURE.to_vec();
        let mut ihdr = Vec::new();
        ihdr.extend_from_slice(&width.to_be_bytes());
        ihdr.extend_from_slice(&height.to_be_bytes());
        ihdr.extend_from_slice(&[bit_depth, color_type, 0, 0, interlace]);
        out.extend(png_chunk(b"IHDR", &ihdr));
        for chunk in extra_chunks_before_idat {
            out.extend_from_slice(chunk);
        }
        out.extend(png_chunk(b"IDAT", idat_payload));
        out.extend(png_chunk(b"IEND", &[]));
        out
    }

    fn valid_gray_8x8_png() -> Vec<u8> {
        let raw: Vec<u8> = (0..8)
            .flat_map(|_| std::iter::once(0u8).chain(std::iter::repeat(100u8).take(8)))
            .collect();
        build_png_bytes(8, 8, 8, 0, 0, &[], &zlib_wrap_stored(&raw))
    }

    /// Minimal fixed-Huffman Deflate encoder for tests.
    ///
    /// Bits are emitted LSB first; Huffman codes are emitted MSB first (RFC 1951).
    struct FixedHuffmanTestEncoder {
        bytes: Vec<u8>,
        accumulator: u64,
        bit_count: u32,
    }

    impl FixedHuffmanTestEncoder {
        fn new() -> Self {
            Self {
                bytes: vec![0x78, 0x01],
                accumulator: 0,
                bit_count: 0,
            }
        }
        fn put_bits(&mut self, value: u32, count: u32) {
            self.accumulator |= u64::from(value) << self.bit_count;
            self.bit_count += count;
            while self.bit_count >= 8 {
                self.bytes.push((self.accumulator & 0xFF) as u8);
                self.accumulator >>= 8;
                self.bit_count -= 8;
            }
        }
        fn put_code_msb_first(&mut self, code: u32, length: u32) {
            let mut reversed = 0u32;
            for bit in 0..length {
                if code & (1 << bit) != 0 {
                    reversed |= 1 << (length - 1 - bit);
                }
            }
            self.put_bits(reversed, length);
        }
        fn put_literal_or_length_symbol(&mut self, symbol: u32) {
            match symbol {
                0..=143 => self.put_code_msb_first(0x30 + symbol, 8),
                144..=255 => self.put_code_msb_first(0x190 + symbol - 144, 9),
                256..=279 => self.put_code_msb_first(symbol - 256, 7),
                _ => self.put_code_msb_first(0xC0 + symbol - 280, 8),
            }
        }
        fn put_distance_symbol(&mut self, symbol: u32) {
            self.put_code_msb_first(symbol, 5);
        }
        fn finish(mut self, raw_for_adler: &[u8]) -> Vec<u8> {
            if self.bit_count > 0 {
                self.bytes.push((self.accumulator & 0xFF) as u8);
            }
            let mut adler = Adler32::new();
            adler.update(raw_for_adler);
            self.bytes.extend_from_slice(&adler.finish().to_be_bytes());
            self.bytes
        }
    }

    /// Encodes "abc" + back-reference(length 3, distance symbol), with BFINAL set.
    fn fixed_huffman_abc_then_match(distance_symbol: u32) -> Vec<u8> {
        let mut encoder = FixedHuffmanTestEncoder::new();
        encoder.put_bits(1, 1); // BFINAL
        encoder.put_bits(1, 2); // BTYPE = 01 (fixed Huffman)
        for &literal in b"abc" {
            encoder.put_literal_or_length_symbol(u32::from(literal));
        }
        encoder.put_literal_or_length_symbol(257); // length 3
        encoder.put_distance_symbol(distance_symbol);
        encoder.put_literal_or_length_symbol(256); // end of block
        encoder.finish(b"abcabc")
    }

    fn square_scene(left: usize, top: usize) -> impl Fn(usize, usize) -> u8 {
        move |x, y| {
            if x >= left && x < left + 10 && y >= top && y < top + 8 {
                200
            } else {
                20
            }
        }
    }

    // ---------------- custom types ----------------

    #[test]
    fn test_bounded_width_and_height_enforce_limits() {
        assert_eq!(
            BoundedWidth::new(7),
            Err(SlimerError::ValIntegrityDimUnderflow)
        );
        assert_eq!(
            BoundedWidth::new(1025),
            Err(SlimerError::ValIntegrityDimOverflow)
        );
        assert_eq!(
            BoundedHeight::new(4),
            Err(SlimerError::ValIntegrityDimUnderflow)
        );
        assert_eq!(
            BoundedHeight::new(2000),
            Err(SlimerError::ValIntegrityDimOverflow)
        );
        assert_eq!(BoundedWidth::new(8).unwrap().get().unwrap(), 8);
        assert_eq!(BoundedHeight::new(1024).unwrap().get().unwrap(), 1024);
    }

    #[test]
    fn test_bounded_types_detect_corrupted_storage() {
        // Simulates a post-construction bit-flip by building the struct directly.
        let corrupted_width = BoundedWidth { pixels: 0 };
        assert_eq!(
            corrupted_width.get(),
            Err(SlimerError::ValCorruptedBitFlipDetected)
        );
        let corrupted_height = BoundedHeight { pixels: 1 << 20 };
        assert_eq!(
            corrupted_height.get(),
            Err(SlimerError::ValCorruptedBitFlipDetected)
        );
        let corrupted_shift = RegistrationShift {
            dx: 99,
            dy: 0,
            confidence_ppm: 0,
        };
        assert_eq!(
            corrupted_shift.get_dx(),
            Err(SlimerError::ValCorruptedBitFlipDetected)
        );
        assert_eq!(
            corrupted_shift.get_dy(),
            Err(SlimerError::ValCorruptedBitFlipDetected)
        );
    }

    #[test]
    fn test_registration_shift_integrity_and_identity_fallback() {
        assert_eq!(
            RegistrationShift::new(33, 0, 1),
            Err(SlimerError::ValIntegrityShiftOutOfBounds)
        );
        assert_eq!(
            RegistrationShift::new(0, -33, 1),
            Err(SlimerError::ValIntegrityShiftOutOfBounds)
        );
        assert_eq!(
            RegistrationShift::new(0, 0, 1_000_001),
            Err(SlimerError::ValIntegrityScoreCorrupt)
        );
        let shift = RegistrationShift::new(-12, 8, 920_000).unwrap();
        assert_eq!(shift.get_dx().unwrap(), -12);
        assert_eq!(shift.get_dy().unwrap(), 8);
        assert_eq!(shift.get_confidence().unwrap(), 920_000);
        let identity = RegistrationShift::identity_fallback();
        assert_eq!(
            (identity.get_dx().unwrap(), identity.get_dy().unwrap()),
            (0, 0)
        );
        assert_eq!(identity.get_confidence().unwrap(), 0);
    }

    // ---------------- error enum ----------------

    #[test]
    fn test_error_codes_are_stable_and_recovery_flags_correct() {
        // Append-only guard: these numbers must never change.
        assert_eq!(SlimerError::PngMagicInvalid.code(), 120);
        assert_eq!(SlimerError::PngDeflateCorrupt.code(), 127);
        assert_eq!(SlimerError::IoReadFailed.code(), 200);
        assert_eq!(SlimerError::ValCorruptedBitFlipDetected.code(), 304);
        assert_eq!(SlimerError::RegInsufficientContrast.code(), 402);
        assert_eq!(SlimerError::PipeBufferExhaustion.code(), 503);
        assert_eq!(SlimerError::CliModeUnknown.code(), 601);
        assert_eq!(SlimerError::MeshFilenameInvalid.code(), 700);
        assert_eq!(std::mem::size_of::<SlimerError>(), 2);

        assert!(SlimerError::IoReadFailed.is_retryable());
        assert!(SlimerError::IoWriteFailed.is_retryable());
        assert!(!SlimerError::IoUnexpectedEof.is_retryable());
        assert!(!SlimerError::IoInputNotFound.is_retryable());
        assert!(!SlimerError::PngIdatCrcMismatch.is_retryable());
        assert!(!SlimerError::RegInsufficientContrast.is_retryable());
        assert!(SlimerError::RegInsufficientContrast.is_degradable());
        assert!(!SlimerError::RegDimMismatch.is_degradable());
    }

    #[test]
    fn test_io_error_classification() {
        assert_eq!(
            classify_read_error(ErrorKind::UnexpectedEof),
            SlimerError::IoUnexpectedEof
        );
        assert_eq!(
            classify_read_error(ErrorKind::Other),
            SlimerError::IoReadFailed
        );
        assert_eq!(
            classify_open_error(ErrorKind::NotFound),
            SlimerError::IoInputNotFound
        );
        assert_eq!(
            classify_open_error(ErrorKind::PermissionDenied),
            SlimerError::IoPermissionDenied
        );
        assert_eq!(
            classify_create_error(ErrorKind::AlreadyExists),
            SlimerError::IoOutputAlreadyExists
        );
        assert_eq!(
            classify_create_error(ErrorKind::Other),
            SlimerError::IoCreateFailed
        );
    }

    // ---------------- checksums ----------------

    #[test]
    fn test_crc32_adler32_known_vectors() {
        let mut crc = Crc32::new();
        crc.update(b"123456789");
        assert_eq!(crc.finish(), 0xCBF4_3926);
        let mut iend_crc = Crc32::new();
        iend_crc.update(b"IEND");
        assert_eq!(iend_crc.finish(), 0xAE42_6082);
        let mut adler = Adler32::new();
        adler.update(b"123456789");
        assert_eq!(adler.finish(), 0x091E_01DE);
    }

    // ---------------- filters & luma ----------------

    #[test]
    fn test_paeth_predictor_and_unfilter_all_types() {
        assert_eq!(png_paeth_predictor(10, 10, 10).unwrap(), 10);
        assert_eq!(png_paeth_predictor(100, 20, 20).unwrap(), 100);
        assert_eq!(png_paeth_predictor(20, 100, 20).unwrap(), 100);
        assert_eq!(png_unfilter_byte(5, 0, 1, 2, 3).unwrap(), 5);
        assert_eq!(png_unfilter_byte(5, 1, 250, 0, 0).unwrap(), 255);
        assert_eq!(png_unfilter_byte(10, 2, 0, 250, 0).unwrap(), 4); // wraps
        assert_eq!(png_unfilter_byte(1, 3, 10, 20, 0).unwrap(), 16);
        assert_eq!(png_unfilter_byte(1, 4, 0, 5, 0).unwrap(), 6);
    }

    #[test]
    fn test_unfilter_rejects_invalid_filter_type() {
        assert_eq!(
            png_unfilter_byte(1, 5, 0, 0, 0),
            Err(SlimerError::PngScanlineFilterInvalid)
        );
    }

    #[test]
    fn test_luma_from_pixel_color_kinds() {
        assert_eq!(luma_from_pixel(PngColorKind::Gray, &[42]).unwrap(), 42);
        assert_eq!(
            luma_from_pixel(PngColorKind::GrayAlpha, &[42, 0]).unwrap(),
            42
        );
        assert_eq!(
            luma_from_pixel(PngColorKind::Rgb, &[255, 255, 255]).unwrap(),
            255
        );
        assert_eq!(
            luma_from_pixel(PngColorKind::Rgba, &[255, 0, 0, 9]).unwrap(),
            76
        );
        assert_eq!(
            luma_from_pixel(PngColorKind::Rgb, &[1, 2]),
            Err(SlimerError::ValIndexOutOfBounds)
        );
    }

    // ---------------- Huffman ----------------

    #[test]
    fn test_huffman_rejects_oversubscribed_code() {
        let mut table = HuffmanTable::new();
        assert_eq!(
            table.build(&[1, 1, 1]),
            Err(SlimerError::PngHuffmanTableInvalid)
        );
        assert_eq!(table.build(&[16]), Err(SlimerError::PngHuffmanTableInvalid));
    }

    #[test]
    fn test_huffman_accepts_complete_and_fixed_codes() {
        let mut table = HuffmanTable::new();
        assert!(table.build(&[2, 2, 2, 2]).is_ok());
        assert!(table.build(&fixed_litlen_code_lengths()).is_ok());
        assert!(table.build(&[5u8; MAX_DIST_SYMBOLS]).is_ok());
    }

    // ---------------- shift solver ----------------

    #[test]
    fn test_solve_1d_shift_positive_and_negative() {
        let mut a = [100u64; 64];
        let mut b = [100u64; 64];
        a[20] = 5000;
        b[27] = 5000;
        let (shift, confidence) = solve_1d_shift(&a, &b, 64, 16).unwrap();
        assert_eq!(shift, 7);
        assert!(confidence > 0);

        let mut c = [100u64; 64];
        c[15] = 5000;
        let (negative_shift, _) = solve_1d_shift(&a, &c, 64, 16).unwrap();
        assert_eq!(negative_shift, -5);
    }

    #[test]
    fn test_solve_1d_shift_rejects_bad_arguments() {
        let signal = [0u64; 16];
        assert_eq!(
            solve_1d_shift(&signal, &signal, 4, 8),
            Err(SlimerError::ValIntegrityDimUnderflow)
        );
        assert_eq!(
            solve_1d_shift(&signal, &signal, 32, 8),
            Err(SlimerError::ValArgumentInvalid)
        );
        assert_eq!(
            solve_1d_shift(&signal, &signal, 16, -1),
            Err(SlimerError::ValArgumentInvalid)
        );
    }

    #[test]
    fn test_solve_1d_shift_flat_signal_prefers_zero_with_zero_confidence() {
        let flat = [777u64; 64];
        assert_eq!(solve_1d_shift(&flat, &flat, 64, 32).unwrap(), (0, 0));
    }

    // ---------------- writer formatting ----------------

    #[test]
    fn test_stack_buf_writer_number_formatting() {
        let path = unique_test_path("numbers.txt");
        {
            let mut file = File::create(&path).unwrap();
            let mut writer = StackBufWriter::new(&mut file);
            writer.write_u64(0).unwrap();
            writer.write_bytes(b" ").unwrap();
            writer.write_u64(4_294_967_295).unwrap();
            writer.write_bytes(b" ").unwrap();
            writer.write_i32(i32::MIN).unwrap();
            writer.write_bytes(b" ").unwrap();
            writer.write_u64_zero_padded(42, 4).unwrap();
            writer.write_bytes(b" ").unwrap();
            writer.write_u64(u64::MAX).unwrap();
            writer.write_bytes(b"\n").unwrap();
            writer.finish().unwrap();
        }
        assert_eq!(
            std::fs::read_to_string(&path).unwrap(),
            "0 4294967295 -2147483648 0042 18446744073709551615\n"
        );
        remove_test_file(&path);
    }

    #[test]
    fn test_stack_buf_writer_large_write_preserves_order() {
        let path = unique_test_path("large.txt");
        let large = vec![b'x'; 10_000];
        {
            let mut file = File::create(&path).unwrap();
            let mut writer = StackBufWriter::new(&mut file);
            writer.write_bytes(b"ab").unwrap();
            writer.write_bytes(&large).unwrap();
            writer.write_bytes(b"cd").unwrap();
            writer.finish().unwrap();
        }
        let content = std::fs::read(&path).unwrap();
        assert_eq!(content.len(), 10_004);
        assert_eq!(&content[..2], b"ab");
        assert_eq!(&content[10_002..], b"cd");
        remove_test_file(&path);
    }

    #[test]
    fn test_normalized_uv_pixel_centres() {
        let path = unique_test_path("uv.txt");
        {
            let mut file = File::create(&path).unwrap();
            let mut writer = StackBufWriter::new(&mut file);
            write_normalized_uv(&mut writer, 0, 64).unwrap();
            writer.write_bytes(b" ").unwrap();
            write_normalized_uv(&mut writer, 63, 64).unwrap();
            assert_eq!(
                write_normalized_uv(&mut writer, 64, 64),
                Err(SlimerError::ValArgumentInvalid)
            );
            writer.finish().unwrap();
        }
        assert_eq!(std::fs::read_to_string(&path).unwrap(), "0.0078 0.9921");
        remove_test_file(&path);
    }

    #[test]
    fn test_extract_and_validate_mtl_names() {
        assert_eq!(
            extract_filename_bytes(b"/a/b/photo.png").unwrap(),
            b"photo.png"
        );
        assert_eq!(extract_filename_bytes(b"C:\\x\\y.png").unwrap(), b"y.png");
        assert_eq!(extract_filename_bytes(b"plain.png").unwrap(), b"plain.png");
        assert_eq!(
            extract_filename_bytes(b"/dir/"),
            Err(SlimerError::MeshFilenameInvalid)
        );
        assert!(validate_mtl_reference_name(b"my photo.png").is_ok());
        assert_eq!(
            validate_mtl_reference_name(b"evil\nmap_Kd /etc/passwd"),
            Err(SlimerError::MeshFilenameInvalid)
        );
        assert_eq!(
            validate_mtl_reference_name(b""),
            Err(SlimerError::MeshFilenameInvalid)
        );
        let too_long = vec![b'a'; MAX_MTL_REFERENCE_NAME_BYTES + 1];
        assert_eq!(
            validate_mtl_reference_name(&too_long),
            Err(SlimerError::MeshFilenameInvalid)
        );
    }

    #[test]
    fn test_grid_and_axis_helpers() {
        assert_eq!(count_grid_samples(64).unwrap(), 4);
        assert_eq!(count_grid_samples(65).unwrap(), 5);
        assert_eq!(count_grid_samples(8).unwrap(), 1);
        assert_eq!(grid_vertex_index(1, 2, 4).unwrap(), 7);
        assert_eq!(
            grid_vertex_index(u32::MAX, 2, 4),
            Err(SlimerError::MeshIndexOverflow)
        );
        assert_eq!(up_axis_coordinate(64, 0).unwrap(), 63);
        assert_eq!(up_axis_coordinate(64, 63).unwrap(), 0);
        assert_eq!(
            up_axis_coordinate(64, 64),
            Err(SlimerError::ValIndexOutOfBounds)
        );
    }

    #[test]
    fn test_pseudo_depth_and_windowed_disparity() {
        let mut line_a = [0u8; 32];
        let mut line_b = [0u8; 32];
        line_a[10..13].fill(200);
        line_b[12..15].fill(200); // same feature shifted right by 2
        assert_eq!(
            estimate_local_disparity(&line_a, &line_b, 11, 0, 32).unwrap(),
            2
        );
        assert_eq!(
            estimate_local_disparity(&line_a, &line_a, 11, 0, 32).unwrap(),
            0
        );
        // Featureless rows: every disparity ties, tie rule must pick 0.
        let flat = [50u8; 32];
        assert_eq!(
            estimate_local_disparity(&flat, &flat, 16, 0, 32).unwrap(),
            0
        );
        assert_eq!(compute_pseudo_depth(200, 2).unwrap(), 56);
        assert_eq!(compute_pseudo_depth(0, -6).unwrap(), -18);
        assert_eq!(compute_pseudo_depth(255, 6).unwrap(), 81);
    }

    // ---------------- PNG writer ----------------

    #[test]
    fn test_png_writer_roundtrip_gray_stored_blocks() {
        // Regression for v1 defect #1: stored Deflate blocks were never closed, so Slimer
        // could not decode its own 2d output (one stored block per row is written here).
        let pattern = |x: usize, y: usize| ((x * 7 + y * 13) % 256) as u8;
        let path = write_gray_png_fixture("roundtrip.png", 37, 11, pattern);
        let rows = read_luma_rows(&path);
        assert_eq!(rows.len(), 11);
        for (y, row) in rows.iter().enumerate() {
            assert_eq!(row.len(), 37);
            for (x, &value) in row.iter().enumerate() {
                assert_eq!(value, pattern(x, y), "pixel ({}, {})", x, y);
            }
        }
        remove_test_file(&path);
    }

    #[test]
    fn test_png_writer_enforces_exact_row_count() {
        let row = [0u8; 8];
        let overflow_path = unique_test_path("rows_overflow.png");
        {
            let mut file = File::create(&overflow_path).unwrap();
            let mut writer = PngGrayStreamWriter::begin(
                &mut file,
                BoundedWidth::new(8).unwrap(),
                BoundedHeight::new(8).unwrap(),
            )
            .unwrap();
            assert_eq!(
                writer.write_row(&row[..4]),
                Err(SlimerError::ValArgumentInvalid)
            );
            for _ in 0..8 {
                writer.write_row(&row).unwrap();
            }
            assert_eq!(
                writer.write_row(&row),
                Err(SlimerError::PipeRowCountExceeded)
            );
        }
        remove_test_file(&overflow_path);

        let incomplete_path = unique_test_path("rows_incomplete.png");
        {
            let mut file = File::create(&incomplete_path).unwrap();
            let mut writer = PngGrayStreamWriter::begin(
                &mut file,
                BoundedWidth::new(8).unwrap(),
                BoundedHeight::new(8).unwrap(),
            )
            .unwrap();
            for _ in 0..7 {
                writer.write_row(&row).unwrap();
            }
            assert_eq!(writer.finish(), Err(SlimerError::PipeRowCountIncomplete));
        }
        remove_test_file(&incomplete_path);
    }

    // ---------------- PNG header parsing ----------------

    #[test]
    fn test_parse_header_rejects_invalid_inputs() {
        let zlib = zlib_wrap_stored(&[0u8; 9 * 8]);
        let mut bad_magic = valid_gray_8x8_png();
        bad_magic[0] = 0;
        let mut bad_ihdr_crc = valid_gray_8x8_png();
        bad_ihdr_crc[29] ^= 0xFF;
        let truncated_file = valid_gray_8x8_png()[..20].to_vec();
        let mut no_idat = PNG_SIGNATURE.to_vec();
        {
            let mut ihdr = Vec::new();
            ihdr.extend_from_slice(&8u32.to_be_bytes());
            ihdr.extend_from_slice(&8u32.to_be_bytes());
            ihdr.extend_from_slice(&[8, 0, 0, 0, 0]);
            no_idat.extend(png_chunk(b"IHDR", &ihdr));
            no_idat.extend(png_chunk(b"IEND", &[]));
        }

        let cases: Vec<(Vec<u8>, SlimerError)> = vec![
            (bad_magic, SlimerError::PngMagicInvalid),
            (bad_ihdr_crc, SlimerError::PngIhdrCrcMismatch),
            (
                build_png_bytes(8, 8, 16, 0, 0, &[], &zlib),
                SlimerError::PngUnsupportedBitDepth,
            ),
            (
                build_png_bytes(8, 8, 8, 3, 0, &[], &zlib),
                SlimerError::PngUnsupportedColorType,
            ),
            (
                build_png_bytes(8, 8, 8, 0, 1, &[], &zlib),
                SlimerError::PngUnsupportedInterlace,
            ),
            (
                build_png_bytes(4, 8, 8, 0, 0, &[], &zlib),
                SlimerError::ValIntegrityDimUnderflow,
            ),
            (
                build_png_bytes(2000, 8, 8, 0, 0, &[], &zlib),
                SlimerError::ValIntegrityDimOverflow,
            ),
            (
                build_png_bytes(8, 2000, 8, 0, 0, &[], &zlib),
                SlimerError::ValIntegrityDimOverflow,
            ),
            (
                build_png_bytes(8, 8, 8, 0, 0, &[png_chunk(b"ABCD", &[1])], &zlib),
                SlimerError::PngUnsupportedCriticalChunk,
            ),
            (no_idat, SlimerError::PngCorruptChunk),
            (truncated_file, SlimerError::IoUnexpectedEof),
        ];
        for (index, (bytes, expected)) in cases.iter().enumerate() {
            let path = write_test_file(&format!("bad_header_{}.png", index), bytes);
            let outcome = {
                let mut file = File::open(&path).unwrap();
                parse_png_header(&mut file).err()
            };
            remove_test_file(&path);
            assert_eq!(outcome, Some(*expected), "header case {}", index);
        }
    }

    #[test]
    fn test_parse_header_skips_ancillary_chunks() {
        let raw: Vec<u8> = (0..8usize)
            .flat_map(|y| std::iter::once(0u8).chain((0..8usize).map(move |x| (x * 10 + y) as u8)))
            .collect();
        let bytes = build_png_bytes(
            8,
            8,
            8,
            0,
            0,
            &[
                png_chunk(b"tEXt", b"Comment\0hello"),
                png_chunk(b"gAMA", &[0, 0, 0xB1, 0x8F]),
            ],
            &zlib_wrap_stored(&raw),
        );
        let path = write_test_file("ancillary.png", &bytes);
        let rows = read_luma_rows(&path);
        remove_test_file(&path);
        assert_eq!(rows.len(), 8);
        assert_eq!(rows[3][5], 53);
        assert_eq!(rows[7][7], 77);
    }

    // ---------------- scanline decoding ----------------

    #[test]
    fn test_decode_rgb_rows_with_up_and_sub_filters() {
        let mut raw = Vec::new();
        raw.push(0); // row 0: None, pure red
        for _ in 0..8 {
            raw.extend_from_slice(&[255, 0, 0]);
        }
        raw.push(2); // row 1: Up with zero deltas -> copy of row 0
        raw.extend_from_slice(&[0u8; 24]);
        raw.push(1); // row 2: Sub, first pixel green, zero deltas -> all green
        raw.extend_from_slice(&[0, 255, 0]);
        raw.extend_from_slice(&[0u8; 21]);
        for _ in 0..5 {
            raw.push(2); // rows 3..7: Up -> copies of row 2
            raw.extend_from_slice(&[0u8; 24]);
        }
        let path = write_test_file(
            "rgb.png",
            &build_png_bytes(8, 8, 8, 2, 0, &[], &zlib_wrap_stored(&raw)),
        );
        let rows = read_luma_rows(&path);
        remove_test_file(&path);
        assert_eq!(rows.len(), 8);
        assert!(rows[0].iter().all(|&v| v == 76)); // (255*77) >> 8
        assert!(rows[1].iter().all(|&v| v == 76));
        for row in &rows[2..] {
            assert!(row.iter().all(|&v| v == 149)); // (255*150) >> 8
        }
    }

    #[test]
    fn test_decode_gray_rows_with_average_and_paeth_filters() {
        let mut raw = Vec::new();
        raw.push(0);
        raw.extend_from_slice(&[10, 20, 30, 40, 50, 60, 70, 80]);
        raw.push(3); // Average with zero deltas
        raw.extend_from_slice(&[0u8; 8]);
        raw.push(4); // Paeth with zero deltas
        raw.extend_from_slice(&[0u8; 8]);
        raw.push(1); // Sub
        raw.extend_from_slice(&[7, 1, 1, 1, 1, 1, 1, 1]);
        for _ in 0..4 {
            raw.push(0);
            raw.extend_from_slice(&[0u8; 8]);
        }
        let path = write_test_file(
            "gray_filters.png",
            &build_png_bytes(8, 8, 8, 0, 0, &[], &zlib_wrap_stored(&raw)),
        );
        let rows = read_luma_rows(&path);
        remove_test_file(&path);
        assert_eq!(rows[0], vec![10, 20, 30, 40, 50, 60, 70, 80]);
        assert_eq!(rows[1], vec![5, 12, 21, 30, 40, 50, 60, 70]);
        assert_eq!(rows[2], vec![5, 12, 21, 30, 40, 50, 60, 70]);
        assert_eq!(rows[3], vec![7, 8, 9, 10, 11, 12, 13, 14]);
        assert_eq!(rows[7], vec![0u8; 8]);
    }

    #[test]
    fn test_decode_gray_alpha_ignores_alpha() {
        let raw: Vec<u8> = (0..8)
            .flat_map(|_| std::iter::once(0u8).chain((0..8u8).flat_map(|x| [x * 10, 255])))
            .collect();
        let path = write_test_file(
            "gray_alpha.png",
            &build_png_bytes(8, 8, 8, 4, 0, &[], &zlib_wrap_stored(&raw)),
        );
        let rows = read_luma_rows(&path);
        remove_test_file(&path);
        assert_eq!(rows[4], vec![0, 10, 20, 30, 40, 50, 60, 70]);
    }

    #[test]
    fn test_decode_truncated_image_data_errors() {
        // Header claims 8 rows; the compressed stream only holds 4.
        let raw: Vec<u8> = (0..4)
            .flat_map(|_| std::iter::once(0u8).chain(std::iter::repeat(9u8).take(8)))
            .collect();
        let path = write_test_file(
            "truncated.png",
            &build_png_bytes(8, 8, 8, 0, 0, &[], &zlib_wrap_stored(&raw)),
        );
        let mut results = Vec::new();
        {
            let mut file = File::open(&path).unwrap();
            let header = parse_png_header(&mut file).unwrap();
            let mut reader = PngScanlineReader::new(&mut file, &header).unwrap();
            let mut row = [0u8; MAX_IMAGE_WIDTH];
            for _ in 0..5 {
                results.push(reader.next_luma_row(&mut row));
            }
        }
        remove_test_file(&path);
        assert!(results[..4].iter().all(|r| *r == Ok(true)));
        assert_eq!(results[4], Err(SlimerError::PngImageDataTruncated));
    }

    // ---------------- Deflate / IDAT stream ----------------

    #[test]
    fn test_deflate_fixed_huffman_literals_and_back_reference() {
        let path = write_test_file(
            "fixed.idat",
            &png_chunk(b"IDAT", &fixed_huffman_abc_then_match(2)),
        );
        let (decoded, after_end) = {
            let mut file = File::open(&path).unwrap();
            let mut stream = DeflateByteStream::new(&mut file, 0).unwrap();
            let decoded: Vec<u8> = (0..6).map(|_| stream.next_byte().unwrap()).collect();
            (decoded, stream.next_byte())
        };
        remove_test_file(&path);
        assert_eq!(decoded, b"abcabc");
        assert_eq!(after_end, Err(SlimerError::PngImageDataTruncated));
    }

    #[test]
    fn test_deflate_rejects_distance_beyond_history() {
        // Distance symbol 3 = distance 4, but only 3 bytes exist (v1 read stale zeros).
        let path = write_test_file(
            "far.idat",
            &png_chunk(b"IDAT", &fixed_huffman_abc_then_match(3)),
        );
        let results: Vec<Result<u8, SlimerError>> = {
            let mut file = File::open(&path).unwrap();
            let mut stream = DeflateByteStream::new(&mut file, 0).unwrap();
            (0..4).map(|_| stream.next_byte()).collect()
        };
        remove_test_file(&path);
        assert_eq!(results[..3], [Ok(b'a'), Ok(b'b'), Ok(b'c')]);
        assert_eq!(results[3], Err(SlimerError::PngDeflateDistanceTooFar));
    }

    #[test]
    fn test_deflate_spans_multiple_and_empty_idat_chunks() {
        // Regression for v1 defect #2: zero-length IDAT chunk caused usize underflow.
        let zlib = fixed_huffman_abc_then_match(2);
        let mut bytes = png_chunk(b"IDAT", &zlib[..3]);
        bytes.extend(png_chunk(b"IDAT", &[]));
        bytes.extend(png_chunk(b"IDAT", &zlib[3..]));
        let path = write_test_file("multi.idat", &bytes);
        let decoded: Vec<u8> = {
            let mut file = File::open(&path).unwrap();
            let mut stream = DeflateByteStream::new(&mut file, 0).unwrap();
            (0..6).map(|_| stream.next_byte().unwrap()).collect()
        };
        remove_test_file(&path);
        assert_eq!(decoded, b"abcabc");
    }

    #[test]
    fn test_deflate_detects_idat_crc_mismatch() {
        let zlib = fixed_huffman_abc_then_match(2);
        let mut first_chunk = png_chunk(b"IDAT", &zlib[..3]);
        let last = first_chunk.len() - 1;
        first_chunk[last] ^= 0xFF; // corrupt stored CRC
        let mut bytes = first_chunk;
        bytes.extend(png_chunk(b"IDAT", &zlib[3..]));
        let path = write_test_file("badcrc.idat", &bytes);
        let first = {
            let mut file = File::open(&path).unwrap();
            let mut stream = DeflateByteStream::new(&mut file, 0).unwrap();
            stream.next_byte() // needs bits from chunk 2 -> crosses -> CRC verified
        };
        remove_test_file(&path);
        assert_eq!(first, Err(SlimerError::PngIdatCrcMismatch));
    }

    #[test]
    fn test_deflate_rejects_bad_zlib_headers_and_chunk_sequence() {
        let cases: Vec<(Vec<u8>, SlimerError)> = vec![
            (
                png_chunk(b"IDAT", &[0x78, 0x02, 0, 0]),
                SlimerError::PngZlibHeaderInvalid,
            ),
            (
                png_chunk(b"IDAT", &[0x78, 0x20, 0, 0]),
                SlimerError::PngZlibPresetDictionary,
            ),
            (
                png_chunk(b"tEXt", &[0x78, 0x01, 0, 0]),
                SlimerError::PngChunkSequenceInvalid,
            ),
        ];
        for (index, (bytes, expected)) in cases.iter().enumerate() {
            let path = write_test_file(&format!("zlib_case_{}.idat", index), bytes);
            let outcome = {
                let mut file = File::open(&path).unwrap();
                DeflateByteStream::new(&mut file, 0).err()
            };
            remove_test_file(&path);
            assert_eq!(outcome, Some(*expected), "zlib case {}", index);
        }
    }

    // ---------------- registration & outputs (end-to-end) ----------------

    #[test]
    fn test_register_recovers_known_translation() {
        let path_a = write_gray_png_fixture("reg_a.png", 64, 64, square_scene(20, 25));
        let path_b = write_gray_png_fixture("reg_b.png", 64, 64, square_scene(23, 23));
        let shift = {
            let mut file_a = File::open(&path_a).unwrap();
            let mut file_b = File::open(&path_b).unwrap();
            register_png_images(&mut file_a, &mut file_b).unwrap()
        };
        remove_test_file(&path_a);
        remove_test_file(&path_b);
        assert_eq!(shift.get_dx().unwrap(), 3);
        assert_eq!(shift.get_dy().unwrap(), -2);
        assert!(shift.get_confidence().unwrap() >= REGISTRATION_MIN_CONFIDENCE_PPM);
    }

    #[test]
    fn test_register_flat_images_reports_insufficient_contrast() {
        let path_a = write_gray_png_fixture("flat_a.png", 32, 32, |_, _| 50);
        let path_b = write_gray_png_fixture("flat_b.png", 32, 32, |_, _| 50);
        let outcome = {
            let mut file_a = File::open(&path_a).unwrap();
            let mut file_b = File::open(&path_b).unwrap();
            register_png_images(&mut file_a, &mut file_b)
        };
        remove_test_file(&path_a);
        remove_test_file(&path_b);
        assert_eq!(outcome, Err(SlimerError::RegInsufficientContrast));
        assert!(SlimerError::RegInsufficientContrast.is_degradable());
    }

    #[test]
    fn test_register_rejects_dimension_mismatch() {
        let path_a = write_gray_png_fixture("dim_a.png", 64, 64, square_scene(20, 25));
        let path_b = write_gray_png_fixture("dim_b.png", 64, 32, square_scene(20, 10));
        let outcome = {
            let mut file_a = File::open(&path_a).unwrap();
            let mut file_b = File::open(&path_b).unwrap();
            register_png_images(&mut file_a, &mut file_b)
        };
        remove_test_file(&path_a);
        remove_test_file(&path_b);
        assert_eq!(outcome, Err(SlimerError::RegDimMismatch));
    }

    /// Registers `path_reference` against `path_target`, writes the 2d output, and
    /// returns the decoded output rows.
    fn register_and_enhance(path_reference: &Path, path_target: &Path, tag: &str) -> Vec<Vec<u8>> {
        let out_path = unique_test_path(tag);
        {
            let mut file_a = File::open(path_reference).unwrap();
            let mut file_b = File::open(path_target).unwrap();
            let shift = register_png_images(&mut file_a, &mut file_b).unwrap();
            let mut out_file = File::create(&out_path).unwrap();
            enhance_png_streamed(&mut file_a, &mut file_b, &shift, &mut out_file).unwrap();
        }
        let rows = read_luma_rows(&out_path);
        remove_test_file(&out_path);
        rows
    }

    #[test]
    fn test_enhance_registered_pair_reproduces_reference_both_directions() {
        // Registered partners carry identical values, so the fused output must equal the
        // reference exactly. Running both directions covers dy < 0 and dy > 0 (pre-skip).
        let path_a = write_gray_png_fixture("enh_a.png", 64, 64, square_scene(20, 25));
        let path_b = write_gray_png_fixture("enh_b.png", 64, 64, square_scene(23, 23));
        let reference_a = read_luma_rows(&path_a);
        let reference_b = read_luma_rows(&path_b);
        let fused_ab = register_and_enhance(&path_a, &path_b, "enh_ab.png");
        let fused_ba = register_and_enhance(&path_b, &path_a, "enh_ba.png");
        remove_test_file(&path_a);
        remove_test_file(&path_b);
        assert_eq!(fused_ab, reference_a);
        assert_eq!(fused_ba, reference_b);
    }

    /// Runs one 3D generator into a temp file and returns the text it produced.
    fn run_text_generator(
        tag: &str,
        generator: impl Fn(
            &mut File,
            &mut File,
            &RegistrationShift,
            &mut File,
        ) -> Result<(), SlimerError>,
    ) -> String {
        let path_a =
            write_gray_png_fixture(&format!("{}_a.png", tag), 64, 64, square_scene(20, 25));
        let path_b =
            write_gray_png_fixture(&format!("{}_b.png", tag), 64, 64, square_scene(23, 23));
        let out_path = unique_test_path(tag);
        {
            let mut file_a = File::open(&path_a).unwrap();
            let mut file_b = File::open(&path_b).unwrap();
            let shift = register_png_images(&mut file_a, &mut file_b).unwrap();
            let mut out_file = File::create(&out_path).unwrap();
            generator(&mut file_a, &mut file_b, &shift, &mut out_file).unwrap();
        }
        let text = std::fs::read_to_string(&out_path).unwrap();
        remove_test_file(&path_a);
        remove_test_file(&path_b);
        remove_test_file(&out_path);
        text
    }

    #[test]
    fn test_wireframe_obj_vertex_and_edge_counts() {
        let text = run_text_generator("wire.obj", generate_3d_wireframe_obj);
        // 64x64 at step 16 -> 4 x 4 grid: 16 vertices, 4*3 + 3*4 = 24 edges.
        assert_eq!(text.lines().filter(|l| l.starts_with("v ")).count(), 16);
        assert_eq!(text.lines().filter(|l| l.starts_with("l ")).count(), 24);
        assert!(text.contains("o terrain_wireframe"));
    }

    #[test]
    fn test_textured_mesh_obj_and_mtl_contents() {
        let text = run_text_generator("mesh.obj", |a, b, s, o| {
            generate_3d_textured_mesh_obj(a, b, s, o, b"/some/dir/terrain.mtl")
        });
        assert!(text.contains("mtllib terrain.mtl\n")); // file name only, no directory
        assert_eq!(text.lines().filter(|l| l.starts_with("v ")).count(), 16);
        assert_eq!(text.lines().filter(|l| l.starts_with("vt ")).count(), 16);
        assert_eq!(text.lines().filter(|l| l.starts_with("f ")).count(), 18); // 2*3*3

        let mtl_path = unique_test_path("terrain.mtl");
        {
            let mut mtl_file = File::create(&mtl_path).unwrap();
            write_mtl_material_file(&mut mtl_file, b"/home/user/photos/photo.png").unwrap();
        }
        let mtl_text = std::fs::read_to_string(&mtl_path).unwrap();
        remove_test_file(&mtl_path);
        assert!(mtl_text.contains("map_Kd photo.png\n"));
        assert!(!mtl_text.contains("/home/user"));
    }

    #[test]
    fn test_textured_mesh_rejects_injected_material_name() {
        let path_a = write_gray_png_fixture("inject_a.png", 32, 32, square_scene(5, 5));
        let out_path = unique_test_path("inject.obj");
        let outcome = {
            let mut file_a = File::open(&path_a).unwrap();
            let mut file_b = File::open(&path_a).unwrap();
            let mut out_file = File::create(&out_path).unwrap();
            generate_3d_textured_mesh_obj(
                &mut file_a,
                &mut file_b,
                &RegistrationShift::identity_fallback(),
                &mut out_file,
                b"bad\nname.mtl",
            )
        };
        remove_test_file(&path_a);
        remove_test_file(&out_path);
        assert_eq!(outcome, Err(SlimerError::MeshFilenameInvalid));
    }

    #[test]
    fn test_svg_wireframe_structure_and_edge_count() {
        let text = run_text_generator("wire.svg", generate_3d_wireframe_svg);
        assert!(text.starts_with("<svg xmlns=\"http://www.w3.org/2000/svg\" viewBox=\""));
        assert!(text.ends_with("</g>\n</svg>\n"));
        assert_eq!(text.matches("<line ").count(), 24);
    }

    // ---------------- error log ----------------

    #[test]
    fn test_error_log_entry_format_and_tag_sanitising() {
        let path = unique_test_path("error.log");
        {
            let mut log_file = File::create(&path).unwrap();
            append_error_log_entry(&mut log_file, SlimerError::PngDeflateCorrupt, b"decode")
                .unwrap();
            append_error_log_entry(&mut log_file, SlimerError::IoReadFailed, b"Bad Tag/../x")
                .unwrap();
        }
        let text = std::fs::read_to_string(&path).unwrap();
        remove_test_file(&path);
        let lines: Vec<&str> = text.lines().collect();
        assert_eq!(lines.len(), 2);
        assert!(lines[0].starts_with("slimer_error unix_time="));
        assert!(lines[0].ends_with(" code=127 stage=decode"));
        assert!(lines[1].ends_with(" code=200 stage=invalid_tag"));
    }
}
