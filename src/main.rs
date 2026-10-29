//! main.rs
//!
//! # Slimer CLI runner (v2.0 MVP)
//!
//! A thin command-line shell around `slimer_slim_image_registration_module`.
//!
//! ## Responsibilities
//! 1. Parse arguments (`-h` help, mode, two input PNGs, optional output name).
//! 2. Resolve all paths to absolute paths.
//! 3. Run one command with the Three-Level Recovery Hierarchy:
//!    - Tier 1: bounded retry with backoff for retryable I/O codes.
//!    - Tier 2: identity fallback when registration finds too little contrast. On any
//!      failure, partial output files are deleted.
//!    - Tier 3: every attempt re-opens all files and re-parses from scratch. No state
//!      carries over between attempts except the immutable `CommandPlan`.
//! 4. Report results.
//!    - Errors are printed as stage + numeric code only (no paths, no OS text).
//!    - Each failure is appended to `slimer_error_log.txt` next to the executable.
//!
//! ## Process lifetime policy
//! This is a one-shot batch tool: completing (or definitively failing) its single
//! command IS its job. `main` returns an `ExitCode` instead of calling
//! `process::exit` (v1), so destructors and flushes always run.
//! Status output never panics: `println!`/`eprintln!` panic on a closed pipe, so they
//! are replaced by `write_stdout_line` / `write_stderr_line`.
//!
//! ## Heap note
//! The engine is zero-heap. This runner uses std's OS-facing types (`OsString`,
//! `PathBuf`), which allocate. The argument list is bounded to
//! `MAX_CLI_ARGUMENTS + 1` entries. `std::env::args_os()` is used because
//! `std::env::args()` PANICS on non-UTF-8 arguments.
//!
//! ## Output safety policy
//! Outputs are created with `create_new`: an existing file is NEVER overwritten.
//! This also makes it impossible to clobber an input (a v1 data-loss bug).

#![forbid(unsafe_code)]

mod slimer_slim_image_registration_module;

use slimer_slim_image_registration_module::{
    RegistrationShift, SlimerError, append_error_log_entry, classify_create_error,
    classify_open_error, enhance_png_streamed, generate_3d_textured_mesh_obj,
    generate_3d_wireframe_obj, generate_3d_wireframe_svg, register_png_images,
    write_mtl_material_file,
};
use std::ffi::OsString;
use std::fs::{File, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::time::Duration;

/// Tier-1 bound: total attempts for one command (1 initial + 2 retries).
const MAX_COMMAND_ATTEMPTS: u32 = 3;
/// Linear backoff step between attempts.
const RETRY_BACKOFF_BASE_MILLISECONDS: u64 = 100;
/// program + MODE + REFERENCE + TARGET + OUTPUT.
const MAX_CLI_ARGUMENTS: usize = 5;
/// Error log, appended next to the executable (e.g. target/release/ under cargo run).
const ERROR_LOG_FILE_NAME: &str = "slimer_error_log.txt";
const EXIT_CODE_COMMAND_FAILED: u8 = 1;
const EXIT_CODE_USAGE_ERROR: u8 = 2;

const HELP_TEXT: &str = "\
SLIMER v2.0 - slim, zero-heap image registration (alignment) and fusion

USAGE
  slimer <MODE> <REFERENCE_IMAGE.png> <TARGET_IMAGE.png> [OUTPUT_FILE]
  slimer -h | --help | help
  via cargo:  cargo run --release -- <MODE> <REFERENCE> <TARGET> [OUTPUT]

WHAT IT DOES
  1. Streams both PNGs one scanline at a time (never loads a whole image).
  2. Estimates the global shift (dx, dy) that maps REFERENCE onto TARGET:
     rows and columns of brightness are summed into two 1D 'projection'
     signatures per image, which are matched by Sum of Absolute Differences.
     Search range: +/-32 px, limited to half the image size.
     Convention: content at reference (x, y) is at target (x + dx, y + dy).
     Only translation is estimated (no rotation, scale, or perspective).
  3. Reports a confidence score from 0 to 1,000,000 ppm. Below 40,000 ppm
     (too little contrast or structure) Slimer warns and continues with
     dx = 0, dy = 0 (identity fallback) instead of failing.
  4. Writes the output selected by MODE.

MODES
  2d    Fused 8-bit GRAYSCALE PNG, same size as the reference. Each pixel is
        averaged with its registered partner where the two agree (difference
        < 32 levels); elsewhere the reference pixel is kept. Reduces noise; it
        is NOT super-resolution. Output is uncompressed (about W x H bytes).
  3d    Wavefront OBJ wireframe: vertices (v) every 16 px plus grid edges (l).
        Open with MeshLab, Blender, etc.
  mesh  Wavefront OBJ surface with UV texture coordinates (vt) and triangle
        faces (f), plus a companion .mtl file (same name, .mtl extension) that
        drapes the REFERENCE photo over the surface. Open the .obj in MeshLab.
  svg   2:1 isometric projection of the wireframe as an SVG vector image.
        Open in any web browser.
  Aliases: --2d, --3d, --mesh, --svg.

  3D NOTE: Z (height) is a visual heuristic, NOT measured depth:
           Z = brightness / 4 + local stereo disparity * 3.

PARAMETERS
  MODE             Required. One of: 2d, 3d, mesh, svg.
  REFERENCE_IMAGE  Required. First image. Defines the output geometry and is
                   the texture for 'mesh'.
  TARGET_IMAGE     Required. Second image of the same scene. Must have exactly
                   the same width and height as the reference.
  OUTPUT_FILE      Optional. Must NOT already exist: Slimer never overwrites.
                   Defaults (in the current directory):
                     2d -> enhanced_2d.png   3d  -> wireframe_3d.obj
                     mesh -> textured_mesh.obj (+ textured_mesh.mtl)
                     svg  -> topology_3d.svg

INPUT REQUIREMENTS
  - PNG files, 8 bits per channel, non-interlaced.
  - Colour types: grayscale, gray+alpha, RGB, RGBA. Colour is converted to
    brightness (luma); alpha is ignored.
  - Width and height: 8 to 1024 pixels each.
  - Not supported: palette/indexed PNG, 16-bit PNG, interlaced PNG, JPEG and
    other formats. Convert these to 8-bit RGB or grayscale PNG first.

MESHLAB NOTE ('mesh' mode)
  The .mtl refers to the reference photo by FILE NAME ONLY (no directories are
  embedded). Keep the reference PNG in the same directory as the .obj/.mtl.

EXIT STATUS
  0  success (including identity fallback, which is reported as a warning)
  1  command failed
  2  usage error (bad or missing arguments)
  Failures print the stage and a numeric error code (no file paths). Codes are
  also appended to slimer_error_log.txt next to the executable. Transient I/O
  failures are retried up to 3 times. Partial output files are deleted.

EXAMPLES
  slimer 2d   left.png right.png fused.png
  slimer 3d   left.png right.png wireframe.obj
  slimer mesh photo1.png photo2.png terrain.obj
  slimer svg  left.png right.png topology.svg
";

// ============================================================================
// NON-PANICKING TERMINAL OUTPUT
// ============================================================================

/// Writes one line to stdout without the panic that `println!` raises on a closed pipe.
///
/// A terminal-output failure is non-actionable, so the run continues (Tier 2: degrade
/// silently). This returns `()` by design; there is nothing a caller could do with an error.
fn write_stdout_line(arguments: std::fmt::Arguments<'_>) {
    let mut stdout = std::io::stdout().lock();
    if stdout.write_fmt(arguments).is_ok() {
        if let Err(_e) = stdout.write_all(b"\n") {
            // stdout unavailable; continue.
        }
    }
}

/// Same as `write_stdout_line`, for stderr.
fn write_stderr_line(arguments: std::fmt::Arguments<'_>) {
    let mut stderr = std::io::stderr().lock();
    if stderr.write_fmt(arguments).is_ok() {
        if let Err(_e) = stderr.write_all(b"\n") {
            // stderr unavailable; continue.
        }
    }
}

/// Writes a static text block (the help page) to stdout without panicking.
fn write_stdout_text(text: &str) {
    let mut stdout = std::io::stdout().lock();
    if let Err(_e) = stdout.write_all(text.as_bytes()) {
        // stdout unavailable; continue.
    }
}

// ============================================================================
// COMMAND MODEL
// ============================================================================

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
enum ExecutionMode {
    Enhance2D,
    Wireframe3D,
    TexturedMesh3D,
    IsometricSvg,
}

impl ExecutionMode {
    /// Maps the MODE token.
    ///
    /// Unknown tokens are an error. v1 silently fell back to 2D and treated the mode
    /// word as a file path.
    fn from_cli_token(token: &str) -> Result<Self, SlimerError> {
        match token {
            "2d" | "--2d" => Ok(Self::Enhance2D),
            "3d" | "--3d" => Ok(Self::Wireframe3D),
            "mesh" | "--mesh" => Ok(Self::TexturedMesh3D),
            "svg" | "--svg" => Ok(Self::IsometricSvg),
            _ => {
                #[cfg(debug_assertions)]
                eprintln!("CLI-601: unknown mode token {:?}", token);
                Err(SlimerError::CliModeUnknown)
            }
        }
    }

    fn default_output_file_name(self) -> &'static str {
        match self {
            Self::Enhance2D => "enhanced_2d.png",
            Self::Wireframe3D => "wireframe_3d.obj",
            Self::TexturedMesh3D => "textured_mesh.obj",
            Self::IsometricSvg => "topology_3d.svg",
        }
    }

    fn description(self) -> &'static str {
        match self {
            Self::Enhance2D => "2D registered fusion (grayscale .png)",
            Self::Wireframe3D => "3D wireframe lines (.obj)",
            Self::TexturedMesh3D => "3D textured surface mesh (.obj + .mtl)",
            Self::IsometricSvg => "3D isometric vector wireframe (.svg)",
        }
    }
}

/// Where in the command a failure happened (printed as a label, logged as a tag).
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
enum CommandStage {
    ParseArguments,
    OpenInputs,
    Register,
    CreateOutputs,
    WriteMaterial,
    GenerateOutput,
    SyncOutputs,
}

impl CommandStage {
    fn label(self) -> &'static str {
        match self {
            Self::ParseArguments => "parse arguments",
            Self::OpenInputs => "open input images",
            Self::Register => "register images",
            Self::CreateOutputs => "create output files",
            Self::WriteMaterial => "write .mtl material",
            Self::GenerateOutput => "generate output",
            Self::SyncOutputs => "sync outputs to disk",
        }
    }

    /// Lowercase ASCII tag for the error log (validated again by the module).
    fn log_tag(self) -> &'static [u8] {
        match self {
            Self::ParseArguments => b"parse_arguments",
            Self::OpenInputs => b"open_inputs",
            Self::Register => b"register",
            Self::CreateOutputs => b"create_outputs",
            Self::WriteMaterial => b"write_material",
            Self::GenerateOutput => b"generate_output",
            Self::SyncOutputs => b"sync_outputs",
        }
    }
}

/// Immutable, fully validated description of one run.
///
/// All paths are absolute. This is the only state that survives between retry
/// attempts.
#[derive(Debug)]
struct CommandPlan {
    mode: ExecutionMode,
    reference_image_path: PathBuf,
    target_image_path: PathBuf,
    output_path: PathBuf,
    /// Present only in `mesh` mode.
    material_path: Option<PathBuf>,
}

enum CliRequest {
    ShowHelp,
    Execute(CommandPlan),
}

#[derive(Copy, Clone, Debug)]
struct StageFailure {
    stage: CommandStage,
    error: SlimerError,
}

struct CommandReport {
    shift: RegistrationShift,
    used_identity_fallback: bool,
}

/// Which outputs THIS attempt created; only these may be deleted on failure.
#[derive(Default)]
struct CreatedOutputs {
    output_file_created: bool,
    material_file_created: bool,
}

/// Adapter to attach a stage to an error: `.map_err(at_stage(CommandStage::X))`.
fn at_stage(stage: CommandStage) -> impl Fn(SlimerError) -> StageFailure {
    move |error| StageFailure { stage, error }
}

// ============================================================================
// ARGUMENT PARSING
// ============================================================================

/// Resolves a user path to an absolute path (lexically; no filesystem access needed).
fn resolve_absolute_path(raw: &Path) -> Result<PathBuf, SlimerError> {
    if raw.as_os_str().is_empty() {
        return Err(SlimerError::CliArgumentsMissing);
    }
    match std::path::absolute(raw) {
        Ok(absolute) => Ok(absolute),
        Err(_e) => {
            #[cfg(debug_assertions)]
            eprintln!("CLI-210: absolute path resolution failed: {}", _e);
            Err(SlimerError::IoPathResolveFailed)
        }
    }
}

/// Parses `[program, MODE, REFERENCE, TARGET, OUTPUT?]`.
///
/// Rules:
/// - No arguments, or `-h` / `--help` / `help`, shows help.
/// - Unknown modes, missing arguments, and surplus arguments are usage errors.
/// - Output and material paths must not collide with the inputs or each other. This is
///   a lexical check; `create_new` is the definitive guard against clobbering.
fn parse_command_line(arguments: &[OsString]) -> Result<CliRequest, SlimerError> {
    let first_argument = match arguments.get(1) {
        Some(argument) => argument,
        None => return Ok(CliRequest::ShowHelp),
    };
    // A non-UTF-8 mode token cannot match any mode; mapping it to "" rejects it below.
    let first_token = first_argument.to_str().unwrap_or("");
    if matches!(first_token, "-h" | "--help" | "help") {
        return Ok(CliRequest::ShowHelp);
    }
    if arguments.len() > MAX_CLI_ARGUMENTS {
        #[cfg(debug_assertions)]
        eprintln!("CLI-602: more than {} arguments", MAX_CLI_ARGUMENTS - 1);
        return Err(SlimerError::CliTooManyArguments);
    }
    let mode = ExecutionMode::from_cli_token(first_token)?;

    let (reference_raw, target_raw) = match (arguments.get(2), arguments.get(3)) {
        (Some(reference), Some(target)) => (reference, target),
        _ => {
            #[cfg(debug_assertions)]
            eprintln!("CLI-600: REFERENCE and TARGET images are required");
            return Err(SlimerError::CliArgumentsMissing);
        }
    };
    let reference_image_path = resolve_absolute_path(Path::new(reference_raw))?;
    let target_image_path = resolve_absolute_path(Path::new(target_raw))?;
    let output_path = match arguments.get(4) {
        Some(raw) => resolve_absolute_path(Path::new(raw))?,
        None => resolve_absolute_path(Path::new(mode.default_output_file_name()))?,
    };
    let material_path = if mode == ExecutionMode::TexturedMesh3D {
        Some(output_path.with_extension("mtl"))
    } else {
        None
    };

    if output_path == reference_image_path || output_path == target_image_path {
        #[cfg(debug_assertions)]
        eprintln!("CLI-603: output path equals an input path");
        return Err(SlimerError::CliOutputPathConflict);
    }
    if let Some(material) = &material_path {
        if *material == output_path
            || *material == reference_image_path
            || *material == target_image_path
        {
            #[cfg(debug_assertions)]
            eprintln!("CLI-603: .mtl companion path collides with another path");
            return Err(SlimerError::CliOutputPathConflict);
        }
    }

    Ok(CliRequest::Execute(CommandPlan {
        mode,
        reference_image_path,
        target_image_path,
        output_path,
        material_path,
    }))
}

// ============================================================================
// FILE HANDLING
// ============================================================================

fn open_input_file(path: &Path) -> Result<File, SlimerError> {
    match File::open(path) {
        Ok(file) => Ok(file),
        Err(e) => {
            #[cfg(debug_assertions)]
            eprintln!("CLI-OPEN: open failed: {}", e);
            Err(classify_open_error(e.kind()))
        }
    }
}

/// Creates a NEW file. It fails with `IoOutputAlreadyExists` instead of overwriting.
fn create_new_output_file(path: &Path) -> Result<File, SlimerError> {
    match OpenOptions::new().write(true).create_new(true).open(path) {
        Ok(file) => Ok(file),
        Err(e) => {
            #[cfg(debug_assertions)]
            eprintln!("CLI-CREATE: create failed: {}", e);
            Err(classify_create_error(e.kind()))
        }
    }
}

/// Forces written data to stable storage (power-failure resilience).
fn sync_output_file(file: &File) -> Result<(), SlimerError> {
    match file.sync_all() {
        Ok(()) => Ok(()),
        Err(_e) => {
            #[cfg(debug_assertions)]
            eprintln!("CLI-204: sync_all failed: {}", _e);
            Err(SlimerError::IoFlushFailed)
        }
    }
}

/// UTF-8 bytes of a path's final component (used inside OBJ/MTL text).
fn file_name_utf8_bytes(path: &Path) -> Result<&[u8], SlimerError> {
    match path.file_name().and_then(|name| name.to_str()) {
        Some(name) => Ok(name.as_bytes()),
        None => Err(SlimerError::MeshFilenameInvalid),
    }
}

/// Deletes outputs created by a failed attempt, so no half-written file is left that
/// could be mistaken for a valid result.
fn remove_partial_outputs(plan: &CommandPlan, created: &CreatedOutputs) {
    if created.output_file_created {
        if let Err(_e) = std::fs::remove_file(&plan.output_path) {
            write_stderr_line(format_args!(
                "[WARN] A partial output file could not be removed; delete it manually (path shown in the summary above)."
            ));
        }
    }
    if created.material_file_created {
        if let Some(material_path) = &plan.material_path {
            if let Err(_e) = std::fs::remove_file(material_path) {
                write_stderr_line(format_args!(
                    "[WARN] A partial .mtl file could not be removed; delete it manually (path shown in the summary above)."
                ));
            }
        }
    }
}

// ============================================================================
// ERROR LOG & USER-FACING HINTS
// ============================================================================

fn error_log_path() -> Result<PathBuf, SlimerError> {
    let executable = std::env::current_exe().map_err(|_| SlimerError::IoPathResolveFailed)?;
    let directory = executable
        .parent()
        .ok_or(SlimerError::IoPathResolveFailed)?;
    Ok(directory.join(ERROR_LOG_FILE_NAME))
}

/// Appends one terse line (time, code, stage) to the error log.
///
/// Logging is best-effort: a logging failure must never stop the program, so this
/// returns `()`. In debug builds it also prints the detailed diagnostic.
fn record_failure_in_log(failure: StageFailure) {
    #[cfg(debug_assertions)]
    eprintln!(
        "MAIN-debug: stage='{}' error={}",
        failure.stage.label(),
        failure.error
    );

    let log_path = match error_log_path() {
        Ok(path) => path,
        Err(_e) => return,
    };
    match OpenOptions::new().create(true).append(true).open(&log_path) {
        Ok(mut log_file) => {
            if let Err(_e) =
                append_error_log_entry(&mut log_file, failure.error, failure.stage.log_tag())
            {
                #[cfg(debug_assertions)]
                eprintln!("MAIN-debug: error log append failed: {}", _e);
            }
        }
        Err(_e) => {
            #[cfg(debug_assertions)]
            eprintln!("MAIN-debug: error log open failed: {}", _e);
        }
    }
}

/// Static, PII-free guidance for the most common failure codes.
fn failure_hint(error: SlimerError) -> &'static str {
    match error {
        SlimerError::IoInputNotFound => "An input file was not found. Check the path.",
        SlimerError::IoPermissionDenied => {
            "Permission denied reading an input or writing an output."
        }
        SlimerError::IoOutputAlreadyExists => {
            "An output file already exists. Slimer never overwrites: choose a new OUTPUT_FILE name or remove the old file."
        }
        SlimerError::PngMagicInvalid => {
            "An input is not a PNG file. Convert JPEG/other formats to PNG first."
        }
        SlimerError::PngUnsupportedColorType
        | SlimerError::PngUnsupportedBitDepth
        | SlimerError::PngUnsupportedInterlace
        | SlimerError::PngUnsupportedCompression
        | SlimerError::PngUnsupportedCriticalChunk => {
            "Unsupported PNG variant. Use 8-bit, non-interlaced gray/RGB/RGBA PNG (see -h, INPUT REQUIREMENTS)."
        }
        SlimerError::ValIntegrityDimUnderflow | SlimerError::ValIntegrityDimOverflow => {
            "Image width and height must each be 8 to 1024 pixels."
        }
        SlimerError::RegDimMismatch => "Both images must have exactly the same width and height.",
        SlimerError::PngIhdrCrcMismatch
        | SlimerError::PngIdatCrcMismatch
        | SlimerError::PngCorruptChunk
        | SlimerError::PngChunkSequenceInvalid
        | SlimerError::PngZlibHeaderInvalid
        | SlimerError::PngZlibPresetDictionary
        | SlimerError::PngDeflateCorrupt
        | SlimerError::PngHuffmanTableInvalid
        | SlimerError::PngDeflateDistanceTooFar
        | SlimerError::PngScanlineFilterInvalid
        | SlimerError::PngFilterTypeInvalid
        | SlimerError::PngImageDataTruncated
        | SlimerError::IoUnexpectedEof => "An input PNG appears corrupted or truncated.",
        SlimerError::CliModeUnknown => "Unknown MODE. Use one of: 2d, 3d, mesh, svg.",
        SlimerError::CliArgumentsMissing => {
            "Missing arguments: MODE, REFERENCE_IMAGE and TARGET_IMAGE are required."
        }
        SlimerError::CliTooManyArguments => {
            "Too many arguments (at most: MODE REFERENCE TARGET OUTPUT)."
        }
        SlimerError::CliOutputPathConflict => {
            "The output path (or its .mtl companion) collides with an input or with each other."
        }
        SlimerError::MeshFilenameInvalid => {
            "A file name written into the OBJ/MTL is empty, not UTF-8, or contains control characters."
        }
        _ => "The numeric code identifies the exact cause; it is also recorded in the error log.",
    }
}

// ============================================================================
// COMMAND EXECUTION (ONE ATTEMPT)
// ============================================================================

/// Creates and writes the `.mtl` companion for `mesh` mode.
fn write_material_companion(
    plan: &CommandPlan,
    created: &mut CreatedOutputs,
) -> Result<(), StageFailure> {
    let material_path = plan.material_path.as_deref().ok_or(StageFailure {
        stage: CommandStage::WriteMaterial,
        error: SlimerError::ValArgumentInvalid,
    })?;
    let texture_name = file_name_utf8_bytes(&plan.reference_image_path)
        .map_err(at_stage(CommandStage::WriteMaterial))?;
    let mut material_file =
        create_new_output_file(material_path).map_err(at_stage(CommandStage::CreateOutputs))?;
    created.material_file_created = true;
    write_mtl_material_file(&mut material_file, texture_name)
        .map_err(at_stage(CommandStage::WriteMaterial))?;
    sync_output_file(&material_file).map_err(at_stage(CommandStage::SyncOutputs))
}

/// Runs the full command once, from scratch (Tier-3 style re-initialisation).
fn execute_command_once(
    plan: &CommandPlan,
    created: &mut CreatedOutputs,
) -> Result<CommandReport, StageFailure> {
    let mut reference_file =
        open_input_file(&plan.reference_image_path).map_err(at_stage(CommandStage::OpenInputs))?;
    let mut target_file =
        open_input_file(&plan.target_image_path).map_err(at_stage(CommandStage::OpenInputs))?;

    // Tier 2: low contrast degrades to the identity transform instead of failing.
    let (shift, used_identity_fallback) =
        match register_png_images(&mut reference_file, &mut target_file) {
            Ok(shift) => (shift, false),
            Err(error) if error.is_degradable() => (RegistrationShift::identity_fallback(), true),
            Err(error) => {
                return Err(StageFailure {
                    stage: CommandStage::Register,
                    error,
                });
            }
        };

    let mut output_file =
        create_new_output_file(&plan.output_path).map_err(at_stage(CommandStage::CreateOutputs))?;
    created.output_file_created = true;

    let generation = match plan.mode {
        ExecutionMode::Enhance2D => enhance_png_streamed(
            &mut reference_file,
            &mut target_file,
            &shift,
            &mut output_file,
        ),
        ExecutionMode::Wireframe3D => generate_3d_wireframe_obj(
            &mut reference_file,
            &mut target_file,
            &shift,
            &mut output_file,
        ),
        ExecutionMode::IsometricSvg => generate_3d_wireframe_svg(
            &mut reference_file,
            &mut target_file,
            &shift,
            &mut output_file,
        ),
        ExecutionMode::TexturedMesh3D => {
            write_material_companion(plan, created)?;
            let material_path = plan.material_path.as_deref().ok_or(StageFailure {
                stage: CommandStage::GenerateOutput,
                error: SlimerError::ValArgumentInvalid,
            })?;
            let material_name = file_name_utf8_bytes(material_path)
                .map_err(at_stage(CommandStage::GenerateOutput))?;
            generate_3d_textured_mesh_obj(
                &mut reference_file,
                &mut target_file,
                &shift,
                &mut output_file,
                material_name,
            )
        }
    };
    generation.map_err(at_stage(CommandStage::GenerateOutput))?;
    sync_output_file(&output_file).map_err(at_stage(CommandStage::SyncOutputs))?;

    Ok(CommandReport {
        shift,
        used_identity_fallback,
    })
}

// ============================================================================
// REPORTING
// ============================================================================

fn print_plan_summary(plan: &CommandPlan) {
    write_stdout_line(format_args!(
        "------------------------------------------------------------"
    ));
    write_stdout_line(format_args!(
        "Slimer v2.0 | mode: {}",
        plan.mode.description()
    ));
    write_stdout_line(format_args!(
        "Reference image : {}",
        plan.reference_image_path.display()
    ));
    write_stdout_line(format_args!(
        "Target image    : {}",
        plan.target_image_path.display()
    ));
    write_stdout_line(format_args!(
        "Output file     : {}",
        plan.output_path.display()
    ));
    if let Some(material_path) = &plan.material_path {
        write_stdout_line(format_args!(
            "Material file   : {}",
            material_path.display()
        ));
    }
    write_stdout_line(format_args!(
        "------------------------------------------------------------"
    ));
}

fn print_success(plan: &CommandPlan, report: &CommandReport) {
    match (
        report.shift.get_dx(),
        report.shift.get_dy(),
        report.shift.get_confidence(),
    ) {
        (Ok(dx), Ok(dy), Ok(confidence)) => write_stdout_line(format_args!(
            "Registration: dx = {} px, dy = {} px, confidence = {} / 1000000 ppm",
            dx, dy, confidence
        )),
        _ => write_stdout_line(format_args!(
            "Registration: values failed integrity re-check (not shown)"
        )),
    }
    if report.used_identity_fallback {
        write_stderr_line(format_args!(
            "[WARN] Too little contrast to register reliably: used identity fallback (dx = 0, dy = 0)."
        ));
    }
    write_stdout_line(format_args!(
        "[SUCCESS] Output written: {}",
        plan.output_path.display()
    ));
    match plan.mode {
        ExecutionMode::TexturedMesh3D => {
            if let Some(material_path) = &plan.material_path {
                write_stdout_line(format_args!(
                    "[SUCCESS] Material written: {}",
                    material_path.display()
                ));
            }
            write_stdout_line(format_args!(
                "Open the .obj in MeshLab to see the reference photo draped over the surface."
            ));
            if plan.reference_image_path.parent() != plan.output_path.parent() {
                write_stderr_line(format_args!(
                    "[NOTE] The .mtl references the texture by file name only: copy the reference image into the output directory for MeshLab to find it."
                ));
            }
        }
        ExecutionMode::IsometricSvg => {
            write_stdout_line(format_args!("Open the .svg in a web browser."));
        }
        ExecutionMode::Wireframe3D => {
            write_stdout_line(format_args!("Open the .obj in MeshLab or Blender."));
        }
        ExecutionMode::Enhance2D => {}
    }
}

// ============================================================================
// RECOVERY LOOP & ENTRY POINT
// ============================================================================

/// Tier 1 + Tier 3: bounded re-execution of the whole command for retryable codes.
/// Each attempt starts from fresh file handles; partial outputs are removed first.
fn run_with_recovery(plan: &CommandPlan) -> ExitCode {
    for attempt in 1..=MAX_COMMAND_ATTEMPTS {
        let mut created = CreatedOutputs::default();
        match execute_command_once(plan, &mut created) {
            Ok(report) => {
                print_success(plan, &report);
                return ExitCode::SUCCESS;
            }
            Err(failure) => {
                remove_partial_outputs(plan, &created);
                record_failure_in_log(failure);
                if failure.error.is_retryable() && attempt < MAX_COMMAND_ATTEMPTS {
                    write_stderr_line(format_args!(
                        "[RETRY] Transient I/O failure (code {}); starting attempt {} of {}.",
                        failure.error.code(),
                        attempt + 1,
                        MAX_COMMAND_ATTEMPTS
                    ));
                    let backoff =
                        RETRY_BACKOFF_BASE_MILLISECONDS.saturating_mul(u64::from(attempt));
                    std::thread::sleep(Duration::from_millis(backoff));
                    continue;
                }
                write_stderr_line(format_args!(
                    "[FAIL] Stage '{}' failed with code {}. {}",
                    failure.stage.label(),
                    failure.error.code(),
                    failure_hint(failure.error)
                ));
                return ExitCode::from(EXIT_CODE_COMMAND_FAILED);
            }
        }
    }
    ExitCode::from(EXIT_CODE_COMMAND_FAILED)
}

fn main() -> ExitCode {
    // Bounded: keep at most MAX_CLI_ARGUMENTS + 1 entries (the +1 detects "too many").
    let arguments: Vec<OsString> = std::env::args_os().take(MAX_CLI_ARGUMENTS + 1).collect();

    let plan = match parse_command_line(&arguments) {
        Ok(CliRequest::ShowHelp) => {
            write_stdout_text(HELP_TEXT);
            return ExitCode::SUCCESS;
        }
        Ok(CliRequest::Execute(plan)) => plan,
        Err(error) => {
            record_failure_in_log(StageFailure {
                stage: CommandStage::ParseArguments,
                error,
            });
            write_stderr_line(format_args!(
                "[USAGE ERROR] code {}: {}",
                error.code(),
                failure_hint(error)
            ));
            write_stderr_line(format_args!("Run with -h for full usage."));
            return ExitCode::from(EXIT_CODE_USAGE_ERROR);
        }
    };

    print_plan_summary(&plan);
    run_with_recovery(&plan)
}

// ============================================================================
// CARGO TESTS (GATED WITH #[cfg(test)])
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn os_arguments(items: &[&str]) -> Vec<OsString> {
        items.iter().map(|item| OsString::from(*item)).collect()
    }

    fn expect_plan(items: &[&str]) -> CommandPlan {
        match parse_command_line(&os_arguments(items)) {
            Ok(CliRequest::Execute(plan)) => plan,
            Ok(CliRequest::ShowHelp) => panic!("expected a plan, got help"),
            Err(error) => panic!("expected a plan, got error code {}", error.code()),
        }
    }

    #[test]
    fn test_cli_help_requests() {
        assert!(matches!(
            parse_command_line(&os_arguments(&["slimer"])),
            Ok(CliRequest::ShowHelp)
        ));
        for flag in ["-h", "--help", "help"] {
            assert!(matches!(
                parse_command_line(&os_arguments(&["slimer", flag])),
                Ok(CliRequest::ShowHelp)
            ));
        }
        assert!(HELP_TEXT.contains("MODES") && HELP_TEXT.contains("INPUT REQUIREMENTS"));
    }

    #[test]
    fn test_cli_usage_errors() {
        let unknown = parse_command_line(&os_arguments(&["slimer", "3D", "a.png", "b.png"]));
        assert_eq!(unknown.err(), Some(SlimerError::CliModeUnknown));
        let missing = parse_command_line(&os_arguments(&["slimer", "2d", "a.png"]));
        assert_eq!(missing.err(), Some(SlimerError::CliArgumentsMissing));
        let too_many = parse_command_line(&os_arguments(&[
            "slimer", "2d", "a.png", "b.png", "c.png", "d",
        ]));
        assert_eq!(too_many.err(), Some(SlimerError::CliTooManyArguments));
    }

    #[test]
    fn test_cli_defaults_are_absolute_and_mode_specific() {
        let plan = expect_plan(&["slimer", "2d", "a.png", "b.png"]);
        assert_eq!(plan.mode, ExecutionMode::Enhance2D);
        assert!(plan.reference_image_path.is_absolute());
        assert!(plan.target_image_path.is_absolute());
        assert!(plan.output_path.is_absolute());
        assert!(plan.output_path.ends_with("enhanced_2d.png"));
        assert!(plan.material_path.is_none());

        let svg_plan = expect_plan(&["slimer", "--svg", "a.png", "b.png"]);
        assert_eq!(svg_plan.mode, ExecutionMode::IsometricSvg);
        assert!(svg_plan.output_path.ends_with("topology_3d.svg"));
    }

    #[test]
    fn test_cli_mesh_material_path_derived_from_output() {
        let plan = expect_plan(&["slimer", "mesh", "a.png", "b.png", "out/terrain.obj"]);
        assert_eq!(plan.mode, ExecutionMode::TexturedMesh3D);
        let material = plan
            .material_path
            .expect("mesh mode must have a material path");
        assert!(material.is_absolute());
        assert!(material.ends_with("out/terrain.mtl"));
    }

    #[test]
    fn test_cli_rejects_output_path_conflicts() {
        let same_as_input =
            parse_command_line(&os_arguments(&["slimer", "2d", "a.png", "b.png", "a.png"]));
        assert_eq!(
            same_as_input.err(),
            Some(SlimerError::CliOutputPathConflict)
        );
        let material_equals_output = parse_command_line(&os_arguments(&[
            "slimer", "mesh", "a.png", "b.png", "x.mtl",
        ]));
        assert_eq!(
            material_equals_output.err(),
            Some(SlimerError::CliOutputPathConflict)
        );
    }

    #[test]
    fn test_stage_tags_and_hints_are_safe() {
        let stages = [
            CommandStage::ParseArguments,
            CommandStage::OpenInputs,
            CommandStage::Register,
            CommandStage::CreateOutputs,
            CommandStage::WriteMaterial,
            CommandStage::GenerateOutput,
            CommandStage::SyncOutputs,
        ];
        for stage in stages {
            let tag = stage.log_tag();
            assert!(!tag.is_empty());
            assert!(tag.iter().all(|&b| b.is_ascii_lowercase() || b == b'_'));
            assert!(!stage.label().is_empty());
        }
        assert!(failure_hint(SlimerError::IoOutputAlreadyExists).contains("never overwrites"));
        assert!(!failure_hint(SlimerError::RegArithmeticOverflow).is_empty());
    }
}
