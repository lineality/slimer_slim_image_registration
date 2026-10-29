#### slimer_slim_image_registration

```
SLIMER v2.0 - slim, zero-heap image registration (alignment) and fusion

USAGE
  slimer <MODE> <REFERENCE_IMAGE.png> <TARGET_IMAGE.png> [OUTPUT_FILE]
  slimer -h | --help | help
  via cargo:  cargo run --release -- <MODE> <REFERENCE> <TARGET> [OUTPUT]

FUNCTIONALITIES
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
```
