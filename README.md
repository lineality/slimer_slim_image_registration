#### slimer_slim_image_registration

- https://en.wikipedia.org/wiki/Image_registration

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

### stretch-goals: (maybe infeasible)

1. an iterative version of Phase Correlation (2D FFT) that does not require loading the entire file (e.g. comparing sections)
or sampling?

2. resulting wire-frame-tolology/coordinates from two different video angles streamed in real-time

3. SVG 3D-Wireframe Projection (2D vector) (or some kind of image-over frame)

4. processing video-stream input

5. 16-bit-per-channel (or RGB565) color depth and palette-based (indexed) PNGs with 16 colors

6. N-images is a stretch goal, in case that is a process that can be modularized for a fixed pre-allocated modular process. (or may not be a goal at all, or it the real goal may be image file and image-stream processing... e.g. a kind of N-updates process done linearly?

7. GPU parallelization (where that makes sense for various types of input/use)

#### References: Image Registration
1. Why Anton keeps coming back to C, Wookash Podcast
- https://www.youtube.com/watch?v=7lOupIR1620 

2. Designing and Evaluating Reusable Components - 2004, Casey Muratori
- https://www.youtube.com/watch?v=ZQ5_u8Lgvyk 

3. API Design by Carson Gross ~ BSDC 2025, Montana Programmers
- https://www.youtube.com/watch?v=dTstnhS3moc 


[Alpha MVP-Demo Developed using web-UI for Gemini 3.8 Flash  & Claude]
