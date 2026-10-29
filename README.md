#### slimer_slim_image_registration
(Very experimental, exploratory, under-construction)

- See: https://en.wikipedia.org/wiki/Image_registration

# Slimer
An 'image registration' module rust crate that \
iterates over sections of files and uses only pre-allocated stack memory \
(no image file fully loaded into memory, no heap) \
with output options for map, 2D enhancement, 3D mesh, SVG-overlay, etc.


#### Help Guide
```
SLIMER - slim, zero-heap image registration (alignment) and fusion

USAGE
  slimer <MODE> <REFERENCE_IMAGE.png> <TARGET_IMAGE.png> [OUTPUT_FILE]
  slimer -h | --help | help
  via cargo:  cargo run --release -- <MODE> <REFERENCE> <TARGET> [OUTPUT]

FUNCTIONALITIES
  1. Streams both PNGs one scanline at a time (never loads whole image).
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
  Aliases: --2d, --3d, --mesh, --svg

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

### Stretch-Goals: (maybe infeasible; exploratory)

1. an iterative version of Phase Correlation (2D FFT) that does not require loading the entire file (e.g. comparing sections)
or sampling?

2. resulting wire-frame-tolology/coordinates from two different video angles streamed in real-time

3. SVG 3D-Wireframe Projection (2D vector) (or some kind of image-over frame)

4. processing video-stream input

5. 16-bit-per-channel (or RGB565) color depth and palette-based (indexed) PNGs with 16 colors

6. N-images is a stretch goal, in case that is a process that can be modularized for a fixed pre-allocated modular process. (or may not be a goal at all, or it the real goal may be image file and image-stream processing... e.g. a kind of N-updates process done linearly?

7. GPU parallelization (where that makes sense for various types of input/use)

8. Explore modular non-expanding-memory ways to use methods that normally do a 'sloppy-power-grab' on memory:
- Not Implemented ZNCC (Zero-Mean Normalized Cross-Correlation) [2D, 3D], lighting
- Not Implemented FAST Corner Detector + Binary Descriptors (e.g., BRIEF/ORB) [2D, Sparse 3D]


### Optional Output Modes: any or all of these
A. A pixel/coordinate map between the images (maybe two items of 2D process is not same as 3D result process)
B. Resulting 2D image file 
C. Resulting 3D image file
D. Process/Issue Details Report(s)


## Possible Use Cases (direct or indirect):
1. multi-sensor resolution-boosting: use a larger N number of very cheap low quality sensors (e.g. cameras) to create a good-enough higher-resolution image (possibly for CCTV type low-rez that is notorious nearly useless for identifying anything in any useful detail: "We can see this grey-blurr entering the camera at 8:37 pm!"

2. Position/Motion Tracking Analysis for Physical Therapy:
- Identifying positions or movements that, e.g. represent indicators or health dangers or health improvements
- Possibly using 3D mesh-data input for classification models rather than color-image-2D photos

3. position/motion analysis on top of other previous layers of processing (not static-map analysis, but in-map-dynamics analysis

4. Classification / Scoring / Detection


## Scope: (planning notes) Possible Parameters
- single-thread vs. parallel
- best-approximation vs. succeed/fail
- segment-stitching or not
- speed vs. precision
- pixel-dimensions of output (e.g. same as input vs. higher than input)
- file format output (jpg?, png?, svg?, bitmap?, other?)


## Similarity Metrics / Matching Kernels
Choice of method (or list of retry-options, e.g. if one method fails, try the next).

1. 1D Projections (Row / Column Collapsing) [2D]

2. SAD / SSD (Sum of Absolute / Squared Differences) [2D, 3D]
- anchors

3. Identity Transform with DegradedConfidence


### Approaches to Registration
- Classic Algorithm
- Machine-learning (from GOFAI to stats to deep etc; trained/unsupervised)
- Granmo-models!


### Separation of Processes:
- image-to-image mapping
- 'Matching Kernel'
- 2D uses of map
- semi-3D uses of map
- 3D uses of map
- use-cases, hardware, software, context
- various optional or combined strategies
- later steps such as mesh-to-mesh comparison


### Separation of 'Modes' & Uses:
(Also see list of Work/Use Context/Cases below)
- best-approximation vs. succeed/fail
- speed vs. precision


### Questions:
1. What are the factors around parallelization, 'hot-loop' sections of code, GPU vs. CPU, stateless (lambda) function calculations (e.g. for parallelization)? Where does it make sense to use GPU or CPU parallelization and where not? (e.g. cost of overhead of setup for parallel-run)

2. In terms of computer vision, there seems to be an obvious disjunction between Post-Deep-Learning-Era image processing vs. 'GOFAI'/normal-math approaches: map?

3. Real-World-Approximation: in many real world cases (either for live-stream processing on an embedded device or visual display for software) it seems that a fast-fudge approach is needed optional direction and different from an 'whatever it takes' resource-cost-policy for medical image analysis that is focused on highest levels of detail and precision regardless of resources.



## Fascinating Intersection of STEM Areas:
- Medical/Health
- Statistics
- Computer-Vision (Systems Programming, Data Science, Machine-Learning)
- Game/Theater/Animation/Art/Literature
- Hardware Software Planning & Design

For example, in two parts, A. imagine asking these people to develop something to perform image registration, then ask the same set of other people which of the resulting code they could use directly or indirectly:


#### Work/Use Context/Cases:
- Academic: R-data-scientist
- Academic: python-sklearn Data-Scientist
- Academic: mathematician
- Academic: Physicist
- Academic: Statistician
- 'Tiny-ML' Data-Scientist
- Academic: Operations Research
- Field-research sensors (might include space probes, academic/commercial)
- Business Analyst (SQL, Power-BI) [Also within 'data science']
- FrontEnd React Developer
- Backend No-SQL Endpoint Developer
- BackEnd Serverless-Function Developer
- Mobile-app game developer
- mobile multi-player game developer
- Mobile native app developer (on device) [android, iOS]
- Mobile cloud app developer (on device) [android, iOS]
- c++ PC-Game +GPU developer
- High Performance GPU developer
- High Performance Mainframe developer
- High Performance OS (IBM-AIX) developer
- Console Game Developer (PS, XBox, etc.)
- Handheld Game Developer (Switch, 
- Therapists including Physical Therapist, Speech Language Pathologist, Occupational Therapist (PT, SLP, OT)
- Medial-imaging: hospital image lab technician/nurse/doctor (x-ray, CT, etc.)
- Physical trainer / Coach
- on-edge arduino developer
- on-edge raspberry-pi developer
- Embedded device code for NASA/Military
- Robotics Lab with computer vision
- Cyber Security
- Self-Driving Car lab
- Deep-Learning Lab
- Diffusion-Model Lab
- Attention-Model Lab (Large Model)
- Attention-Model Lab (Small/Tiny Model)
- Security Devices (cameras, video)
- Municipal and First Responder (Fire, Ambulance, etc.)
- Animator/Digital Artist
- Digital-Twin Simulation
- Risk/Eval Modeling
- Optics Hardware Production
- Specific Hardware Use



### What issues come up when comparing use-contexts?
- Garbage-Collection
- static-analysis and auditability of software
- performance optimization
- maintainability
- resource-use
- language-choice
- hardware-choice




#### References: Image Registration

1. https://en.wikipedia.org/wiki/Image_registration 

2. Why Anton keeps coming back to C, Wookash Podcast
- https://www.youtube.com/watch?v=7lOupIR1620 

3. Designing and Evaluating Reusable Components - 2004, Casey Muratori
- https://www.youtube.com/watch?v=ZQ5_u8Lgvyk 

4. API Design by Carson Gross ~ BSDC 2025, Montana Programmers
- https://www.youtube.com/watch?v=dTstnhS3moc 


#### AI-Use
[Alpha-MVP-Demo code was developed using web-UI review & suggestions for Gemini 3.8 Flash & Claude]
