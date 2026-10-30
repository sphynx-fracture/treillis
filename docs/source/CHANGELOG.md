# Change Log
All notable changes to treillis will be documented in this file.
 
The format is based on [Keep a Changelog](http://keepachangelog.com/)
and this project adheres to [Semantic Versioning](http://semver.org/).
 
## [1.4.2] - 2026-10
 
Trying to fix some bugs that impact cross-OS reproducibility and overall lattice generation.
 
### Added

- Added this changelog, sorry for not doing it earlier, and I can't guarantee I will update it consistantly!
- Added `tbb` to the environment.yml so that numba will work reproducibly between Windows and Unix systems 
 
### Changed


### Fixed

- array beam sections should not raise a syntax error anymore when manipulating the lattices

### To-Do

- simplify installation by doing a deep-dive in conda recipes and making one that works from the get-go
- add tomography-like visualization in 3D
- convince colleagues and students to learn Python and git so that I am not the only one maintaining this repo anymore
- add a change_box method to lattices to keep behavior consistent when flattening the boundaries instead of having to update all related parameters by hand
- would be great if I could find a way to encode them properly in some CAD software to play with and print without intersecting faces
- [major] update `fast_display` to the latest Datoviz OR Vispy when stable