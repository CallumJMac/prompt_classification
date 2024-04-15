# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]
## [1.0.1]
### Changed
- Clean data preprocessing

### Added
- Add second perturbation function ([#202](https://github.com/owner/name/issues/202))

### Removed
- Remove redundant parameters from `method()` method ([`53bd925`](https://github.com/owner/name/commit/53bd925))

### Fixed
- Resolve bug in `method()` method ([`53bd926`](https://github.com/owner/name/commit/53bd926))

## [Released]

## [1.0.0] - 2023-11-29
### Changed
- Rename loss variables

### Added
- Add data poisoning ([`53bd922`](https://github.com/owner/name/commit/53bd922))

### Removed
- **Breaking:** Drop support of PyTorch==2.0.0 ([#222](https://github.com/owner/name/issues/222))

### Fixed
- Resolve bug in `method()` method ([#223](https://github.com/owner/name/issues/223))

## [0.1.0] - 2023-11-28
### Changed
- Upgrade dependencies: PyTorch==2.0.0 --> PyTorch==2.1.0

### Added
- Add data poisoning ([`53bd922`](https://github.com/owner/name/commit/53bd922))

### Removed
- Remove the `method()` method ([#196](https://github.com/owner/name/issues/196))

### Fixed
- Resolve bug in optimization loop ([#197](https://github.com/owner/name/issues/197))

## [0.0.1] - 2023-11-27
### Changed
- Make optimization loop more efficient

### Added
- Add perturbation function ([#199](https://github.com/owner/name/issues/199))

### Removed
- Remove hardcoded parameters in optimization loop ([#200](https://github.com/owner/name/issues/200))

### Fixed
- Resolve bug in `method()` method ([`53bd924`](https://github.com/owner/name/commit/53bd924))

## [0.0.0] - 2023-11-26
### Added
- Add dataset `dataset`
- Add module `module`
- Add optimization functionality
- Support PyTorch and TensorFlow
- Document the `method()` method
