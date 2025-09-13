# File Scanner Tools

A collection of Python utilities for scanning, analyzing, and managing various file types including CAD files, LiDAR data, and other engineering formats.

## Overview

This project contains three main tools:

1. **CAD Duplicate Cleaner** (`cad_duplicate_cleaner.py`) - Removes duplicate CAD files
2. **LiDAR File Analyzer** (`lidar_file_analyzer.py`) - Analyzes LiDAR point cloud files
3. **File Scanner** (`file_scanner.py`) - General file discovery and reporting tool

## Features

### CAD Duplicate Cleaner
- Identifies and removes duplicate CAD files (.dwg, .dgn)
- Uses MD5 hashing for exact duplicate detection
- Moves duplicates to quarantine folder
- Handles files with numeric suffixes (e.g., `file_1.dwg`, `file_2.dwg`)

### LiDAR File Analyzer
- Scans for LiDAR files (.las, .laz, .e57)
- Analyzes point cloud metadata and properties
- Calculates density, coverage area, and intensity statistics
- Generates detailed CSV reports
- Supports project-based grouping and analysis

### File Scanner
- Discovers files by extension across directory trees
- Generates summary reports with file counts and sizes
- Copies CAD files to specified destination
- Provides directory-level statistics

## Installation

### Prerequisites
- Python 3.7+
- Required packages for LiDAR analysis:
  ```bash
  pip install laspy lazrs
  ```

### Setup
1. Clone or download this repository
2. Install required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Configuration

All tools use environment variables for configuration to avoid hardcoded paths:

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `CAD_DIR` | Directory containing CAD files | `C:\path\to\your\cad\files` |
| `SCAN_ROOT` | Root directory to scan | `D:\\` |
| `PROJECT_ANCHOR` | Base path for project grouping | `D:\Caltrans Construction Project Sync` |
| `GROUP_DEPTH` | Depth for project grouping | `7` |
| `REPORT_CSV` | Output path for LiDAR report | `C:\temp\D_Drive_Laser_Intensity_Report.csv` |
| `DISCOVERY_CSV` | Output path for LiDAR discovery | `C:\temp\D_Drive_Laser_Discovery.csv` |
| `DEST_DIR` | Destination for copied CAD files | `C:\temp\CAD_Files` |

### Setting Environment Variables

**Windows (PowerShell):**
```powershell
$env:CAD_DIR = "C:\Your\CAD\Directory"
$env:SCAN_ROOT = "D:\"
```

**Windows (Command Prompt):**
```cmd
set CAD_DIR=C:\Your\CAD\Directory
set SCAN_ROOT=D:\
```

**Linux/macOS:**
```bash
export CAD_DIR="/path/to/your/cad/files"
export SCAN_ROOT="/path/to/scan"
```

## Usage

### CAD Duplicate Cleaner

```bash
# Use default CAD_DIR from environment
python cad_duplicate_cleaner.py

# Specify directory as argument
python cad_duplicate_cleaner.py "C:\path\to\cad\files"
```

### LiDAR File Analyzer

```bash
# Use environment variables for configuration
python lidar_file_analyzer.py
```

The analyzer will:
- Scan the specified root directory
- Generate two CSV files:
  - Discovery CSV: Detailed file-by-file information
  - Report CSV: Project-level summaries
- Display summary statistics in the console

### File Scanner

```bash
# Use default scan root from environment
python file_scanner.py

# Specify directory to scan
python file_scanner.py "C:\path\to\scan"
```

## Output Files

### LiDAR Analyzer Outputs

1. **Discovery CSV**: Contains detailed information for each LiDAR file:
   - File metadata (size, format, version)
   - Point cloud properties (count, density, bounds)
   - Coordinate system information
   - Intensity statistics
   - Error information

2. **Report CSV**: Contains project-level summaries:
   - File counts by type
   - Total points and coverage area
   - Density statistics
   - Software and system information
   - Classification results

### File Scanner Output

- Console output with file listings and statistics
- Optional CAD file copying to destination directory

## File Types Supported

- **CAD Files**: `.dwg`, `.dgn`
- **LiDAR Files**: `.las`, `.laz`, `.e57`
- **General**: Any file extension can be configured

## Error Handling

- Files that cannot be read are logged with error messages
- Missing dependencies are detected and reported
- Invalid paths are handled gracefully
- Large files are processed in chunks to manage memory usage

## Performance Considerations

- LiDAR analysis uses chunked reading for large files
- MD5 hashing is optimized with configurable chunk sizes
- File scanning uses efficient directory traversal
- Memory usage is controlled through sampling limits

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## License

This project is provided as-is for educational and professional use.

## Troubleshooting

### Common Issues

1. **"laspy not installed"**: Install with `pip install laspy`
2. **"LAZ support missing"**: Install with `pip install lazrs`
3. **Permission errors**: Ensure write access to output directories
4. **Path not found**: Verify environment variables are set correctly

### Getting Help

- Check that all required dependencies are installed
- Verify environment variables are set correctly
- Ensure sufficient disk space for output files
- Check file permissions for source and destination directories

## Examples

### Example 1: Clean CAD Duplicates
```bash
# Set environment variable
$env:CAD_DIR = "C:\Projects\CAD_Files"

# Run cleaner
python cad_duplicate_cleaner.py
```

### Example 2: Analyze LiDAR Data
```bash
# Set configuration
$env:SCAN_ROOT = "D:\LiDAR_Data"
$env:REPORT_CSV = "C:\Reports\LiDAR_Analysis.csv"

# Run analyzer
python lidar_file_analyzer.py
```

### Example 3: Scan and Copy Files
```bash
# Scan specific directory and copy CAD files
python file_scanner.py "E:\Engineering_Data"
```
