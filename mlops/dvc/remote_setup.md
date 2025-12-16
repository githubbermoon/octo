# DVC Remote Storage Configuration Guide
==========================================

This guide explains how to configure DVC with various remote storage backends
for data and model versioning in the EO Pipeline.

## Overview

DVC (Data Version Control) tracks large files and datasets alongside your Git
repository. Remote storage is used to:
- Store large data files outside of Git
- Share data across team members
- Enable reproducible pipelines

## How DVC Links with Git

```
┌─────────────────────────────────────────────────────────────────┐
│                         Your Project                             │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   Git Repository                    DVC Remote Storage           │
│   ┌─────────────┐                   ┌─────────────────┐         │
│   │ .git/       │                   │ S3 / GCS / etc  │         │
│   │ ├── HEAD    │                   │ ├── 1a2b3c...   │         │
│   │ └── ...     │                   │ ├── 4d5e6f...   │         │
│   │             │     ◄──────────►  │ └── ...         │         │
│   │ data.dvc    │     dvc push/pull │                 │         │
│   │ model.dvc   │                   │ (actual data)   │         │
│   │ (pointers)  │                   │                 │         │
│   └─────────────┘                   └─────────────────┘         │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Key Concepts

1. **`.dvc` files**: Small pointer files that Git tracks, containing MD5 hashes
2. **DVC cache**: Local cache at `.dvc/cache/` storing data by hash
3. **Remote storage**: Cloud storage where data is pushed for sharing

---

## Initial Setup

```bash
# Initialize DVC in your project
dvc init

# This creates:
# .dvc/           - DVC configuration directory
# .dvc/config     - DVC settings
# .dvcignore      - Files to ignore
```

---

## Remote Storage Options

### Option 1: Amazon S3

```bash
# Install S3 support
pip install dvc-s3

# Configure remote
dvc remote add -d storage s3://your-bucket/dvc-storage

# Configure credentials (one of these methods):

# Method A: Environment variables
export AWS_ACCESS_KEY_ID=your_access_key
export AWS_SECRET_ACCESS_KEY=your_secret_key

# Method B: Configure in DVC
dvc remote modify storage access_key_id your_access_key
dvc remote modify storage secret_access_key your_secret_key

# Method C: Use AWS CLI profile
dvc remote modify storage profile your_profile_name
```

### Option 2: Google Cloud Storage

```bash
# Install GCS support
pip install dvc-gs

# Configure remote
dvc remote add -d storage gs://your-bucket/dvc-storage

# Configure credentials
export GOOGLE_APPLICATION_CREDENTIALS=/path/to/credentials.json

# Or use gcloud CLI
gcloud auth application-default login
```

### Option 3: Google Drive (Free for small projects)

```bash
# Install Google Drive support
pip install dvc-gdrive

# Create a folder in Google Drive and get its ID from the URL
# https://drive.google.com/drive/folders/{FOLDER_ID}

# Configure remote
dvc remote add -d storage gdrive://{FOLDER_ID}

# First push will prompt for authentication
dvc push
```

### Option 4: MinIO (Self-hosted S3)

```bash
# Install S3 support
pip install dvc-s3

# Configure for MinIO (included in docker-compose.yml)
dvc remote add -d storage s3://data-bucket/dvc-storage
dvc remote modify storage endpointurl http://localhost:9000
dvc remote modify storage access_key_id minioadmin
dvc remote modify storage secret_access_key minioadmin
```

### Option 5: SSH/SFTP

```bash
# Install SSH support
pip install dvc-ssh

# Configure remote
dvc remote add -d storage ssh://user@server.com/path/to/storage
dvc remote modify storage password your_password
# Or use SSH key authentication
```

---

## Basic Workflow

### Tracking Data

```bash
# Add a file/directory to DVC
dvc add data/raw/sentinel_tiles/

# This creates:
# data/raw/sentinel_tiles.dvc  <- Pointer file (commit to Git)
# .gitignore                    <- Updated to ignore actual data

# Commit the pointer to Git
git add data/raw/sentinel_tiles.dvc data/raw/.gitignore
git commit -m "Add Sentinel-2 tiles to DVC"
```

### Pushing to Remote

```bash
# Push all tracked data to remote
dvc push

# Push specific files
dvc push data/raw/sentinel_tiles.dvc
```

### Pulling Data

```bash
# Pull all data from remote
dvc pull

# Pull specific files
dvc pull data/raw/sentinel_tiles.dvc

# Pull for a specific Git commit
git checkout v1.0.0
dvc pull
```

### Switching Between Versions

```bash
# Checkout a previous version
git checkout v1.0.0
dvc checkout

# Return to latest
git checkout main
dvc checkout
```

---

## Pipeline Integration

DVC pipelines (defined in `dvc.yaml`) automatically:
1. Track dependencies between stages
2. Cache intermediate outputs
3. Skip unchanged stages on re-run

```bash
# Run the full pipeline
dvc repro

# Run specific stage
dvc repro train_lulc

# View pipeline DAG
dvc dag

# View what changed
dvc status
```

---

## Best Practices

### 1. Use `.dvcignore`

```bash
# .dvcignore
# Ignore temporary files
*.tmp
*.log
__pycache__/
*.pyc
```

### 2. Store Remote Config in Git

```bash
# Make remote config public (not credentials!)
dvc remote modify storage --local access_key_id your_key
# The --local flag stores in .dvc/config.local (gitignored)
```

### 3. Use Consistent Naming

```
data/
├── raw/              # Original data (DVC tracked)
├── processed/        # Processed data (DVC tracked)
├── cache/            # Temporary cache (gitignored)
└── external/         # External datasets (DVC tracked)
```

### 4. Document Data Sources

Create a `DATA_SOURCES.md` documenting:
- Where raw data comes from
- How to obtain API credentials
- Data licensing terms

---

## Common Commands

| Command | Description |
|---------|-------------|
| `dvc init` | Initialize DVC |
| `dvc add <path>` | Track file/directory |
| `dvc push` | Upload to remote |
| `dvc pull` | Download from remote |
| `dvc checkout` | Sync workspace with .dvc files |
| `dvc repro` | Run pipeline |
| `dvc dag` | Show pipeline graph |
| `dvc status` | Show changed files |
| `dvc diff` | Show data changes between commits |
| `dvc metrics show` | Show tracked metrics |
| `dvc plots show` | Generate plots |

---

## Troubleshooting

### "No remote storage" error
```bash
# Check remote configuration
dvc remote list
dvc remote default

# Add default remote
dvc remote add -d storage s3://bucket/path
```

### "Access denied" errors
```bash
# Verify credentials
aws s3 ls s3://your-bucket/  # For S3
gsutil ls gs://your-bucket/   # For GCS
```

### Large file upload failures
```bash
# Increase timeout
dvc remote modify storage read_timeout 600

# Use chunked transfer
dvc remote modify storage upload_part_size 104857600  # 100MB chunks
```
