from pathlib import Path

# Labelbox Access
LABELBOX_API_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1c2VySWQiOiJja3k0MG9pNjkyZjYwMHplcGdlM3o2anpyIiwib3JnYW5pemF0aW9uSWQiOiJja3k0MG9ocXEyZjV6MHplcGVsYjk1N3YyIiwiYXBpS2V5SWQiOiJja3lieXAxdnUwOThnMHpieDNycmdjMThiIiwic2VjcmV0IjoiY2M0YTUzYTA2MmYxMDM4NmY1MWJiNTExM2Q1YTgxYTQiLCJpYXQiOjE2NDIwMTcyODUsImV4cCI6MjI3MzE2OTI4NX0.kXsSSgzrAeFdYdryYgzdok6eiyHydLA88ZP_Pd7EnuQ"
LABELBOX_PROJECT_ID = "cky4nw7aaohqu0zdh6d75gobs"

# Paths
DATASET_PATH = Path("./datasets")
VIDEO_DATA_FILE_NAME = "test-vid.mp4"
ANNOTATIONS_FILE_NAME = "test-annotations.txt"

VIDEO_PATH = str((DATASET_PATH/VIDEO_DATA_FILE_NAME).resolve())
# 