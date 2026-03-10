import os

WORKSPACE_DIR = os.environ.get('REASONRANK_WORKSPACE_DIR', '{YOUR_WORKSPACE_DIR}')
PROJECT_DIR = os.environ.get('REASONRANK_PROJECT_DIR', os.path.dirname(os.path.abspath(__file__)))