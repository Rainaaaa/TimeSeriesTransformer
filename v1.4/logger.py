import sys
import os

class Logger:
    def __init__(self, filename):
        self.filepath = f'./output/{filename}'
        self.original_stdout = sys.stdout

    def __enter__(self):
        self.file = open(self.filepath, 'w')
        sys.stdout = self.file

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout = self.original_stdout
        self.file.close()