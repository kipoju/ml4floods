"""
INPUTS: bucket data
OUTPUTS: bucket data writen into raw bucket area
LOGIC:
1. for every source .SAFE file, create a directory in raw/archive/worldfloods (or something similar) IN DEV TO BEGIN WITH
2. write the contents of the file into the directory
3. if difficult to write, bytearray it and save it

Backlog:
Logging
"""
