"""
Inputs: list of source files
Outputs: saved files in staging as JSON
Logic:
1. for each source file, run a separate ingestion process
2. for source file, get from gcp bucket with data_getter.py
3. for each source file, write contents into raw bucket area with archiver.py
4. for each source file, embed in dataclass with dataclass_constructor.py
5. for each dataclass, write to staging bucket with json_writer.py


TEST:
* You can connect to the source and target buckets
* You can handle empty .SAFEs
* You can handle missing channels in the data
"""
from src.ingestion_etl.world_floods import archiver, data_getter, dataclass_constructor, json_writer
