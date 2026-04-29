from dataclasses import dataclass

@dataclass
class PineconeConfig:
    index_name : str = "ingestion-open-ai-embedding-small"
    metric : str = "dotproduct"
    cloud : str = "aws"
    region : str = "us-east-1"
    batch_size : int = 200
