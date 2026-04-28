from dataclasses import dataclass

@dataclass
class PineconeConfig:
    index_name : str = "final-rag-index-openai-small"
    metric : str = "dotproduct"
    cloud : str = "aws"
    region : str = "us-east-1"
