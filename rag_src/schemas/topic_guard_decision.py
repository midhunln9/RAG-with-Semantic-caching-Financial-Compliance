from pydantic import BaseModel, Field


class TopicGuardDecision(BaseModel):
    is_on_topic: bool = Field(
        description=(
            "The bool value of whether the prompt is on the topic of "
            "financial compliance or not."
        )
    )
