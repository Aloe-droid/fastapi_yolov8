from typing import List
from pydantic import BaseModel

class EventHeader(BaseModel):
    Id: int
    UserId: str
    CameraId: int
    Created: str
    Path: str
    IsRequiredObjectDetection: bool
    EventVideoId: int

class EventBody(BaseModel):
    Id: int
    EventHeaderId: int
    Label: int
    Left: int
    Top: int
    Right: int
    Bottom: int

class Event(BaseModel):
    EventHeader: EventHeader
    EventBodies: List[EventBody]
    Error: str