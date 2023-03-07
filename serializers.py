from typing import List, Optional
from pydantic import BaseModel

class EventHeader(BaseModel):
    Id: Optional[int]
    UserId: str
    CameraId: int
    Created: str
    Path: str
    IsRequiredObjectDetection: bool
    EventVideoId: Optional[int]

class EventBody(BaseModel):
    Id: Optional[int]
    EventHeaderId: Optional[int]
    Label: int
    Left: int
    Top: int
    Right: int
    Bottom: int

class Event(BaseModel):
    EventHeader: EventHeader
    EventBodies: List[EventBody]
    Error: Optional[str]