from dataclasses import dataclass

@dataclass
class SceneMetadata:
    scene_id: str
    title: str
    description: str
    latitude: float
    longitude: float

@dataclass
class PlaceRecognizeResult:
    metadata: SceneMetadata
    distance: float