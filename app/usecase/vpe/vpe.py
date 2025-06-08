from typing import List, Dict, Any
from app.usecase.video_processor.video_processor import VideoProcessor
from app.usecase.vpr.vpr import VPRSystem
from app.usecase.loader.scene_loader import load_scene_images_by_id
from app.usecase.filter.geometry import is_valid_match


class VPEProcessor:
    def __init__(self, vpr_system: VPRSystem, frame_step: int = 30):
        self.vpr = vpr_system
        self.frame_step = frame_step

    def process_video(self, video_path: str) -> List[Dict[str, Any]]:
        video_processor = VideoProcessor(video_path, self.frame_step)
        seen_coords = set()
        results: List[Dict[str, Any]] = []

        for img, frame_idx in video_processor.frames():
            res = self.vpr.search(img)
            if not res:
                continue

            try:
                scene_images = load_scene_images_by_id(res.metadata.scene_id)
            except FileNotFoundError as e:
                print(f"⚠️ {e}")
                continue

            # Применяем гомографию к каждому изображению сцены
            if not any(is_valid_match(img, ref_img) for ref_img in scene_images):
                continue

            md = res.metadata
            coord_key = (round(md.latitude, 6), round(md.longitude, 6))

            if coord_key in seen_coords:
                continue

            seen_coords.add(coord_key)

            results.append({
                "scene_id": md.scene_id,
                "title": md.title,
                "description": md.description,
                "latitude": md.latitude,
                "longitude": md.longitude,
                "distance": res.distance,
            })

        return results
