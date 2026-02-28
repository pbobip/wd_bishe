import json
import os
import shutil
from pathlib import Path

from PIL import Image, ImageDraw


JSON_DIR = Path(r"c:\Users\pyd111\Desktop\标注3\数据")
IMAGE_DIR = Path(r"c:\Users\pyd111\Desktop\标注3\单晶图像_png")
OUTPUT_DIR = Path(r"c:\Users\pyd111\Desktop\标注3\overlay_preview")

OVERLAY_COLOR = (255, 0, 0)  # red
OVERLAY_ALPHA = 0.4


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def resolve_image_path(image_path: str, json_path: Path) -> Path | None:
    if not image_path:
        return None

    image_path = image_path.strip()
    candidate = Path(image_path)
    if candidate.is_absolute() and candidate.exists():
        return candidate

    filename = Path(image_path).name
    from_image_dir = IMAGE_DIR / filename
    if from_image_dir.exists():
        return from_image_dir

    from_json_dir = json_path.parent / filename
    if from_json_dir.exists():
        return from_json_dir

    return None


def build_overlay(image_path: Path, shapes: list[dict]) -> Image.Image:
    with Image.open(image_path) as im:
        original = im.convert("RGB")
        mask = Image.new("L", original.size, 0)
        draw = ImageDraw.Draw(mask)

        for shape in shapes:
            shape_type = shape.get("shape_type")
            if shape_type not in (None, "polygon"):
                continue
            points = shape.get("points") or []
            if len(points) < 3:
                continue
            draw.polygon([tuple(p) for p in points], fill=255, outline=255)

        color_layer = Image.new("RGB", original.size, OVERLAY_COLOR)
        blended = Image.blend(original, color_layer, OVERLAY_ALPHA)
        result = Image.composite(blended, original, mask)
        return result


def main() -> None:
    ensure_dir(OUTPUT_DIR)

    json_files = sorted(JSON_DIR.glob("*.json"))
    if not json_files:
        print(f"未找到 JSON：{JSON_DIR}")
        return

    missing_images: list[str] = []
    processed = 0

    for json_path in json_files:
        data = load_json(json_path)
        image_path = data.get("imagePath")
        img_path = resolve_image_path(image_path, json_path)
        if img_path is None:
            missing_images.append(f"{json_path.name} -> {image_path}")
            continue

        # 拷贝原图与 JSON 到同一输出目录，方便 labelme 直接编辑
        shutil.copy2(img_path, OUTPUT_DIR / img_path.name)
        shutil.copy2(json_path, OUTPUT_DIR / json_path.name)

        overlay = build_overlay(img_path, data.get("shapes", []))
        overlay_name = f"{img_path.stem}_overlay.png"
        overlay.save(OUTPUT_DIR / overlay_name)

        processed += 1

    print(f"完成叠加：{processed} 个")
    if missing_images:
        print("以下 JSON 未找到对应原图：")
        for item in missing_images:
            print(f" - {item}")


if __name__ == "__main__":
    main()
